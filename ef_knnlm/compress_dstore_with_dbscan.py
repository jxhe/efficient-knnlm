"""this script performs kmeans for all the vectors
corresponding to the same token
"""

import argparse
import faiss
import pickle
import os
import time
import ctypes

import numpy as np

from collections import Counter, defaultdict

parser = argparse.ArgumentParser(description='')
parser.add_argument('--tokid', type=int,
    help='the token id')
parser.add_argument('--dstore-keys', type=str,
    help='the dstore vectors')
parser.add_argument('--dstore-size', type=int,
    help='the dstore size')
parser.add_argument('--fp32', action='store_true', default=False,
    help='float 32  vectors in dstore')
parser.add_argument('--dim', type=int, default=1024,
    help='dimensions of vectors')
parser.add_argument('--tok2pos', type=str,
    help='the pickled dict file that maps token ids to \
    a list of positions which are used to index dstore vectors')
parser.add_argument('--output', type=str,
    help='the output dir')
# parser.add_argument('--seed', type=int, default=22)

#DBSCAN hyperparameters
parser.add_argument('--minpts', type=int, default=3)
parser.add_argument('--eps', type=float, default=5)

args = parser.parse_args()

centroid_dir = os.path.join(args.output, f'centroids_minpts{args.minpts}_eps{args.eps}')
weight_dir = os.path.join(args.output, f'weights_minpts{args.minpts}_eps{args.eps}')
# variance_dir = os.path.join(args.output, 'variance')

os.makedirs(centroid_dir, exist_ok=True)
os.makedirs(weight_dir, exist_ok=True)

dtype = np.float16 if not args.fp32 else np.float32

vecs = np.memmap(args.dstore_keys, dtype=dtype,
    mode='r', shape=(args.dstore_size, args.dim))


# from https://github.com/numpy/numpy/issues/13172
# to speed up access to np.memmap
madvise = ctypes.CDLL("libc.so.6").madvise
madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
madvise.restype = ctypes.c_int
assert madvise(vecs.ctypes.data, vecs.size * vecs.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 1 means MADV_RANDOM

with open(args.tok2pos, 'rb') as fin:
    tok2pos = pickle.load(fin)

vec_ids = tok2pos[args.tokid]
print(f'there are {len(vec_ids)} vectors to cluster')

ttime = time.time()
# import pdb; pdb.set_trace()
x = vecs[vec_ids]
# x = []
# for i, id_ in enumerate(vec_ids):
#     if i % 5000 == 0:
#         print(f'moved {i} records to memory')
#     x.append(vecs[id_].astype(np.float32))

# x = np.array(x).astype(np.float32)

print(f'access finishes, took {time.time() - ttime} seconds', flush=True)
x = x.astype(np.float32)


index = faiss.IndexFlatL2(args.dim)

if len(vec_ids) >= 10000:
    nlist = 1024 if len(vec_ids) >= 1000000 else 128
    index = faiss.IndexIVFFlat(index, args.dim, nlist)
    random_sample = np.random.choice(np.arange(len(vec_ids)), size=[min(1000000, len(vec_ids))], replace=False)
    print('training index begins', flush=True)
    ttime = time.time()
    index.train(x[random_sample])
    print(f'training finishes, took {time.time() - ttime} seconds', flush=True)
    index.add(x)
else:
    index.add(x)

index.nprobe = 8

print('building index finishes', flush=True)


# DBSCAN
# followling pseudocode from https://en.wikipedia.org/wiki/DBSCAN#cite_note-dbscan-1
def range_query(query, index, eps, k, bsz_ids):
    # import pdb; pdb.set_trace()
    ids = []

    # (bsz, k)
    D, I = index.search(query, k)
    select = (D <= eps)
    for i, (select_i, id_i, self_id) in enumerate(zip(select, I, bsz_ids)):
        # avoid retrieving itself
        not_retrieve_self = (id_i != self_id)
        select_i = np.logical_and(select_i, not_retrieve_self)
        ids.append(id_i[select_i])

    return ids


labels = {}
bsz = 128
pointer = 0
db_len = len(vec_ids)
k = 128
min_pts = args.minpts
cluster_id = -1
ttime = time.time()
while pointer < db_len:
    x_batch = []
    bsz_ids = []
    cnt = 0
    while cnt < bsz and pointer < db_len:
        # skip previously processed ones
        if pointer not in labels:
            bsz_ids.append(pointer)
            cnt += 1

        pointer += 1

    x_batch = x[bsz_ids]

    # import pdb; pdb.set_trace()
    neighbor_ids = range_query(x_batch, index, args.eps, k, bsz_ids)
    for i, (xi, neighbor_i, cur_id) in enumerate(zip(x_batch, neighbor_ids, bsz_ids)):
        if len(neighbor_i) < min_pts:
            # outlier points
            labels[cur_id] = -1
            continue

        cluster_id += 1
        labels[cur_id] = cluster_id
        neighbors = set(neighbor_i.tolist())
        while len(neighbors) > 0:
            neighbor_ij = neighbors.pop()

            if neighbor_ij in labels:
                if labels[neighbor_ij] == -1:
                    labels[neighbor_ij] = cluster_id

                continue

            labels[neighbor_ij] = cluster_id
            neighbor_of_neighbor = range_query(np.expand_dims(x[neighbor_ij], axis=0), index, args.eps, k, [neighbor_ij])
            neighbor_of_neighbor = neighbor_of_neighbor[0]

            if len(neighbor_of_neighbor) >= min_pts:
                neighbors = neighbors.union(set(neighbor_of_neighbor.tolist()))


print(f'DBSCAN clustering finishes, there are {cluster_id + 1} clusters, took {time.time() - ttime} seconds')

label2id = defaultdict(list)
for k,v in labels.items():
    label2id[v].append(k)

print(f'there are {len(label2id)} unique labels')
assert (len(label2id) == cluster_id + 1 or len(label2id) == cluster_id + 2)

size = cluster_id + 1 + len(label2id[-1])

print(f'the new dstore size is {size}')

centroids_mmap = np.memmap(os.path.join(centroid_dir, f'dbscan_centroids_tok{args.tokid}_size{size}_dim{args.dim}.npy'),
    dtype=dtype, mode='w+', shape=(size, args.dim))

weights_mmap = np.memmap(os.path.join(weight_dir, f'dbscan_weights_tok{args.tokid}_size{size}.npy'),
    dtype=np.int, mode='w+', shape=(size, 1))

# variance_mmap = np.memmap(os.path.join(variance, f'dbscan_variance_tok{args.tokid}_size{size}_dim{args.dim}.npy'),
    # dtype=dtype, mode='w+', shape=(size, args.dim))

# variance = np.memmap

offset = 0
for k, v in label2id.items():
    # outliers
    if k == -1:
        xi = x[v]
        centroids_mmap[offset:offset + len(xi)] = xi
        weights_mmap[offset:offset + len(xi)] = 1
        # variance_mmap[offset:offset + len(xi)] = 1
        offset += len(xi)
    else:
        xi = x[v]
        centroids_mmap[offset] = xi.mean(axis=0)
        weights_mmap[offset] = len(xi)
        # variance_mmap[offset] = xi.std(axis=0)
        offset += 1
