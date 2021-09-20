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

from collections import Counter

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

args = parser.parse_args()

centroid_dir = os.path.join(args.output, 'centroids')
weight_dir = os.path.join(args.output, 'weights')

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
x = vecs[vec_ids]
print(f'access finishes, took {time.time() - ttime} seconds', flush=True)
x = x.astype(np.float32)

# compress 10 times
if len(vec_ids) >= 1000000:
    ncluster = len(vec_ids) // 20
    niter = 20
else:
    ncluster = len(vec_ids) // 20
    niter = 30

kmeans = faiss.Kmeans(args.dim, ncluster, niter=niter, verbose=True)
kmeans.train(x)

print('training finishes')

centroids = kmeans.centroids

centroids_mmap = np.memmap(os.path.join(centroid_dir, f'kmeans_centroids_tok{args.tokid}_size{centroids.shape[0]}_dim{centroids.shape[1]}.npy'),
    dtype=dtype, mode='w+', shape=centroids.shape)
centroids_mmap[:] = centroids.astype(dtype)

weights_mmap = np.memmap(os.path.join(weight_dir, f'kmeans_weights_tok{args.tokid}_size{centroids.shape[0]}.npy'),
    dtype=np.int, mode='w+', shape=(centroids.shape[0], 1))

D, I = kmeans.index.search(x, 1)

cnt = Counter()
for i in I:
    cnt[i[0]] += 1

for i in range(centroids.shape[0]):
    weights_mmap[i,0] = cnt[i]

