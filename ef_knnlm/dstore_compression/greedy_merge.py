"""traverse all the datastore vectors, delete the ones that
are never hit (excluding itself)
"""

import argparse
import numpy as np
import faiss
import ctypes
import time
import pickle
import os
import random

from multiprocessing import Pool
from collections import defaultdict


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dstore-prefix', type=str,
    help='the dstore vectors')
parser.add_argument('--dstore-size', type=int,
    help='the dstore vectors')
parser.add_argument('--retrieval-dir', type=str,
    help='the directory that saves retrieval results')
parser.add_argument('--save-dir', type=str,
    help='the directory that saves retrieval results')
parser.add_argument('--actual-dstore-size', type=int,
    default=None, help='the dstore vectors')
parser.add_argument('--dstore-fp16', default=False,
    action='store_true')
# parser.add_argument('--nprobe', type=int, default=32)
parser.add_argument('--dimension', type=int, default=1024)
parser.add_argument('--k', type=int, default=5,
    help='the number of nearest neighbors to probe')
parser.add_argument('--seed', type=int, default=22,
    help='seed')

# for the purpose of parallel computation
# parser.add_argument('--start-point', type=int, default=0,
    # help='the starting point to traverse the datastore')
# parser.add_argument('--num', type=int, default=10000,
#     help='number of points to traverse')

args = parser.parse_args()

random.seed(args.seed)

if args.actual_dstore_size is None:
    args.actual_dstore_size = args.dstore_size

print(args)

if args.dstore_fp16:
    keys = np.memmap(args.dstore_prefix + '_keys.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_prefix + '_vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))
else:
    keys = np.memmap(args.dstore_keys, dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_vals, dtype=np.int, mode='r', shape=(args.dstore_size, 1))


def parse_retrieve_fname(fname):
    offset = size = nk = None
    x = fname.split('_')
    for s in x:
        if s.startswith('start'):
            offset = int(s.split('start')[-1])

        if s.startswith('size'):
            size = int(s.split('size')[-1])

        if s.startswith('k'):
            nk = int(s.split('k')[-1])


    if offset is not None and size is not None and nk is not None:
        return offset, size, nk

    raise ValueError(f"parsing error for {fname}")

weights = np.ones(args.dstore_size, dtype=np.int)

def merge_knn(fname):
    print(f'start processing {fname}', flush=True)
    offset, size, nk = parse_retrieve_fname(fname)
    print(f'offset: {offset}, size{size}, k{nk}', flush=True)
    scores = np.zeros(args.dstore_size, dtype=np.float32)

    ret = np.memmap(os.path.join(args.retrieval_dir, fname), dtype=np.int32, mode='r', shape=(size, nk))

    t = time.time()
    ret_mem = np.zeros((size, args.k+1), dtype=np.int32)
    ret_mem[:] = ret[:, :args.k+1]
    print(f'reading index into memory costs {time.time() - t} seconds', flush=True)

    # traverse with random order
    random_order = list(range(size))
    random.shuffle(random_order)

    # import pdb; pdb.set_trace()
    for i, id_ in enumerate(random_order):
        if i % 100000 == 0:
            print(f'processing {i} rows', flush=True)
            # break
        cur_id = offset + id_

        # already removed
        if weights[cur_id] <= 0:
            continue

        # import pdb; pdb.set_trace()
        for k, v in enumerate(ret_mem[id_]):
            if cur_id != v and weights[v] == 1 and vals[v] == vals[cur_id]:
                # select one to drop
                weights[v] = 0
                weights[cur_id] += 1

        # if i == 50000:
        #     break

    del ret_mem


# from https://github.com/numpy/numpy/issues/13172
# to speed up access to np.memmap
madvise = ctypes.CDLL("libc.so.6").madvise
madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
madvise.restype = ctypes.c_int
assert madvise(keys.ctypes.data, keys.size * keys.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 2 means MADV_SEQUENTIAL

fnames = []

for f in os.listdir(args.retrieval_dir):
    if f.startswith('retrieve') and f.endswith('npy'):
        fnames.append(f)

# import pdb; pdb.set_trace()
random.shuffle(fnames)
for fname in fnames:
    merge_knn(fname)

num = (weights > 0).sum()
print(f'the new datastore has {num} entries')

new_key = np.memmap(os.path.join(args.save_dir,
        f'dstore_merge{args.k}_size{num}_embed{args.dimension}_fp16_keys.npy'), mode='w+', dtype=np.float16, shape=(num, args.dimension))
new_val = np.memmap(os.path.join(args.save_dir,
        f'dstore_merge{args.k}_size{num}_embed{args.dimension}_fp16_vals.npy'), mode='w+', dtype=np.int, shape=(num, 1))
new_weight = np.memmap(os.path.join(args.save_dir,
        f'dstore_merge{args.k}_size{num}_embed{args.dimension}_fp16_weights.npy'), mode='w+', dtype=np.int, shape=(num, 1))

cnt = 0
for i, v in enumerate(weights):
    if i % 500000 == 0:
        print(f'writing {i} tokens', flush=True)

    if v > 0:
        new_key[cnt] = keys[i]
        new_val[cnt] = vals[i]
        new_weight[cnt] = v
        cnt += 1

