"""this script randoms samples a subset of the datastore
to form a new datastore
"""

import argparse
import faiss
import pickle
import os
import ctypes

import numpy as np

from collections import Counter

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dstore-keys', type=str,
    help='the dstore vectors')
parser.add_argument('--dstore-vals', type=str,
    help='the dstore vectors')
parser.add_argument('--dstore-size', type=int,
    help='the dstore size')
parser.add_argument('--fp32', action='store_true', default=False,
    help='float 32  vectors in dstore')
parser.add_argument('--dim', type=int, default=1024,
    help='dimensions of vectors')
parser.add_argument('--num', type=int, default=0,
    help='size of the new datastore')
parser.add_argument('--fraction', type=float, default=0,
    help='fraction of the compression')
parser.add_argument('--output-dir', type=str,
    help='the output dir')
parser.add_argument('--seed', type=int, default=22)

args = parser.parse_args()

np.random.seed(args.seed)

os.makedirs(args.output_dir, exist_ok=True)

dtype = np.float16 if not args.fp32 else np.float32

vecs = np.memmap(args.dstore_keys, dtype=dtype,
    mode='r', shape=(args.dstore_size, args.dim))
vals = np.memmap(args.dstore_vals, dtype=np.int,
    mode='r', shape=(args.dstore_size, 1))

# from https://github.com/numpy/numpy/issues/13172
# to speed up access to np.memmap
madvise = ctypes.CDLL("libc.so.6").madvise
madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
madvise.restype = ctypes.c_int
assert madvise(vecs.ctypes.data, vecs.size * vecs.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 1 means MADV_RANDOM

if args.num == 0:
    args.num = round(len(vecs) * args.fraction)

sampled_vec_ids = np.random.choice(range(len(vecs)), args.num, replace=False)

# sequential reading from disk
sampled_vec_ids.sort()

new_keys = np.memmap(os.path.join(args.output_dir, f'sample_dstore_size{args.num}_dim{args.dim}_keys.npy'),
    dtype=dtype, mode='w+', shape=(args.num, args.dim))
new_vals = np.memmap(os.path.join(args.output_dir, f'sample_dstore_size{args.num}_dim{args.dim}_vals.npy'),
    dtype=np.int, mode='w+', shape=(args.num, 1))

print('start writing')
for i, v in enumerate(sampled_vec_ids):
    if i % 1000000 == 0:
        print(f'writing {i} tokens')
    new_keys[i] = vecs[v]
    new_vals[i] = vals[v]

