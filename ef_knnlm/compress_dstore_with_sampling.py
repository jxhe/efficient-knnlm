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
parser.add_argument('--rate', type=int, default=10,
    help='the compression rate')
parser.add_argument('--tok2pos', type=str,
    help='the pickled dict file that maps token ids to \
    a list of positions which are used to index dstore vectors')
parser.add_argument('--output', type=str,
    help='the output dir')
parser.add_argument('--seed', type=int, default=22)

args = parser.parse_args()

np.random.seed(args.seed)

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

sample_size = len(vec_ids) // args.rate
sampled_vec_ids = np.random.choice(vec_ids, sample_size, replace=False)

# sequential reading from disk
sampled_vec_ids.sort()

centroids_mmap = np.memmap(os.path.join(centroid_dir, f'sample_centroids_tok{args.tokid}_size{sample_size}_dim{args.dim}.npy'),
    dtype=dtype, mode='w+', shape=(sample_size, args.dim))
centroids_mmap[:] = vecs[sampled_vec_ids]

weights_mmap = np.memmap(os.path.join(weight_dir, f'sample_weights_tok{args.tokid}_size{sample_size}.npy'),
    dtype=np.int, mode='w+', shape=(sample_size, 1))
weights_mmap[:] = 1


