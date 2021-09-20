"""this script processes the outputs from
'compress_dstore_with_kmeans.py', and produces
a new compressed datastore
"""

import argparse
import faiss
import pickle
import os
import time
import ctypes

import numpy as np

from collections import Counter

def parse_fname(fname):
    x = fname.split('.')[0].split('_')
    tok = int(x[2].split('tok')[-1])
    size = int(x[3].split('size')[-1])

    return tok, size


parser = argparse.ArgumentParser(description='')
parser.add_argument('--tokid-threshod', type=int,
    help='token ids larger than this threshold do not use kmeans')
parser.add_argument('--dstore-keys', type=str,
    help='the dstore vectors')
parser.add_argument('--dstore-vals', type=str,
    help='the dstore vectors')
parser.add_argument('--dstore-size', type=int,
    help='the dstore size')
parser.add_argument('--new-dstore-size', type=int,
    help='the dstore size')
parser.add_argument('--fp32', action='store_true', default=False,
    help='float 32  vectors in dstore')
parser.add_argument('--dim', type=int, default=1024,
    help='dimensions of vectors')
parser.add_argument('--tok2pos', type=str,
    help='the pickled dict file that maps token ids to \
    a list of positions which are used to index dstore vectors')
parser.add_argument('--kmeans-dir', type=str,
    help='the kmeans dir')
parser.add_argument('--sampling-dir', type=str, default=None,
    help='the sampling dir')
parser.add_argument('--output-dir', type=str, default=None,
    help='the sampling dir')

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

centroid_dir = os.path.join(args.kmeans_dir, 'centroids')
weight_dir = os.path.join(args.kmeans_dir, 'weights')

s_centroid_dir = os.path.join(args.sampling_dir, 'centroids')
s_weight_dir = os.path.join(args.sampling_dir, 'weights')


dtype = np.float16 if not args.fp32 else np.float32
dtype_str = 'fp32' if args.fp32 else 'fp16'

keys = np.memmap(args.dstore_keys, dtype=dtype,
    mode='r', shape=(args.dstore_size, args.dim))
vals = np.memmap(args.dstore_vals, dtype=np.int,
    mode='r', shape=(args.dstore_size, 1))

# from https://github.com/numpy/numpy/issues/13172
# to speed up access to np.memmap
madvise = ctypes.CDLL("libc.so.6").madvise
madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
madvise.restype = ctypes.c_int
assert madvise(keys.ctypes.data, keys.size * keys.dtype.itemsize, 2) == 0, "MADVISE FAILED" # 2 means MADV_SEQUENTIAL

new_keys = np.memmap(os.path.join(args.output_dir,
    f'dstore_kmeans_size{args.new_dstore_size}_embed{args.dim}_{dtype_str}_keys.npy'), dtype=dtype,
    mode='w+', shape=(args.new_dstore_size, args.dim))
new_vals = np.memmap(os.path.join(args.output_dir,
    f'dstore_kmeans_size{args.new_dstore_size}_embed{args.dim}_{dtype_str}_vals.npy'), dtype=np.int,
    mode='w+', shape=(args.new_dstore_size, 1))
new_weights = np.memmap(os.path.join(args.output_dir,
    f'dstore_kmeans_size{args.new_dstore_size}_embed{args.dim}_{dtype_str}_weights.npy'), dtype=np.int,
    mode='w+', shape=(args.new_dstore_size, 1))

tok2size = {}
for fname in os.listdir(os.path.join(args.kmeans_dir, 'centroids')):
    try:
        tok, size = parse_fname(fname)
        tok2size[tok] = size
    except:
        print(f'cannot parse the file name: {fname}')

sampling_tok2size = {}
for fname in os.listdir(os.path.join(args.sampling_dir, 'centroids')):
    try:
        tok, size = parse_fname(fname)
        sampling_tok2size[tok] = size
    except:
        print(f'cannot parse the file name: {fname}')

offset = 0
record = set()
for i, val in enumerate(vals):
    if i % 10000 == 0:
        print(f'processed {i} tokens')

    if val[0] in record:
        continue

    flag = False
    if val[0] <= args.tokid_threshod:
        if val[0] in tok2size:
            size = tok2size[val[0]]
            cent_f = os.path.join(centroid_dir,
             f'kmeans_centroids_tok{val[0]}_size{size}_dim{args.dim}.npy')
            weight_f = os.path.join(weight_dir,
             f'kmeans_weights_tok{val[0]}_size{size}.npy')

            if os.path.exists(cent_f) and os.path.exists(weight_f):
                tmp_keys = np.memmap(cent_f, dtype=dtype, mode='r', shape=(size, args.dim))
                tmp_weight = np.memmap(weight_f, dtype=np.int, mode='r', shape=(size, 1))

                new_keys[offset:offset+size] = tmp_keys
                new_vals[offset:offset+size] = val[0]
                new_weights[offset:offset+size] = tmp_weight

                record.update([val[0]])

                offset += size
                flag = True
            else:
                flag = False

        # use sampling for the first several words (temporary solutions)
        if not flag and val[0] <= 10:
            size = sampling_tok2size[val[0]]
            cent_f = os.path.join(s_centroid_dir,
             f'sample_centroids_tok{val[0]}_size{size}_dim{args.dim}.npy')
            weight_f = os.path.join(s_weight_dir,
             f'sample_weights_tok{val[0]}_size{size}.npy')

            if os.path.exists(cent_f) and os.path.exists(weight_f):
                print(f'use sampling centroids: {cent_f}')
                tmp_keys = np.memmap(cent_f, dtype=dtype, mode='r', shape=(size, args.dim))
                tmp_weight = np.memmap(weight_f, dtype=np.int, mode='r', shape=(size, 1))

                new_keys[offset:offset+size] = tmp_keys
                new_vals[offset:offset+size] = val[0]
                new_weights[offset:offset+size] = tmp_weight

                record.update([val[0]])

                offset += size
                flag = True

    # infrequent words and some corner cases
    if not flag:
        new_keys[offset] = keys[i]
        new_vals[offset] = val[0]
        new_weights[offset] = 1
        offset += 1

print(f'there are {offset} entries in the new datastore')

