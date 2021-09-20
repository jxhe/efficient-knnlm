"""traverse all the datastore vectors, delete the ones that
are never hit (excluding itself)
"""

import argparse
import numpy as np
import faiss
import ctypes
import time
import pickle

from collections import defaultdict


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dstore-prefix', type=str,
    help='the dstore vectors')
parser.add_argument('--dstore-size', type=int,
    help='the dstore vectors')
parser.add_argument('--actual-dstore-size', type=int,
    default=None, help='the dstore vectors')
parser.add_argument('--index', type=str,
    help='the faiss index file')
parser.add_argument('--dstore-fp16', default=False,
    action='store_true')
parser.add_argument('--nprobe', type=int, default=32)
parser.add_argument('--dimension', type=int, default=1024)
parser.add_argument('--k', type=int, default=1024,
    help='the number of nearest neighbors')
parser.add_argument('--save', type=str,
    help='the number of nearest neighbors')

# for the purpose of parallel computation
parser.add_argument('--start-point', type=int, default=0,
    help='the starting point to traverse the datastore')
parser.add_argument('--num', type=int, default=1e11,
    help='number of points to traverse')

args = parser.parse_args()

if args.actual_dstore_size is None:
    args.actual_dstore_size = args.dstore_size

print(args)

print(f'shape ({args.dstore_size}, {args.dimension})')

if args.dstore_fp16:
    keys = np.memmap(args.dstore_prefix + '_keys.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_prefix + '_vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))
else:
    keys = np.memmap(args.dstore_prefix + '_keys.npy', dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_prefix + '_vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))

save_size = min(args.num, args.actual_dstore_size - args.start_point)

retrieve =  np.memmap(args.save + f'_size{save_size}_k{args.k}_int32.npy', dtype=np.int32, mode='w+', shape=(save_size, args.k))

index = faiss.read_index(args.index, faiss.IO_FLAG_ONDISK_SAME_DIR)

# from https://github.com/numpy/numpy/issues/13172
# to speed up access to np.memmap
madvise = ctypes.CDLL("libc.so.6").madvise
madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
madvise.restype = ctypes.c_int
assert madvise(keys.ctypes.data, keys.size * keys.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 2 means MADV_SEQUENTIAL

bsz = 3072
batches = []
cnt = 0
offset = 0

# hit = set()
score = defaultdict(float)
t = time.time()

for id_, i in enumerate(range(args.start_point, min(args.start_point + args.num, args.actual_dstore_size))):
    if i % 10000 == 0:
        print(f'processing {i}th entries', flush='True')
        # print(f'there are {len(hit)} entries now')
    batches.append(keys[i])
    cnt += 1

    if cnt % bsz == 0:
        # import pdb; pdb.set_trace()
        dists, knns = index.search(np.array(batches).astype(np.float32), args.k)
        assert knns.shape[0] == bsz

        retrieve[offset:offset + knns.shape[0]] = knns
            # hit.update(knns_sub)

        cnt = 0
        batches = []

        offset += knns.shape[0]

if len(batches) > 0:
    dists, knns = index.search(np.array(batches).astype(np.float32), args.k)

    retrieve[offset:offset + knns.shape[0]] = knns
    # assert knns.shape[0] == bsz

        # hit.update(knns_sub)
# print(f'there are {len(hit)} entries in total currently')

# t = time.time()
# # write out the new index
# new_keys = np.memmap(f'{args.dstore_prefix}_hitreduce_k{args.k}_size{len(hit)}_keys.npy', dtype=np.float16, mode='w+', shape=(len(hit), args.dimension))
# new_vals = np.memmap(f'{args.dstore_prefix}_hitreduce_k{args.k}_size{len(hit)}_vals.npy', dtype=np.int, mode='w+', shape=(len(hit), 1))

# for i, v in enumerate(hit):
#     new_keys[i] = keys[v]
#     new_vals[i] = vals[v]

# print(f'writing index costs {time.time() - t}')

