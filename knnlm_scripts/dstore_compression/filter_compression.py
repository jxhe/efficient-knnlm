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

from multiprocessing import Pool
from collections import defaultdict


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dstore-prefix', type=str,
    help='the dstore vectors')
parser.add_argument('--dstore-size', type=int,
    help='the dstore vectors')
parser.add_argument('--retrieval-dir', type=str,
    help='the directory that saves retrieval results')
parser.add_argument('--actual-dstore-size', type=int,
    default=None, help='the dstore vectors')
parser.add_argument('--dstore-fp16', default=False,
    action='store_true')
# parser.add_argument('--nprobe', type=int, default=32)
parser.add_argument('--dimension', type=int, default=1024)
parser.add_argument('--save', type=str,
    help='the number of nearest neighbors')

# for the purpose of parallel computation
# parser.add_argument('--start-point', type=int, default=0,
    # help='the starting point to traverse the datastore')
# parser.add_argument('--num', type=int, default=10000,
#     help='number of points to traverse')

args = parser.parse_args()

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

def get_score(fname):
    print(f'start processing {fname}', flush=True)
    offset, size, nk = parse_retrieve_fname(fname)
    print(f'offset: {offset}, size{size}, k{nk}', flush=True)
    scores = np.zeros(args.dstore_size, dtype=np.float32)

    ret = np.memmap(os.path.join(args.retrieval_dir, fname), dtype=np.int32, mode='r', shape=(size, nk))

    # import pdb; pdb.set_trace()
    for i, row in enumerate(ret):
        if i % 100000 == 0:
            print(f'processing {i} rows', flush=True)
            # break
        scores[row] = scores[row] + 1. / (np.arange(len(row)) + 1)
        # if i == 50000:
        #     break

    return scores

fnames = []

for f in os.listdir(args.retrieval_dir):
    if f.startswith('retrieve') and f.endswith('npy'):
        fnames.append(f)

# import pdb; pdb.set_trace()

print(f'starint {len(fnames)} processes')
with Pool(len(fnames)) as p:
    scores_list = p.map(get_score, fnames)

# get_score(fnames[0])

# import pdb; pdb.set_trace()
scores = sum(scores_list)

sorted_ids = np.argsort(scores)[::-1]

del scores

print('start writing the new datastore', flush=True)

fraction_list = [0.2, 0.4, 0.6, 0.8]

# assert len(sorted_score) == args.actual_dstore_size
# print(f'sorted score len: {len(sorted_ids)}')

def write_dstore(frac):
    num = round(len(sorted_ids) * frac)
    ids = sorted(sorted_ids[:num])
    new_key = np.memmap(os.path.join(args.retrieval_dir,
            f'dstore_scorefilter_size{num}_embed{args.dimension}_keys.npy'), mode='w+', dtype=np.float16, shape=(num, args.dimension))
    new_val = np.memmap(os.path.join(args.retrieval_dir,
            f'dstore_scorefilter_size{num}_embed{args.dimension}_vals.npy'), mode='w+', dtype=np.int, shape=(num, 1))

    print(f'writing fraction {frac}', flush=True)
    for i, id_ in enumerate(ids):
        if i % 500000 == 0:
            print(f'writing {i} tokens')
        new_key[i] = keys[id_]
        new_val[i] = vals[id_]

with Pool(len(fraction_list)) as p:
    _ = p.map(write_dstore, fraction_list)

