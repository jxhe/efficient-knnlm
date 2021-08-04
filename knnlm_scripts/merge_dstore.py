"""merge different dstore files. This script is used to
post-process the dstore files produced by parallel computing
"""

import os
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='merge different datastore files')
parser.add_argument('--dir', type=str, default=None, help='the datastore directory')

args = parser.parse_args()
embed_files = [f for f in os.listdir(args.dir) if 'embed' in f and f.endswith('npy')]
val_files = [f for f in os.listdir(args.dir) if 'vals' in f and f.endswith('npy')]

id2embed = {int(f.split('_')[-1].split('.')[0].split('id')[-1]): f for f in embed_files}
id2val = {int(f.split('_')[-1].split('.')[0].split('id')[-1]): f for f in val_files}
id2size = {}

for id_, f in id2embed.items():
    size = int(f.split('_')[1].split('size')[-1])
    id2size[id_] = size


dtype = id2embed[0].split('_')[-2]
embed_size = int(embed_files[0].split('_')[2].split('embed')[-1])

total_size = sum(id2size.values())


dstore_keys = np.memmap(os.path.join(args.dir, f'dstore_size{total_size}_embed{embed_size}_keys_{dtype}.npy'),
    dtype=np.float16 if dtype=='fp16' else np.float32,
    mode='w+',
    shape=(total_size, embed_size))

dstore_vals = np.memmap(os.path.join(args.dir, f'dstore_size{total_size}_vals.npy'),
    dtype=np.int16 if dtype=='fp16' else np.int,
    mode='w+',
    shape=(total_size, 1))

cur = 0
for id_ in sorted(id2embed):
    size = id2size[id_]
    keys = np.memmap(os.path.join(args.dir, id2embed[id_]),
        dtype=np.float16 if dtype=='fp16' else np.int,
        mode='r',
        shape=(size, embed_size))

    vals = np.memmap(os.path.join(args.dir, id2val[id_]),
        dtype=np.int16 if dtype=='fp16' else np.int,
        mode='r',
        shape=(size, 1))

    dstore_keys[cur:cur+size] = keys
    dstore_vals[cur:cur+size] = vals

    cur += size

