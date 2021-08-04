"""This script computes and saves a dictionary,
which maps token ids in the vocab to its list of
positions in the datastore files
"""

import os
import argparse
import pickle
import numpy as np

from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--dstore-vals', type=str, help='memmap where vals are stored')
parser.add_argument('--dstore-size', type=int, help='the datastore size')
parser.add_argument('--output-dir', type=str, default='datasets/wikitext-103', help='the output dir')
args = parser.parse_args()


vals = np.memmap(args.dstore_vals, dtype=np.int, mode='r', shape=(args.dstore_size, 1))

res = defaultdict(list)

for i, v in enumerate(vals):
    if i % 1000000 == 0:
        print(f'read {i} tokens')
    res[v[0]].append(i)

print(f'there are {len(res)} entries')

print('start writing to files')

output_file = os.path.join(args.output_dir, 'tok2pos.dict.pickle')
with open(output_file, 'wb') as pf:
    pickle.dump(res, pf)

