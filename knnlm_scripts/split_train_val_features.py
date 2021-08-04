"""
This script reads val and test features and create a new 
train/val split from them
"""

import os



def read_input(input, debug=False):
    hypos = []
    fname = 'features_small.jsonl' if args.debug else input
    with open(fname) as fin:
        for line in fin:
            hypos.append(json.loads(line.strip()))

    return hypos


dataset = wiki

