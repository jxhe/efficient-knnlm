"""cache statistics required to compute frequency and fertility features
"""

import os
import argparse
import json
import time
import random
import pickle
import numpy as np

from fairseq.data import Dictionary
from collections import Counter, defaultdict

# from sklearn.preprocessing import StandardScaler
from datasets import load_dataset

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', type=str, default='examples/language_model/wikitext-103/wiki.train.tokens',
    help='the text file used to compute the statistics')
parser.add_argument('--cache', type=str, default='datasets/wikitext-103', help='the frequency cache dir')
parser.add_argument('--overwrite', action='store_true', default=False,
    help='overwrite existing cache files')
parser.add_argument('--dict-path', type=str, default=None,
    help='if specified, keys are stored as token ids')
parser.add_argument('--csize', type=int, default=1,
    help='context size when computing context frequency/fertility')

args = parser.parse_args()


# do not perform scaling over context
def get_ngram_freq(file, ngram=4, dictionary=None):
    res = Counter()
    prev = ['</s>'] * ngram if dictionary is None else [dictionary.index('</s')] * ngram
    with open(file) as fin:
        for i, line in enumerate(fin):
            if i % 100000 == 0:
                print(f'procesed {i} lines')
            for tok in line.strip().split():
                prev = prev[-ngram:]
                for j in range(max(ngram-1, 1), ngram+1):
                    if dictionary is None:
                        res[' '.join(prev[-j:])] += 1
                    else:
                        res[tuple(prev[-j:])] += 1

                prev.append(tok if dictionary is None else dictionary.index(tok))

            prev.append('</s>' if dictionary is None else dictionary.index('</s>'))

    return res

def get_ngram_fertility(file, ngram=4, dictionary=None):
    """compute the next word fertility of the context, which is
    the number of unique words following this context
    """
    res = defaultdict(set)
    prev = ['</s>'] * ngram if dictionary is None else [dictionary.index('</s')] * ngram
    with open(file) as fin:
        for i, line in enumerate(fin):
            if i % 100000 == 0:
                print(f'procesed {i} lines')
            for tok in line.strip().split():
                prev = prev[-ngram:]
                for j in range(max(ngram-1, 1), ngram+1):
                    if dictionary is None:
                        res[' '.join(prev[-j:])].update([tok])
                    else:
                        res[tuple(prev[-j:])].update([tok])
                        # res[tuple(prev[-j:])].update([dictionary.index(tok)])
                prev.append(tok if dictionary is None else dictionary.index(tok))

            prev.append('</s>' if dictionary is None else dictionary.index('</s>'))

    return Counter({key: len(res[key]) for key in res})


tp = time.time()

if args.dict_path is not None:
    freq_path = f'freq_cache_id.pickle'
    fert_path = f'fertility_cache_id.pickle'
    dictionary = Dictionary.load(args.dict_path)
else:
    freq_path = 'freq_cache.pickle'
    fert_path = 'fertility_cache.pickle'
    dictionary = None

freq_cache = os.path.join(args.cache, freq_path)
fertility_cache = os.path.join(args.cache, fert_path)

if not args.overwrite and os.path.isfile(freq_cache):
    print('skip freq cache files since they exist')
else:
    print('compute freq statistics')
    freq_cnt = get_ngram_freq(args.data, ngram=args.csize, dictionary=dictionary)
    if dictionary is not None:
        freq_cnt = Counter({k:np.log(v + 1) for k,v in freq_cnt.items()})
    with open(freq_cache, 'wb') as pf:
        pickle.dump(freq_cnt, pf)


if not args.overwrite and os.path.isfile(fertility_cache):
    print('skip fertility cache files since they exist')
else:
    print('compute fertility statistics')
    fertility_cnt = get_ngram_fertility(args.data, ngram=args.csize, dictionary=dictionary)
    if dictionary is not None:
        fertility_cnt = Counter({k:np.log(v + 1) for k,v in fertility_cnt.items()})
    with open(fertility_cache, 'wb') as pf:
        pickle.dump(fertility_cnt, pf)
