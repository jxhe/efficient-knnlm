"""perform interpolation with the KN ngram LM,
the ngram LM is from the implementation kenlm:
https://github.com/kpu/kenlm
"""

import argparse
import numpy as np
from scipy.special import logsumexp


from datasets import load_dataset


def read_nlm_score(fname):
    dataset = load_dataset('json', data_files=fname, cache_dir='hf_cache', use_threads=True)

    return dataset['train']

def read_kenlm_score(fname):
    with open(fname) as fin:
        for line in fin:
            word_list = line.strip().split('\t')
            for wv in word_list[:-1]:
                try:
                    wv_s = wv.split()
                    w = '='.join(wv_s[0].split('=')[:-1])
                    score = float(wv_s[-1])

                    # change base to e
                    score = score * np.log(10)

                    yield (w, score)

                except:
                    print(f'parsing scores fails: {wv}')

def interpolate(nlm_hypos, kenlm_scores, weight=0.75):
    total_score = 0
    cnt = 0

    for hypo, (w, s) in zip(hypos, read_kenlm_score(args.kenlm_score)):
        assert hypo['s'] == w
        total_score += logsumexp(np.stack((s + np.log(1-weight), hypo['lm_s']+np.log(weight)), axis=-1), axis=-1)
        cnt += 1

    return cnt, np.exp(-total_score / cnt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--nlm-score', type=str,
        help='the json file that contain the nlm scores for every token')
    parser.add_argument('--kenlm-score', type=str,
        help='the output score file from kenlm querying, note that the scores in this kenlm output \
        is in log base 10 by default')
    parser.add_argument('--weight', type=float, default=0.75, help='the weight of NLM at interpolation')

    args = parser.parse_args()


    for w in np.arange(0.9, 1, 0.05):
        hypos = read_nlm_score(args.nlm_score)
        kenlm_iter = read_kenlm_score(args.kenlm_score)
        cnt, ppl = interpolate(hypos, kenlm_iter, w)
        print(f'nlm weight {w}, {cnt} tokens, ppl: {ppl}')

