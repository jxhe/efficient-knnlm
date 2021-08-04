"""this script transforms the kenlm output file
to jsonl format to be consistent with knnlm
"""

import argparse

parser.add_argument('--kenlm-score', type=str,
    help='the output score file from kenlm querying, note that the scores in this kenlm output \
    is in log base 10 by default')
parser.add_argument('--output', type=str,
    help='the output file')



args = parser.parse(args)

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


with open(args.output, 'w') as fout:
    for i, (w, s) in enumerate(read_kenlm_score(args.kenlm_score)):
        if i % 100000 == 0:
            print(f'process {i} tokens')

        fout.write(json.dumps({
            's': w,
            'kenlm_s': s,
            }, ensure_ascii=False))
        fout.write('\n')
        fout.flush()
