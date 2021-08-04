"""create k-fold datasets from the training data
"""


import os
import random

dataset = 'wikitext-103'
data_dir = f'examples/language_model/{dataset}'
outdir = f'datasets/{dataset}/jackknife'

def read_wikitext103(fname):
    """read the wikitext-103 dataset
    in terms of complete articles
    """
    res = []
    article = []
    with open(fname) as fin:
        for line in fin:
            line_s = line.split()
            if len(line_s) >= 2 and line_s[0] == '=' and line_s[-1] == '=' and line_s[1] != '=':
                if article[0].strip() != '' and article[0].strip()[0] == '=':
                    res.append(article)
                article = [line]
            else:
                article.append(line)

    if article != []:
        res.append(article)

    return res

def write_article(fout, article):
    for line in article:
        fout.write(line)

def check_overlap(new, existing):
    for (a, b) in existing:
        if (new[0] >= a and new[0] < b ) \
            or (new[1] >= a and new[1] < b):
            return True

    return False

if not os.path.exists(outdir):
    os.makedirs(outdir)

dfile = os.path.join(data_dir, 'wiki.train.tokens')
val_file = os.path.join(data_dir, 'wiki.valid.tokens')

data = read_wikitext103(dfile)
val_size = len(read_wikitext103(val_file))

kfold = 5
length = len(data)

random.seed(22)

index = list(range(length))
random.shuffle(index)

for i in range(kfold):
    val_index = set(index[i * val_size:(i+1) * val_size])
    with open(os.path.join(outdir, f'train.fold{i}'), 'w') as ftrain, \
        open(os.path.join(outdir, f'val.fold{i}'), 'w') as fval:
        ftrain.write('\n')
        fval.write('\n')
        for j, article in enumerate(data):
            if j not in val_index:
                write_article(ftrain, article)
            else:
                write_article(fval, article)

