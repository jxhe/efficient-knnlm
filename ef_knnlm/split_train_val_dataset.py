import os
import random

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


dataset = 'wikitext-103'
data_dir = f'examples/language_model/{dataset}'
train_size = 0.2
outdir = f'datasets/{dataset}-train{train_size}'

if not os.path.exists(outdir):
    os.makedirs(outdir)

dfile = os.path.join(data_dir, 'wiki.train.tokens')

data = read_wikitext103(dfile)

length = len(data)

random.seed(99)

index = list(range(length))
random.shuffle(index)


train_indexes = index[:int(length * train_size)]
held_out_indexes = index[int(length * train_size):]

with open(os.path.join(outdir, f'train.tokens'), 'w') as ftrain, \
    open(os.path.join(outdir, f'val.tokens'), 'w') as fval:
    ftrain.write('\n')
    fval.write('\n')

    for i in sorted(train_indexes):
        write_article(ftrain, data[i])

    for i in sorted(held_out_indexes):
        write_article(fval, data[i])
