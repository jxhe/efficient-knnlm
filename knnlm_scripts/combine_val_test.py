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
outdir = f'datasets/{dataset}-valtest'

if not os.path.exists(outdir):
    os.makedirs(outdir)

dfile = os.path.join(data_dir, 'wiki.valid.tokens')

old_val_data = read_wikitext103(os.path.join(data_dir, 'wiki.valid.tokens'))
old_test_data = read_wikitext103(os.path.join(data_dir, 'wiki.test.tokens'))

val_size = len(old_test_data)


random.seed(22)

index = list(range(val_size))
random.shuffle(index)

train_indexes = index[:int(val_size * 0.6)]
val_indexes = index[int(val_size * 0.6):]

# use all old val data and part of the test data as the new training data,
# and others as the new val data
with open(os.path.join(outdir, f'train.tokens'), 'w') as ftrain, \
    open(os.path.join(outdir, f'valid.tokens'), 'w') as fval:

    ftrain.write(' \n')
    fval.write(' \n')

    for article in old_val_data:
        write_article(ftrain, article)

    for i in sorted(train_indexes):
        write_article(ftrain, old_test_data[i])

    for i in sorted(val_indexes):
        write_article(fval, old_test_data[i])
