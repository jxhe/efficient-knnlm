"""
sample articles from the given dataset
"""

import argparse
import random


parser = argparse.ArgumentParser(description='')
parser.add_argument('--input', type=str,
    help='input text file')
parser.add_argument('--n', type=int, default=500,
    help='the number of sampled articles')
parser.add_argument('--output', type=str, help='output file prefix')


args = parser.parse_args()

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

data = read_wikitext103(args.input)
print(f'there are {len(data)} articles')
size = args.n

random.seed(22)
sample_articles = random.sample(data, size)

with open(args.output, 'w') as fout:
    fout.write('\n')
    for article in sample_articles:
        write_article(fout, article)

