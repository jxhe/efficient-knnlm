import os
import random

from datasets import load_dataset


dataset = load_dataset('bookcorpusopen')

data = dataset['train']

random.seed(22)

id_list = list(range(len(data)))
random.shuffle(id_list)

nval = ntest = 50

# one third of the full data for training
ntrain = len(data) // 3

val_d = [data[i] for i in id_list[:nval]]
test_d = [data[i] for i in id_list[nval:nval+ntest]]
train_d = [data[i] for i in id_list[nval+ntest:nval+ntest+ntrain]]

outdir = 'datasets/bookcorpus'
os.makedirs(outdir, exist_ok=True)

def write_out(fname, data):
    print(f'write {fname}')
    # import pdb;pdb.set_trace()
    with open(fname, 'w') as fout:
        for r in data:
            fout.write(r['text'])


write_out(os.path.join(outdir, 'train.orig'), train_d)
write_out(os.path.join(outdir, 'valid.orig'), val_d)
write_out(os.path.join(outdir, 'test.orig'), test_d)
