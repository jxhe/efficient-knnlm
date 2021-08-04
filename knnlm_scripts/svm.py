import json
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from collections import Counter
from scipy.special import logsumexp

# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC, SVC


class TokenFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, src, tgt):
        super().__init__()

        self.src = src
        self.tgt = tgt

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, index):
        return{
            "id": index,
            "feature": self.src[index],
            "label": self.tgt[index],
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        def merge(key, dtype=torch.float32):
            return torch.tensor([s[key] for s in samples],
                                dtype=dtype,
                                device=self.device)

        batch = {
            'id': merge('id'),
            'feature': merge('feature'),
            'label': merge('label', dtype=torch.long),
        }
        return batch

    def get_nfeature(self):
        return len(self.src[0])


def get_ngram_freq(file, ngram=4):
    res = Counter()
    prev = ['</s>']
    with open(file) as fin:
        for i, line in enumerate(fin):
            if i % 100000 == 0:
                print(f'procesed {i} lines')
            for tok in line.strip().split():
                prev = prev[-ngram:]
                for j in range(1, ngram+1):
                    res[' '.join(prev[-j:])] += 1
                prev.append(tok)

            prev.append('</s>')

    return res

def preprocess_data(hypos, feature_type: str):
    features = []
    labels = []
    prev = ['</s>']
    token_hypos = []
    keys = hypos[0].keys()
    for hypo in hypos:
        assert len(hypo['lm_context']) == len(hypo['string'])
        for i in range(len(hypo['string'])):
            local_f = []

            if feature_type == 'context':
                local_f.extend(hypo['lm_context'][i])
            elif feature_type == 'freq':
                local_f.extend(hypo['freq'][i])
            elif feature_type == 'all':
                # confidence-related features
                local_f.extend([hypo['lm_entropy'][i], np.exp(hypo['lm_max'][i])])
                local_f.extend(hypo['freq'][i])
                # local_f.extend(hypo['lm_context'][i])
            else:
                raise ValueError

            labels.append(int((hypo['positional_scores'][i] - hypo['lm_scores'][i]) > 0.01))
            features.append(local_f)
            token_hypos.append({key: hypo[key][i] for key in keys})

    return features, labels, token_hypos

def validate(val_dataloader, model):
    model.eval()
    running_loss = 0.
    nsamples = 0
    truth_list = []
    prediction_dict = {}
    truth_dict = {}
    prediction_list = []
    for i, sample in enumerate(val_dataloader, 0):
        inputs, truth = sample['feature'], sample['label']
        truth_list.extend(truth.tolist())
#         inputs, labels = sample_check['feature'], sample_check['label']
        outputs = model(inputs)
        loss = criterion(outputs, truth)

        # (batch)
        _, preds = torch.max(outputs, dim=1)
#         import pdb; pdb.set_trace()
        prediction_list.extend(preds.tolist())

        for id_, p, t in zip(sample['id'], preds, truth):
            prediction_dict[id_.item()] = p.item()
            truth_dict[id_.item()] = t.item()

        running_loss += loss.item() * inputs.size(0)
        nsamples += inputs.size(0)

    print(f"val loss: {running_loss/nsamples:.3f}")
    print(f"val accuracy: {accuracy_score(truth_list, prediction_list):.3f}")
    precision, recall, fscore, _ = precision_recall_fscore_support(truth_list,
                                                              prediction_list,
                                                              average='binary',
                                                              pos_label=1)
    print(f"val precision: {precision:.3f}, recall: {recall:.3f} \
            fscore: {fscore:.3f}")

    return truth_list, prediction_list, prediction_dict, truth_dict

def interpolation(hypos, predictions):
    scores = 0
    cnt = 0
    lambda_ = 0.75
    ndict = 267744
    assert len(predictions) == len(hypos)
    for hypo, pred in zip(hypos, predictions):
        knn_weight = pred * np.log(1-lambda_) + (1 - pred) * (-1e5)
        lm_weight = pred * np.log(lambda_)

        knn_scores = hypo['knn_scores']
        lm_scores = hypo['lm_scores']
        combine = logsumexp(np.stack((knn_scores + knn_weight, lm_scores+lm_weight), axis=-1), axis=-1)
        scores += combine.sum()
        cnt += 1

    return np.exp(-scores / cnt)

def train_test_split(x, y, test_size=0.2):
    assert len(x) == len(y)
    indexes = np.arange(len(x))
    np.random.shuffle(indexes)

    boundary = int(len(x) * test_size)
    test_indexes = indexes[:boundary]
    train_indexes = indexes[boundary:]

    x_train = [x[i] for i in train_indexes]
    y_train = [y[i] for i in train_indexes]

    x_test = [x[i] for i in test_indexes]
    y_test = [y[i] for i in test_indexes]

    return x_train, x_test, y_train, y_test, train_indexes, test_indexes


parser = argparse.ArgumentParser(description='')
parser.add_argument('--negative-weight', type=float, default=1,
        help='weight of the loss from negative examples, range [0,1]')
parser.add_argument('--feature-type', type=str, choices=['context', 'freq', 'all'],
    help='the features to use')
parser.add_argument('--seed', type=int, default=22,
    help='the random seed')

args = parser.parse_args()

np.random.seed(args.seed)

hypos = []
with open('features.jsonl') as fin:
    for line in fin:
        hypos.append(json.loads(line.strip()))


features, labels, token_hypos = preprocess_data(hypos, feature_type=args.feature_type)
# import pdb; pdb.set_trace()


x_train, x_val, y_train, y_val, train_ids, val_ids = train_test_split(features, labels, test_size=0.2)
val_token_hypos = [token_hypos[i] for i in val_ids]

if args.feature_type == 'context':
    x_train_norm = x_train
    x_val_norm = x_val
else:
    scaler = StandardScaler()
    scaler = scaler.fit(x_train)

    x_train_norm = scaler.transform(x_train)
    x_val_norm = scaler.transform(x_val)


# model = LinearSVC(dual=False, verbose=True,
#     class_weight='balanced', random_state=args.seed, tol=1e-6)
model = SVC(kernel='rbf', verbose=True,
    class_weight='balanced', random_state=args.seed, tol=1e-6)


print(f'no retrieval ppl {interpolation(val_token_hypos, np.array([0] * len(val_token_hypos)))}')
print(f'all retrieval ppl {interpolation(val_token_hypos, np.array([1] * len(val_token_hypos)))}')

# import pdb; pdb.set_trace()

model.fit(x_train_norm, y_train)
predictions = model.predict(x_val_norm)

print(f"val accuracy: {accuracy_score(y_val, predictions):.3f}")
precision, recall, fscore, _ = precision_recall_fscore_support(y_val,
                                                          predictions,
                                                          average='binary',
                                                          pos_label=1)
print(f"val precision: {precision:.3f}, recall: {recall:.3f} \
        fscore: {fscore:.3f}")
ppl = interpolation(val_token_hypos, predictions)
print(f'{predictions.sum() / len(predictions)} retrieval, ppl {ppl}')
