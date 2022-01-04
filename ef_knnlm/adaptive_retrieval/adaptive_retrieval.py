import json
import sys
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from collections import Counter, OrderedDict
from scipy.special import logsumexp
from datasets import load_dataset

from moe_modules import MLPMOE, LSTMMOE, TokenFeatureDataset

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "a")

  def write(self, message):
    print(message, end="", file=self.terminal, flush=True)
    print(message, end="", file=self.log, flush=True)

  def flush(self):
    self.terminal.flush()
    self.log.flush()


def validate(val_dataloader, model, args):
    model.eval()
    model.epoch_update()
    running_loss = 0.
    nsamples = 0
    prediction_dict = {}
    for i, sample in enumerate(val_dataloader, 0):
        inputs, lm_scores, knn_scores= sample['feature'], sample['lm_scores'], sample['knn_scores']
#         inputs, labels = sample_check['feature'], sample_check['label']
        # import pdb;pdb.set_trace()
        log_weight = model(inputs)
        cross_entropy = log_weight + torch.stack((lm_scores, knn_scores), dim=-1)

        # (B,)
        cross_entropy = -torch.logsumexp(cross_entropy, dim=-1)
        loss = cross_entropy.mean()
        ent_loss = loss

        if args.l1 > 0:
            loss = loss + args.l1 * torch.abs(log_weight.exp()[:,1]).sum() / log_weight.size(0)

        # (batch)
        preds = log_weight[:, 0]
#         import pdb; pdb.set_trace()

        for id_, p in zip(sample['id'], preds):
            prediction_dict[id_.item()] = p.item()

        bsz = next(iter(inputs.values())).size(0)

        running_loss += ent_loss.item() * bsz
        nsamples += bsz

    val_loss = running_loss / nsamples

    print(f"val loss: {val_loss:.3f}, ppl: {np.exp(val_loss)}")

    return val_loss, prediction_dict

def interpolation(hypos, predictions, lambda_=0.75):
    scores = 0
    cnt = 0
    ndict = 267744
    assert len(predictions) == len(hypos)
    for i, (hypo, pred) in enumerate(zip(hypos, predictions)):
        # if i % 1000 == 0:
        #     print(f'interpolation processed {i} tokens')
        knn_weight = pred * np.log(1-lambda_) + (1 - pred) * (-1e5)
        lm_weight = pred * np.log(lambda_)

        knn_scores = hypo['knn_s']
        lm_scores = hypo['lm_s']
        combine = logsumexp(np.stack((knn_scores + knn_weight, lm_scores+lm_weight), axis=-1), axis=-1)
        scores += combine.sum()
        cnt += 1

    return np.exp(-scores / cnt)


def moe_interpolation(hypos, predictions, cutoff=None, random_mask=None, constant_weight=None, threshold=None):
    """perform interpolation while weights are output from a
    gating network. only perform retrieval in a certain portion
    of tokens when cutoff is not None
    """
    scores = 0
    cnt = 0
    ts = None
    # ndict = 267744
    assert len(predictions) == len(hypos)
    predictions_copy = predictions

    if constant_weight is not None:
        predictions = [constant_weight] * len(predictions_copy)

    if cutoff is not None:
        if random_mask is None:
            if threshold is not None:
                ts = threshold[cutoff * 100]
                mask = (predictions_copy >= ts).astype('float')
                print(f'actual cutoff {mask.sum() / len(mask)}')
            else:
                ts = np.sort(predictions_copy)[int(len(predictions_copy) * (1. - cutoff))]
                mask = (predictions_copy >= ts).astype('float')
        else:
            # mask = np.zeros(len(predictions))
            # mask[int(len(predictions) * (1. - cutoff)):] = 1
            # np.random.shuffle(mask)
            # mask = mask.astype('float')
            ts = None
            mask = random_mask

        lm_weights = (1-mask) * predictions + mask * 0.
        knn_prob = 1. - np.exp(predictions)
        overflow = (knn_prob <= 0)
        knn_prob = np.clip(knn_prob, 1e-5, 1)
        knn_weights = np.log(knn_prob)
        knn_weights[overflow] = -1e5
        knn_weights = (1-mask) * knn_weights + mask * (-1e5)
    else:
        lm_weights = predictions
        knn_prob = 1. - np.exp(predictions)
        overflow = (knn_prob <= 0)
        knn_prob = np.clip(knn_prob, 1e-5, 1)
        knn_weights = np.log(knn_prob)
        knn_weights[overflow] = -1e5

    for hypo, lm_weight, knn_weight in zip(hypos, lm_weights, knn_weights):

        knn_scores = hypo['knn_s']
        lm_scores = hypo['lm_s']
        combine = logsumexp(np.stack((knn_scores + knn_weight, lm_scores+lm_weight), axis=-1), axis=-1)
        scores += combine.sum()
        cnt += 1

    return np.exp(-scores / cnt), ts


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

def save_val_pred(hypos, predictions, path):
    new_hypos = []
    predictions = predictions.astype('float')
    start = 0

    assert len(hypos) == len(predictions)

    for hypo, pred in zip(hypos, predictions):
        hypo['pred'] = pred

        new_hypos.append(hypo)

    with open(path, 'w') as fout:
        for hypo in new_hypos:
            fout.write(json.dumps(hypo, ensure_ascii=False))
            fout.write('\n')
            fout.flush()

def read_input(input, debug=False):
    hypos = []
    fname = 'features_small.jsonl' if args.debug else input

    dataset = load_dataset('json', data_files=fname, cache_dir='hf_cache', use_threads=True)

    return dataset['train']


parser = argparse.ArgumentParser(description='')

parser.add_argument('--train', type=str, default=None,
    help='the input feature file (jsonl)')
parser.add_argument('--val', type=str, default=None,
    help='the input feature file (jsonl)')
parser.add_argument('--train-others', type=str, default=None,
    help='use a specified jsonl file for others feature if specified')
parser.add_argument('--val-others', type=str, default=None,
    help='use a specified jsonl file for others feature if specified')
parser.add_argument('--input', type=str, default=None,
    help='the input feature file (jsonl). Multiple files are separated with comma')
parser.add_argument('--negative-weight', type=float, default=1,
        help='weight of the loss from negative examples, range [0,1]')
parser.add_argument('--feature-type', type=str, default='all',
    help='the features to use, splitted with commas')
parser.add_argument('--seed', type=int, default=22,
    help='the random seed')
parser.add_argument('--debug', action='store_true', default=False,
    help='debug mode')


# interpolation with ngram kenlm instead of knnlm
parser.add_argument('--train-kenlm', type=str, default=None,
    help='the output score file from kenlm querying, note that the scores in this kenlm output \
    is in log base 10 by default')
parser.add_argument('--val-kenlm', type=str, default=None,
    help='the output score file from kenlm querying, note that the scores in this kenlm output \
    is in log base 10 by default')

# training arguments
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--l1', type=float, default=0.,
    help='l1 regularization coefficient')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--ngram', type=int, default=0, help='the ngram features to use')


# model hyperparameters
parser.add_argument('--arch', type=str, choices=['mlp', 'lstm'], default='mlp',
    help='architectures of the expert model')
parser.add_argument('--activation', type=str, choices=['linear', 'relu'], default='relu',
    help='the activation function in mlp')
parser.add_argument('--hidden-units', type=int, default=32, help='hidden units')
parser.add_argument('--nlayers', type=int, default=3, help='number of layerss')
parser.add_argument('--dropout', type=float, default=0, help='dropout')


parser.add_argument('--output-dir', type=str)
parser.add_argument('--move-to-mem', action='store_true', default=False)
parser.add_argument('--load-model', type=str, default=None,
    help='load model checkpoint')
parser.add_argument('--eval', action='store_true', default=False,
    help='perform evaluation')
parser.add_argument('--save-pred', type=str, default=None,
    help='save predictions for analysis')
parser.add_argument('--validate-loss', action='store_true', default=False,
    help='save predictions for analysis')

args = parser.parse_args()

# args.output_dir = f'checkpoint/moe/mlp.nh{args.hidden_units}.nl{args.nlayers}.drop{args.dropout}.lr{args.lr}.ft{args.feature_type}.seed{args.seed}'

# if not os.path.isdir(args.output_dir):
#     os.makedirs(args.output_dir)

logfile = 'stdout.log' if not args.eval else 'eval.log'
sys.stdout = Logger(os.path.join(args.output_dir, logfile))

print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.input is not None:
    hypos = []
    if args.debug:
        hypos = read_input(None, debug=args.debug)
    else:
        for fname in args.input.split(','):
            hypos.extend(read_input(fname, debug=args.debug))

    test_size = 0.2
    indexes = np.arange(len(hypos))
    # np.random.shuffle(indexes)
    boundary = int(len(hypos) * test_size)
    test_indexes = indexes[:boundary]
    train_indexes = indexes[boundary:]

    train_hypos = [hypos[x] for x in train_indexes]
    val_hypos = [hypos[x] for x in test_indexes]
else:
    train_ctxt_hypos = read_input(args.train + '_ctxt.jsonl', debug=args.debug)

    if args.train_others is None:
        train_other_hypos = read_input(args.train + '_others.jsonl', debug=args.debug)
    else:
        train_other_hypos = read_input(args.train_others)

    val_ctxt_hypos = read_input(args.val + '_ctxt.jsonl', debug=args.debug)

    if args.val_others is None:
        val_other_hypos = read_input(args.val + '_others.jsonl', debug=args.debug)
    else:
        val_ctxt_hypos = read_input(args.val_others)

    if args.train_kenlm is not None:
        train_kenlm = read_input(args.train_kenlm)
        val_kenlm = read_input(args.val_kenlm)
    else:
        train_kenlm = None
        val_kenlm = None

    if args.move_to_mem:
        train_ctxt_hypos = [train_ctxt_hypos[i] for i in range(len(train_ctxt_hypos))]
        train_other_hypos = [train_other_hypos[i] for i in range(len(train_other_hypos))]
        val_ctxt_hypos = [val_ctxt_hypos[i] for i in range(len(val_ctxt_hypos))]
        val_other_hypos = [val_other_hypos[i] for i in range(len(val_other_hypos))]

print('complete reading jsonl files')


training_set = TokenFeatureDataset(train_ctxt_hypos, train_other_hypos, train_kenlm, ngram=args.ngram)
val_set = TokenFeatureDataset(val_ctxt_hypos, val_other_hypos, val_kenlm, ngram=args.ngram)

train_sampler = torch.utils.data.SequentialSampler(training_set) if args.arch == 'lstm' else None
val_sampler = torch.utils.data.SequentialSampler(val_set) if args.arch == 'lstm' else None

train_dataloader = torch.utils.data.DataLoader(training_set,
                                               batch_size=args.batch_size,
                                               shuffle=False if args.arch == 'lstm' else True,
                                               sampler=train_sampler,
                                               collate_fn=training_set.collater)

val_dataloader = torch.utils.data.DataLoader(val_set,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             sampler=val_sampler,
                                             collate_fn=val_set.collater)

nepochs = 10

extra_feature_size = None


feature_set = ['ctxt', 'freq', 'lm_ent', 'lm_max', 'fert']

if args.feature_type == 'all':
    feature_size = OrderedDict({key: training_set.get_nfeature(key) for key in feature_set})
else:
    feature_size = OrderedDict({key: training_set.get_nfeature(key) for key in args.feature_type.split(',')})

args.feature_size = feature_size

if args.arch == 'mlp':
    model = MLPMOE(
                feature_size=feature_size,
                hidden_units=args.hidden_units,
                nlayers=args.nlayers,
                dropout=args.dropout,
                activation=args.activation,
                )
elif args.arch == 'lstm':
    model = LSTMMOE(
                feature_size=feature_size,
                hidden_units=args.hidden_units,
                nlayers=args.nlayers,
                dropout=args.dropout,
                )
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([args.negative_weight, 1]))

if args.load_model:
    ckpt_path = os.path.join(args.load_model, 'checkpoint_best.pt')
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['param'])
    print(f"loaded model ckpt from {ckpt_path} at epoch {ckpt['epoch']}")


if torch.cuda.is_available():
    print('use cuda')
    model.cuda()
    # criterion.cuda()

print(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
model.train()

val_hypos_mem = []

tmp = time.time()
print('moving scores to memory')

for i, hypo in enumerate(val_other_hypos):
    if args.val_kenlm is not None:
        assert hypo['s'] == val_kenlm[i]['s']
        knns = val_kenlm[i]['kenlm_s']
    else:
        knns = hypo['knn_s']

    val_hypos_mem.append({'lm_s': hypo['lm_s'], 'knn_s': knns})

print(f'moving scores consumes {time.time() - tmp} seconds')

tmp = time.time()
print(f'no retrieval ppl {interpolation(val_hypos_mem, np.array([0] * len(val_hypos_mem)))}')
print(f'interpolation costs {time.time() - tmp} seconds')

cutoff_list = [10, 30, 50, 70, 90]

# cutoff_list = [50]
random_mask = {}

# log_str = 'random mask, constant weights, val interpolate ppl (cutoff): '
for cutoff in cutoff_list:
    mask = np.zeros(len(val_hypos_mem))
    mask[int(len(mask) * (1. - cutoff / 100)):] = 1
    np.random.shuffle(mask)
    mask = mask.astype('float')
    random_mask[cutoff] = mask

if args.eval:
    val_loss, prediction_dict = validate(val_dataloader, model, args)
    predictions = np.array([prediction_dict[k] for k in range(len(val_hypos_mem))])

    log_str = f'val interpolate ppl (cutoff): '
    ppl, _ = moe_interpolation(val_hypos_mem, predictions)
    log_str += f'0:{ppl:.3f}, '


    for cutoff in cutoff_list:
        ppl_cutoff, _ = moe_interpolation(val_hypos_mem, predictions, cutoff=cutoff / 100, threshold=ckpt['threshold'])
        log_str += f'{cutoff}:{ppl_cutoff:.3f}, '

    print(log_str)

    log_str = f'random mask, learned weights ppl (cutoff): '
    for cutoff in cutoff_list:
        ppl_cutoff, _ = moe_interpolation(val_hypos_mem, predictions, cutoff=cutoff / 100, random_mask=random_mask[cutoff])
        log_str += f'{cutoff}:{ppl_cutoff:.3f}, '
    print(log_str)

    log_str = f'learned mask, constant weights ppl (cutoff): '
    for cutoff in cutoff_list:
        ppl_cutoff, _ = moe_interpolation(val_hypos_mem, predictions, cutoff=cutoff / 100, constant_weight=np.log(0.75))
        log_str += f'{cutoff}:{ppl_cutoff:.3f}, '
    print(log_str)

    log_str = 'random mask, constant weights, val interpolate ppl (cutoff): '
    for cutoff in cutoff_list:

        ppl, _ = moe_interpolation(val_hypos_mem,
            np.zeros(len(val_hypos_mem)), cutoff=cutoff/100,
            random_mask=random_mask[cutoff], constant_weight=np.log(0.75))

        log_str += f'{cutoff}:{ppl:.3f}, '

    print(log_str)

    # print('save predictions')
    # save_val_pred(val_other_hypos, predictions, os.path.join(args.load_model, 'pred.jsonl'))

    sys.exit()

for lambda_ in np.arange(0.1, 0.9, 0.1):
    print(f'all retrieval ppl (lambda {lambda_}) {interpolation(val_hypos_mem, np.array([1] * len(val_hypos_mem)), lambda_)}')

# lambda_ = 0.75
# print(f'all retrieval ppl (lambda {lambda_}) {interpolation(val_hypos_mem, np.array([1] * len(val_hypos_mem)), lambda_)}')

# compute upper bound
mask = np.zeros(len(val_hypos_mem))
for i, hypo in enumerate(val_hypos_mem):
    if hypo['lm_s'] >= hypo['knn_s']:
        mask[i] = 1

ppl, _ = moe_interpolation(val_hypos_mem,
    np.zeros(len(val_hypos_mem)), cutoff=0,
    random_mask=mask, constant_weight=np.log(0.75))

log_str = f'ground-truth mask, constant weights, masked {mask.sum()/len(mask):.2f}, ppl: '
log_str += f'{ppl:.3f}, '


print(log_str)

# cutoff_list = [50]
# random_mask = {}

log_str = 'random mask, constant weights, val interpolate ppl (cutoff): '
for cutoff in cutoff_list:

    ppl, _ = moe_interpolation(val_hypos_mem,
        np.zeros(len(val_hypos_mem)), cutoff=cutoff/100,
        random_mask=random_mask[cutoff], constant_weight=np.log(0.75))

    log_str += f'{cutoff}:{ppl:.3f}, '

print(log_str)



best_loss = 1e5
best_half_cut_ppl = 1e5
for epoch in range(nepochs):
    running_loss = 0.
    nsamples = 0

    model.epoch_update()

    for i, sample in enumerate(train_dataloader, 0):
        inputs, lm_scores, knn_scores = sample['feature'], sample['lm_scores'], sample['knn_scores']
        # import pdb; pdb.set_trace()
        optimizer.zero_grad()

        # (B x 2): log probability
        log_weight = model(inputs)

        cross_entropy = log_weight + torch.stack((lm_scores, knn_scores), dim=-1)

        # (B,)
        cross_entropy = -torch.logsumexp(cross_entropy, dim=-1)
        loss = cross_entropy.mean()

        if args.l1 > 0:
            loss = loss + args.l1 * torch.abs(log_weight.exp()[:,1]).sum() / log_weight.size(0)

        loss.backward()
        optimizer.step()

        bsz = next(iter(inputs.values())).size(0)
        running_loss += loss.item() * bsz
        nsamples += bsz

        if (i+1) % 500 == 0:
            report_loss = running_loss / nsamples
            print(f'epoch: {epoch}, step: {i},  \
                training loss: {report_loss:.3f}, ppl: {np.exp(report_loss)}')
            # running_loss = 0
            # nsamples = 0


    val_loss, prediction_dict = validate(val_dataloader, model, args)
    # torch.save({'epoch': epoch,
    #             'param': model.state_dict()},
    #             os.path.join(args.output_dir, f'checkpoint_{epoch}.pt'))
    # if val_loss < best_loss:
    #     best_loss = val_loss


    predictions = np.array([prediction_dict[k] for k in range(len(val_hypos_mem))])

    log_str = f'epoch: {epoch}, val interpolate ppl (cutoff): '
    ppl, _ = moe_interpolation(val_hypos_mem, predictions)
    log_str += f'0:{ppl:.3f}, '


    cutoff2ppl = {}
    cutoff2ts = {}
    for cutoff in cutoff_list:
        ppl_cutoff, ts = moe_interpolation(val_hypos_mem, predictions, cutoff=cutoff / 100)
        log_str += f'{cutoff}:{ppl_cutoff:.3f}, '

        cutoff2ppl[cutoff] = ppl_cutoff
        cutoff2ts[cutoff] = ts

    if not args.validate_loss:
        # use 50 cutoff ppl to validate performance
        if cutoff2ppl[50] < best_half_cut_ppl:
            best_half_cut_ppl = cutoff2ppl[50]
            print('save model')
            torch.save({'epoch': epoch,
                        'args': args,
                        'threshold': cutoff2ts,
                        'param': model.state_dict()},
                        os.path.join(args.output_dir, 'checkpoint_best.pt'))
    else:
        if val_loss < best_loss:
            best_loss = val_loss
            print('save model')
            torch.save({'epoch': epoch,
                        'args': args,
                        'threshold': cutoff2ts,
                        'param': model.state_dict()},
                        os.path.join(args.output_dir, 'checkpoint_best.pt'))

    print(log_str)

    log_str = f'epoch: {epoch}, random mask, learned weights ppl (cutoff): '
    for cutoff in cutoff_list:
        ppl_cutoff, _ = moe_interpolation(val_hypos_mem, predictions, cutoff=cutoff / 100, random_mask=random_mask[cutoff])
        log_str += f'{cutoff}:{ppl_cutoff:.3f}, '
    print(log_str)

    log_str = f'epoch: {epoch}, learned mask, constant weights ppl (cutoff): '
    for cutoff in cutoff_list:
        ppl_cutoff, _ = moe_interpolation(val_hypos_mem, predictions, cutoff=cutoff / 100, constant_weight=np.log(0.75))
        log_str += f'{cutoff}:{ppl_cutoff:.3f}, '
    print(log_str)
    # test

    # save_val_pred(val_token_hypos, val_hypos, predictions, os.path.join(args.output_dir, f'epoch{epoch}_pred.jsonl'))
    # truths = np.array([truth_dict[k] for k in range(len(val_token_hypos))])
    # ppl = interpolation(val_token_hypos, truths)
    # print(f'upper bound: {truths.sum() / len(truths)} retrieval, ppl {ppl}')
    model.train()

print(f'best val cutoff 50 ppl: {best_half_cut_ppl:.3f}')


