#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import math
import os
import json
import time
import pickle

import torch
import numpy as np

from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data import LMContextWindowDataset
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from fairseq.knnlm import KNN_Dstore


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('fairseq_cli.eval_lm')


class WordStat(object):
    def __init__(self, word, is_bpe):
        self.word = word
        self.is_bpe = is_bpe
        self.log_prob = 0
        self.next_word_prob = 0
        self.count = 0
        self.missing_next_words = 0

    def add(self, log_prob, next_word_prob):
        """ increments counters for the sum of log probs of current word and next
            word (given context ending at current word). Since the next word might be at the end of the example,
            or it might be not counted because it is not an ending subword unit,
            also keeps track of how many of those we have seen """
        if next_word_prob is not None:
            self.next_word_prob += next_word_prob
        else:
            self.missing_next_words += 1
        self.log_prob += log_prob
        self.count += 1

    def __str__(self):
        return '{}\t{}\t{}\t{}\t{}\t{}'.format(self.word, self.count, self.log_prob, self.is_bpe,
                                               self.next_word_prob, self.count - self.missing_next_words)


def main(parsed_args):
    assert parsed_args.path is not None, '--path required for evaluation!'

    utils.import_user_module(parsed_args)

    logger.info(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)

    # Load ensemble
    logger.info('loading model(s) from {}'.format(parsed_args.path))
    models, args = checkpoint_utils.load_model_ensemble(
        parsed_args.path.split(os.pathsep),
        arg_overrides=eval(parsed_args.model_overrides),
        task=task,
    )

    # import pdb; pdb.set_trace()

    logger.info(f'training max tokens {args.max_tokens}, tokens per sample {args.tokens_per_sample}, break mode {args.sample_break_mode}')

    for arg in vars(parsed_args).keys():
        if arg not in {
            'self_target', 'future_target', 'past_target', 'tokens_per_sample',
            'output_size_dictionary', 'add_bos_token',
        }:
            setattr(args, arg, getattr(parsed_args, arg))

    # reduce tokens per sample by the required context window size
    args.tokens_per_sample -= args.context_window
    task = tasks.setup_task(args)

    args.ar_feat_type = args.ar_feat_type.split(',')

    if args.ar_ckpt != 'none' and args.ar_freq_dict != '':
        if 'freq' in args.ar_feat_type:
            print('loading freq cache')
            freq_id_cache = pickle.load(open(os.path.join(args.ar_freq_dict, 'freq_cache_id.pickle'), 'rb'))
        else:
            freq_id_cache = None

        if 'fert' in args.ar_feat_type:
            print('loading fert cache')
            fertility_id_cache = pickle.load(open(os.path.join(args.ar_freq_dict, 'fertility_cache_id.pickle'), 'rb'))
        else:
            fertility_id_cache = None
        # fertility_id_cache=None
        # fertility_id_cache = freq_id_cache
    else:
        freq_id_cache = fertility_id_cache = None

    if args.context_window > 0:
        # Load dataset splits
        task.load_dataset(args.gen_subset)
        dataset = task.dataset(args.gen_subset)

        dataset = LMContextWindowDataset(
            dataset=dataset,
            tokens_per_sample=args.tokens_per_sample,
            context_window=args.context_window,
            pad_idx=task.source_dictionary.pad(),
            freq=freq_id_cache,
            fert=fertility_id_cache,
            knnlm_feat_csize=args.knnlm_feat_csize,
        )

    else:
        task.load_dataset(args.gen_subset,
                          freq=freq_id_cache,
                          fert=fertility_id_cache,
                          knnlm_feat_csize=args.knnlm_feat_csize,
                          )

        dataset = task.dataset(args.gen_subset)

    logger.info('{} {} {} examples'.format(args.data, args.gen_subset, len(dataset)))

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        model.make_generation_fast_()
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    assert len(models) > 0

    logger.info('num. model params: {}'.format(sum(p.numel() for p in models[0].parameters())))

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens or 36000,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(*[
            model.max_positions() for model in models
        ]),
        ignore_invalid_inputs=True,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(task.target_dictionary, args.softmax_batch, args=args)

    score_sum = 0.
    count = 0

    if args.remove_bpe is not None:
        if args.remove_bpe == 'sentencepiece':
            raise NotImplementedError
        else:
            bpe_cont = args.remove_bpe.rstrip()
            bpe_toks = {
                i
                for i in range(len(task.source_dictionary))
                if task.source_dictionary[i].endswith(bpe_cont)
            }
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0

    word_stats = dict()

    if args.knnlm and args.save_knnlm_dstore:
        raise ValueError("Cannot use knnlm while trying to build the datastore!")

    if args.knnlm:
        knn_dstore = KNN_Dstore(args)

    if args.save_feature is not None:
        fout_ctxt = open(args.save_feature + '_ctxt.jsonl', 'w')
        fout_extra = open(args.save_feature + '_others.jsonl', 'w')

        ngram = args.knnlm_feat_csize
        prev = [task.target_dictionary.index('</s>')] * ngram

        freq_cache = os.path.join(args.ar_feat_cache, 'freq_cache_id.pickle')
        fertility_cache = os.path.join(args.ar_feat_cache, 'fertility_cache_id.pickle')

        if os.path.isfile(freq_cache):
            print('loading freq cnt from cache')
            with open(freq_cache, 'rb') as pf:
                freq_cnt = pickle.load(pf)
        else:
            raise ValueError('frequency cache file not existing')

        if os.path.isfile(fertility_cache):
            print('loading fertility cnt from cache')
            with open(fertility_cache, 'rb') as pf:
                fertility_cnt = pickle.load(pf)
        else:
            raise ValueError('fertility cache file not existing')

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()

        if args.save_knnlm_dstore:
            print('keytype being saved:', args.knn_keytype)
            if args.dstore_fp16:
                print('Saving fp16')
                dstore_keys = np.memmap(args.dstore_mmap+f'_size{args.dstore_size}_embed{args.decoder_embed_dim}_fp16_keys.npy', dtype=np.float16, mode='w+', shape=(args.dstore_size, args.decoder_embed_dim))
                dstore_vals = np.memmap(args.dstore_mmap+f'_size{args.dstore_size}_embed{args.decoder_embed_dim}_fp16_vals.npy', dtype=np.int, mode='w+', shape=(args.dstore_size, 1))
            else:
                print('Saving fp32')
                dstore_keys = np.memmap(args.dstore_mmap+f'_size{args.dstore_size}_embed{args.decoder_embed_dim}_fp32_keys.npy', dtype=np.float32, mode='w+', shape=(args.dstore_size, args.decoder_embed_dim))
                dstore_vals = np.memmap(args.dstore_mmap+f'_size{args.dstore_size}_embed{args.decoder_embed_dim}_fp32_vals.npy', dtype=np.int, mode='w+', shape=(args.dstore_size, 1))

        dstore_idx = 0
        for ex_i, sample in enumerate(t):
            if 'net_input' not in sample:
                continue

            sample = utils.move_to_cuda(sample) if use_cuda else sample

            gen_timer.start()
            # print('scorer generate')
            if args.knnlm:
                hypos = scorer.generate(models, sample, knn_dstore=knn_dstore)
            else:
                hypos = scorer.generate(models, sample)
            gen_timer.stop(sample['ntokens'])

            for i, hypos_i in enumerate(hypos):
                hypo = hypos_i[0]
                if args.save_knnlm_dstore:
                    shape = hypo['dstore_keys'].shape
                    # import pdb; pdb.set_trace()
                    if args.sample_break_mode == 'eos':
                        shape = [hypo['tokens'].size(0)]

                    if (shape[0] == args.tokens_per_sample or args.sample_break_mode == 'eos'):
                        if dstore_idx + shape[0] > args.dstore_size:
                            shape = [args.dstore_size - dstore_idx]
                            hypo['dstore_keys'] = hypo['dstore_keys'][:shape[0]]
                        if args.dstore_fp16:
                            dstore_keys[dstore_idx:shape[0]+dstore_idx] = hypo['dstore_keys'][:shape[0]].view(
                                -1, args.decoder_embed_dim).cpu().numpy().astype(np.float16)
                            dstore_vals[dstore_idx:shape[0]+dstore_idx] = hypo['tokens'].view(
                                -1, 1).cpu().numpy().astype(np.int)
                        else:
                            dstore_keys[dstore_idx:shape[0]+dstore_idx] = hypo['dstore_keys'][:shape[0]].view(
                                -1, args.decoder_embed_dim).cpu().numpy().astype(np.float32)
                            dstore_vals[dstore_idx:shape[0]+dstore_idx] = hypo['tokens'].view(
                                -1, 1).cpu().numpy().astype(np.int)

                        dstore_idx += shape[0]
                    else:
                        print('Skipping this one with shape', shape)

                sample_id = sample['id'][i]

                tokens = hypo['tokens']
                tgt_len = tokens.numel()
                pos_scores = hypo['positional_scores'].float()

                if args.add_bos_token:
                    assert hypo['tokens'][0].item() == task.target_dictionary.bos()
                    tokens = tokens[1:]
                    pos_scores = pos_scores[1:]

                if args.save_feature is not None:
                    # hypo_out = {'string': [task.target_dictionary[t.item()] for t in hypo['tokens']],
                    #     'tokens': hypo['tokens'].tolist(),
                    #     'positional_scores': hypo['positional_scores'].tolist(),
                    #     'knn_scores': hypo['knn_scores'].tolist(),
                    #     'lm_scores': hypo['lm_scores'].tolist(),
                    #     'lm_entropy': hypo['lm_entropy'].tolist(),
                    #     'lm_max': hypo['lm_max'].tolist(),
                    #     'lm_context': hypo['lm_context'].tolist(),
                    #     'knn_dists': hypo['knn_dists'].tolist(),
                    #     }

                    # import pdb; pdb.set_trace()
                    # reset prev every sentence
                    # this is only used when the dataset uses eos as the
                    # boundary, e.g. the law-MT dataset in the paper
                    # this is commented out for wikitext-103
                    # prev = [task.target_dictionary.index('</s>')] * ngram

                    for k in range(len(hypo['tokens'])):
                        # tok = task.target_dictionary[hypo['tokens'][k].item()]
                        tok = hypo['tokens'][k].item()
                        prev = prev[-ngram:]
                        hypo_others_tmp = {'s': task.target_dictionary[tok],
                                    't': tok,
                                    'int_s': hypo['positional_scores'][k].item(),
                                    'knn_s': hypo['knn_scores'][k].item() if hypo['knn_scores'] is not None else None,
                                    'lm_s': hypo['lm_scores'][k].item(),
                                    'lm_ent': hypo['lm_entropy'][k].item(),
                                    'lm_max': np.exp(hypo['lm_max'][k].item()),
                                    'freq': [freq_cnt[tuple(prev[-j:])] for j in range(1, ngram + 1)],
                                    'fert': [fertility_cnt[tuple(prev[-j:])] for j in range(1, ngram + 1)],
                                    # 'knn_dists': hypo['knn_dists'][k].tolist(),
                            }
                        if args.analyze_knn:
                            hypo_others_tmp.update({
                                'old_d': hypo['knn_old_dists'][k].tolist(),
                                'new_d': hypo['knn_new_dists'][k].tolist(),
                                'knn': task.target_dictionary.string(hypo['knn_ids'][k]),
                                })
                        prev.append(tok)

                        hypo_ctxt_tmp = {'ctxt': hypo['lm_context'][k].tolist()}
                        fout_ctxt.write(json.dumps(hypo_ctxt_tmp, ensure_ascii=False))
                        fout_ctxt.write('\n')
                        fout_ctxt.flush()

                        fout_extra.write(json.dumps(hypo_others_tmp, ensure_ascii=False))
                        fout_extra.write('\n')
                        fout_extra.flush()

                skipped_toks = 0
                if bpe_toks is not None:
                    for i in range(tgt_len - 1):
                        if tokens[i].item() in bpe_toks:
                            skipped_toks += 1
                            pos_scores[i + 1] += pos_scores[i]
                            pos_scores[i] = 0

                #inf_scores = pos_scores.eq(float('inf')) | pos_scores.eq(float('-inf'))
                #if inf_scores.any():
                #    logger.info(
                #        'skipping tokens with inf scores:',
                #        task.target_dictionary.string(tokens[inf_scores.nonzero()])
                #    )
                #    pos_scores = pos_scores[(~inf_scores).nonzero()]
                # import pdb; pdb.set_trace()
                score_sum += pos_scores.sum().cpu()
                count += pos_scores.numel() - skipped_toks

                if args.output_word_probs or args.output_word_stats:
                    w = ''
                    word_prob = []
                    is_bpe = False
                    for i in range(len(tokens)):
                        w_ind = tokens[i].item()
                        w += task.source_dictionary[w_ind]
                        if bpe_toks is not None and w_ind in bpe_toks:
                            w = w[:-bpe_len]
                            is_bpe = True
                        else:
                            word_prob.append((w, pos_scores[i].item()))

                            next_prob = None
                            ind = i + 1
                            while ind < len(tokens):
                                if pos_scores[ind].item() != 0:
                                    next_prob = pos_scores[ind]
                                    break
                                ind += 1

                            word_stats.setdefault(w, WordStat(w, is_bpe)).add(pos_scores[i].item(), next_prob)
                            is_bpe = False
                            w = ''
                    if args.output_word_probs:
                        logger.info(
                            str(int(sample_id)) + " "
                            + ('\t'.join('{} [{:2f}]'.format(x[0], x[1]) for x in word_prob))
                        )

            wps_meter.update(sample['ntokens'])
            t.log({'wps': round(wps_meter.avg)})

    if args.save_knnlm_dstore:
        print("dstore_idx", dstore_idx, "final shape", shape)
        print("Keys", dstore_keys.shape, dstore_keys.dtype)
        print("Vals", dstore_vals.shape, dstore_vals.dtype)

    avg_nll_loss = -score_sum / count / math.log(2)  # convert to base 2
    logger.info(f'count {count} tokens')
    logger.info('Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)'.format(
        gen_timer.n, gen_timer.sum, 1. / gen_timer.avg
    ))
    logger.info('Loss (base 2): {:.4f}, Perplexity: {:.2f}'.format(
        avg_nll_loss, 2**avg_nll_loss
    ))

    if args.output_word_stats:
        for ws in sorted(word_stats.values(), key=lambda x: x.count, reverse=True):
            logger.info(ws)


def cli_main():
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
