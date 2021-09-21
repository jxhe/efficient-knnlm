# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys
import numpy as np
import time

from fairseq import utils
from fairseq.data import Dictionary


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict, softmax_batch=None, compute_alignment=False, args=None):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos()
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.args = args

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        # import pdb; pdb.set_trace()
        net_input = sample['net_input']

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff, log_moe_w=None, retrieval_mask=None):
            if log_moe_w is None:
                combine_probs = torch.stack([vocab_p, knn_p], dim=0)
                coeffs = torch.ones_like(combine_probs)
                coeffs[0] = np.log(1 - coeff)
                coeffs[1] = np.log(coeff)
                curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)
            else:
                # use learned mixture of experts to perform interpolation
                lm_weights = retrieval_mask * log_moe_w
                knn_prob = 1. - torch.exp(log_moe_w)
                overflow = (knn_prob <= 0)
                knn_prob = torch.clip(knn_prob, 1e-5, 1)
                knn_weights = torch.log(knn_prob)
                knn_weights[overflow] = -1e5
                knn_weights = retrieval_mask * knn_weights + (1-retrieval_mask) * (-1e5)

                combine_probs = torch.stack([vocab_p + lm_weights, knn_p + knn_weights], dim=0)
                curr_prob = torch.logsumexp(combine_probs, dim=0)

            return curr_prob

        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        cur_time = time.time()
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            attn = decoder_out[1]
            if type(attn) is dict:
                attn = attn.get('attn', None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            lm_entropy = None
            lm_max = None

            lm_entropy_flag = lm_max_flag = False
            if self.args and self.args.save_feature is not None:
                lm_entropy_flag = lm_max_flag = True
            elif self.args.ar_ckpt != 'none':
                if 'lm_ent' in self.args.ar_feat_type:
                    lm_entropy_flag = True

                if 'lm_max' in self.args.ar_feat_type:
                    lm_max_flag = True

            for i, (bd, tgt, is_single) in enumerate(batched):
                sample['target'] = tgt
#                 import pdb; pdb.set_trace()
#                 curr_prob = model.get_normalized_probs(bd, log_probs=len(models) == 1, sample=sample).data

                # to have log prob for every word (in the adaptive softmax case)
                curr_prob = model.get_normalized_probs(bd, log_probs=len(models) == 1, sample=None).data
                # import pdb; pdb.set_trace()
                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)

                    # currently entropy computation only supports single model
                    if lm_entropy_flag:
                        # import pdb; pdb.set_trace()
                        lm_entropy = -(curr_prob.exp() * curr_prob).sum(dim=-1)

                    if lm_max_flag:
                        lm_max, _ = curr_prob.max(dim=-1)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                        if lm_entropy_flag:
                            lm_entropy = curr_prob.new(orig_target.numel())

                        if lm_max_flag:
                            lm_max = curr_prob.new(orig_target.numel())

                        # entropy todo here
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt)

                    if lm_entropy_flag:
                        # entropy_local = curr_prob.view(tgt.shape + (curr_prob.size(-1),))
                        # entropy_local = -(entropy_local.exp() * entropy_local).sum(dim=-1)
                        entropy_local = -(curr_prob.exp() * curr_prob).sum(dim=-1)
                        lm_entropy[idx:end] = entropy_local.view(-1)
                        # pass

                    if lm_max_flag:
                        # max_prob, _ = curr_prob.view(tgt.shape + (curr_prob.size(-1),)).max(dim=-1)
                        max_prob, _ = curr_prob.max(dim=-1)
                        lm_max[idx:end] = max_prob.view(-1)

                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample['target'] = orig_target

            lm_probs = probs.view(sample['target'].shape)
            probs = lm_probs

            if lm_entropy is not None:
                lm_entropy = lm_entropy.view(sample['target'].shape)

            if lm_max is not None:
                lm_max = lm_max.view(sample['target'].shape)

            # print(f'forward consumes {time.time() - cur_time} seconds')

            if self.args and self.args.save_feature is not None:
                lm_context = bd[1][self.args.knn_keytype].permute(1, 0, 2)

            else:
                lm_context = None

            # cur_time = time.time()
            if 'knn_dstore' in kwargs:
                dstore = kwargs['knn_dstore']
                # TxBxC
                queries = bd[1][self.args.knn_keytype]
                # if self.args.save_feature is not None:
                #     lm_context = queries.permute(1, 0, 2)
                # else:
                #     lm_context = None
                # import pdb; pdb.set_trace()
                if len(models) != 1:
                    raise ValueError('Only knn *log* probs are supported.')

                yhat_knn_prob, log_moe_w, retrieval_mask, knn_old_dists, knn_new_dists, knn_ids = dstore.get_knn_log_prob(
                        queries.permute(1,0,2).contiguous(),
                        orig_target,
                        pad_idx=self.pad,
                        return_knn=self.args.analyze_knn,
                        freq=sample['freq'] if 'freq' in sample else None,
                        fert=sample['fert'] if 'fert' in sample else None,
                        lm_entropy=lm_entropy if lm_entropy is not None else None,
                        lm_max=lm_max.exp() if lm_max is not None else None,
                        )

                # yhat_knn_prob = yhat_knn_prob.permute(1, 0, 2).squeeze(-1)
                yhat_knn_prob = yhat_knn_prob.squeeze(-1)
                if self.args.fp16:
                    yhat_knn_prob = yhat_knn_prob.half()
                    lm_probs = lm_probs.half()

                # print(f'knn consumes {time.time() - cur_time} seconds')

                # cur_time = time.time()
                probs = combine_knn_and_vocab_probs(
                            yhat_knn_prob, lm_probs,
                            self.args.lmbda, log_moe_w,
                            retrieval_mask)
                # print(f'interpolation consumes {time.time() - cur_time} seconds')



            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        cur_time = time.time()
        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        lm_probs_i = lm_entropy_i = lm_max_i = knn_probs_i = lm_context_i = None
        knn_old_dists_i = knn_new_dists_i = knn_ids_i = None
        # import pdb; pdb.set_trace()
        for i in range(bsz):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            if self.args and self.args.save_feature is not None:
                lm_probs_i = lm_probs[i][start_idxs[i]:start_idxs[i] + tgt_len].cpu()
                lm_entropy_i = lm_entropy[i][start_idxs[i]:start_idxs[i] + tgt_len].cpu()
                lm_max_i = lm_max[i][start_idxs[i]:start_idxs[i] + tgt_len].cpu()
                lm_context_i = lm_context[i][start_idxs[i]:start_idxs[i] + tgt_len][:].cpu()

                if self.args.analyze_knn:
                    knn_old_dists_i = knn_old_dists[i][start_idxs[i]:start_idxs[i] + tgt_len][:].cpu()
                    knn_new_dists_i = knn_new_dists[i][start_idxs[i]:start_idxs[i] + tgt_len][:].cpu()
                    knn_ids_i = knn_ids[i][start_idxs[i]:start_idxs[i] + tgt_len][:].cpu()

                if 'knn_dstore' in kwargs:
                    knn_probs_i = yhat_knn_prob[i][start_idxs[i]:start_idxs[i] + tgt_len].cpu()
                else:
                    knn_probs_i = None
            else:
                lm_probs_i = knn_probs_i = None
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample['net_input']['src_tokens'][i],
                        sample['target'][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            hypos.append([{
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_probs_i,
                'knn_scores': knn_probs_i,
                'lm_scores': lm_probs_i,
                'lm_entropy': lm_entropy_i,
                'lm_max': lm_max_i,
                'lm_context': lm_context_i,
                'knn_old_dists': knn_old_dists_i,
                'knn_new_dists': knn_new_dists_i,
                'knn_ids': knn_ids_i,
                'dstore_keys': decoder_out[1][self.args.knn_keytype][start_idxs[i]:,i,:] if self.args.save_knnlm_dstore else None,
            }])

        # print(f'processing output consumes {time.time() - cur_time} seconds')
        return hypos
