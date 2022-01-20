import torch
import faiss
import math
import ctypes
import numpy as np

from torch.cuda.amp import autocast

from fairseq import utils, options
import time
from fairseq.data import Dictionary

from moe_modules import MLPMOE


class KNN_Dstore(object):
    def __init__(self, args):
        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.drop_top1 = args.drop_top1
        self.knn_temp = args.knn_temp

        self.index = self.setup_faiss(args)

        if args.ar_ckpt != '' and args.ar_ckpt != 'none':
            self.moe = self.setup_moe(args.ar_ckpt, args.ar_cutoff)
        else:
            self.moe = None

    def setup_moe(self, ckpt_path, ar_cutoff=50):
        ckpt_moe = torch.load(ckpt_path)

        moe_args = ckpt_moe['args']
        moe_epoch = ckpt_moe['epoch']
        self.moe_threshold = ckpt_moe['threshold'][ar_cutoff]
        # self.moe_threshold=999
        moe_model = MLPMOE(
            feature_size=moe_args.feature_size,
            hidden_units=moe_args.hidden_units,
            nlayers=moe_args.nlayers,
            dropout=moe_args.dropout,
            activation=moe_args.activation,
            )

        moe_model.load_state_dict(ckpt_moe['param'])

        print(f'loaded models at epoch {moe_epoch} from {ckpt_path}')
        print(f'cutoff {ar_cutoff}, threshod: {self.moe_threshold}')

        if torch.cuda.is_available():
            print('use cuda')
            # moe_model = moe_model.half()
            moe_model.cuda()

        moe_model.eval()

        return moe_model

    def setup_faiss(self, args):
        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print('Reading datastore took {} s'.format(time.time() - start))
        index.nprobe = args.probe

        if options.eval_bool(args.gpu_index):
            print('gpu faiss index')
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index, co)

            index.nprobe = args.probe

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        else:
            print('Keys are fp32 and vals are int')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))

        if options.eval_bool(args.dstore_weight):
            self.weights = np.memmap(args.dstore_filename+'_weights.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        else:
            self.weights = None

        if hasattr(self, 'keys'):
            # from https://github.com/numpy/numpy/issues/13172
            # to speed up access to np.memmap
            madvise = ctypes.CDLL("libc.so.6").madvise
            madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            madvise.restype = ctypes.c_int
            assert madvise(self.keys.ctypes.data, self.keys.size * self.keys.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 1 means MADV_RANDOM

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension), dtype=np.float16 if args.dstore_fp16 else np.float32)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.vals
            self.vals_from_memmap = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int)
            self.vals = self.vals_from_memmap[:]
            self.vals = self.vals.astype(np.int)
            print('Loading to memory took {} s'.format(time.time() - start))

        return index


    def get_knns(self, queries):
        dists, knns = self.index.search(queries.detach().cpu().float().numpy(), self.k)

        # test the overhead from data communication
        # queries = queries.detach().cpu().float().numpy()
        # bsz, feat = queries.shape
        # dists = np.random.rand(bsz, self.k)
        # knns = np.random.randint(103225480, size=(bsz, self.k))
        # knns = np.random.randint(19048862, size=(bsz, self.k))

        # TODO: this may be an ok way to avoid retrieving itself, but not guranteed due
        # to the aproximation, needs to be carefully checked
        if self.drop_top1:
            dists = dists[:, 1:]
            knns = knns[:, 1:]

        return dists, knns


    def get_knn_log_prob(self,
                         queries,
                         tgt,
                         pad_idx,
                         return_knn=False,
                         freq=None,
                         fert=None,
                         lm_entropy=None,
                         lm_max=None):
        # print('get knn log prob')
        def dist_func(d, k, q, function=None):
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = q.shape
                # import pdb; pdb.set_trace()
                if self.metric_type == 'l2':
                    start = time.time()
                    # knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                    # import pdb; pdb.set_trace()
                    # if self.half:
                    #     knns_vecs = knns_vecs.half()
                    # query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                    # l2 = torch.sum((query_vecs - knns_vecs.detach())**2, dim=2)

                    # added by junxian
                    # perform distance recomputation on cpu to avoid gpu oom
                    knns_vecs = torch.from_numpy(self.keys[k]).view(qsize[0], self.k, -1)
                    # if self.half:
                    #     knns_vecs = knns_vecs.half()
                    query_vecs = q.cpu().view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                    l2 = torch.sum(((query_vecs - knns_vecs).float())**2, dim=2)
                    l2 = l2.cuda()
                    return -1 * l2
                return d

            if function == 'dot':
                qsize = q.shape
                return (torch.from_numpy(self.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

            if function == 'do_not_recomp_l2':
                return -1 * d

            raise ValueError("Invalid knn similarity function!")

        # import pdb; pdb.set_trace()
        # queries  are TxBxC
        # reshape: (TxB)xC
        # import pdb; pdb.set_trace()
        qshape = queries.shape
        queries = queries.view(-1, qshape[-1])
        # tgt = tgt.contiguous().view(-1)
        tgt = tgt.view(-1)
        start = time.time()

        if self.moe is not None:
            # import pdb; pdb.set_trace()
            # log weight of lm
            log_moe_lmw = self.moe({
                'ctxt': queries.float(),
                'lm_ent': lm_entropy.view(-1, 1).float() if lm_entropy is not None else None,
                'lm_max': lm_max.view(-1, 1).float() if lm_max is not None else None,
                'freq': freq.view(-1, freq.size(-1)) if freq is not None else None,
                'fert': fert.view(-1, fert.size(-1)) if fert is not None else None,
                })
            # with autocast():
            #     log_moe_lmw = self.moe({
            #         'ctxt': queries,
            #         'lm_ent': lm_entropy.view(-1, 1),
            #         'lm_max': lm_max.view(-1, 1),
            #         'freq': freq.view(-1, freq.size(-1)) if freq is not None else None,
            #         'fert': fert.view(-1, fert.size(-1)) if fert is not None else None,
            #         })
            # log_moe_lmw = self.moe({
            #     'ctxt': queries,
            #     'lm_ent': lm_entropy.view(-1, 1),
            #     'lm_max': lm_max.view(-1, 1),
            #     'freq': freq.view(-1, freq.size(-1)) if freq is not None else None,
            #     'fert': fert.view(-1, fert.size(-1)) if fert is not None else None,
            #     })

            log_moe_lmw = log_moe_lmw[:, 0]

            # only perform retrieval when this is true
            retrieval_mask = (log_moe_lmw < self.moe_threshold)

            knn_mask = torch.logical_and(retrieval_mask, tgt != pad_idx)

            # another althernative implementation
            # log_moe_lmw = queries.new_full((queries.size(0), ), 1e4, dtype=torch.float)
            # log_moe_lmw[tgt != pad_idx] = self.moe({
            #     'ctxt': queries[tgt != pad_idx].float()
            #     })[:, 0]

            # # only perform retrieval when this is true
            # retrieval_mask = (log_moe_lmw < self.moe_threshold)

            # knn_mask = retrieval_mask

        else:
            log_moe_lmw = retrieval_mask =  None
            knn_mask = (tgt != pad_idx)

        dists, knns = self.get_knns(queries[knn_mask])
        # print(f'retrieval consumes {time.time() - start} seconds')
        # (T_reducedxB)xK
        dists = torch.from_numpy(dists).cuda()
        start = time.time()
        dists = dist_func(dists, knns, queries[knn_mask, :], function=self.sim_func)

        orig_dists = dists
        if self.weights is not None:
            # import pdb;pdb.set_trace()
            weights = dists.new_tensor(self.weights[knns]).squeeze(-1)
            dists = dists + torch.log(weights)

        # print(f'computing distance consumes {time.time() - start} seconds')
        probs = utils.log_softmax(dists / self.knn_temp, dim=-1)

        index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1), tgt[knn_mask].unsqueeze(-1)).float()
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone()
        full_yhat_knn_prob = torch.full([qshape[0]*qshape[1]], -10000, dtype=yhat_knn_prob.dtype).cuda()
        full_yhat_knn_prob[knn_mask] = yhat_knn_prob

        if log_moe_lmw is not None:
            log_moe_lmw = log_moe_lmw.view(qshape[0], qshape[1])
            retrieval_mask = retrieval_mask.view(qshape[0], qshape[1]).float()

        # import pdb; pdb.set_trace()
        if return_knn:
            full_dists = dists.new_full([qshape[0]*qshape[1], orig_dists.size(-1)], -10000)
            full_dists[knn_mask] = -orig_dists
            full_dists = full_dists[:, :10]

            new_dists = dists.new_full([qshape[0]*qshape[1], dists.size(-1)], -10000)
            new_dists[knn_mask] = -dists
            new_dists = new_dists[:, :10]

            knns = self.vals[knns[:, :10]].squeeze(-1)
            full_knns = dists.new_full([qshape[0]*qshape[1], knns.shape[1]], -10000, dtype=torch.int)
            full_knns[knn_mask] = dists.new_tensor(knns, dtype=torch.int)

            # import pdb; pdb.set_trace()
            return full_yhat_knn_prob.view(qshape[0], qshape[1], 1), log_moe_lmw, retrieval_mask, full_dists.view(qshape[0], qshape[1], -1), \
                    new_dists.view(qshape[0], qshape[1], -1), full_knns.view(qshape[0], qshape[1], -1)

        else:
            # return dists for analysis purpose
            # TxBx1
            return full_yhat_knn_prob.view(qshape[0], qshape[1], 1), log_moe_lmw, retrieval_mask, None, None, None

