#!/bin/bash

# benchmark adaptive retrieval


# the cutoff ratio in adaptive retrieval
# by default we cut off half of the retrieval
cutoff=50
ar_ckpt=$1

bash knnlm_scripts/utils_cmd/eval_knnlm.sh \
    -d wikitext-103 \
    -s test \
    -p dstore/dstore_size103225485_embed1024_fp16 \
    -i dstore/knn.default.index \
    -c knnlm_ckpt/wt103_checkpoint_best.pt \
    -n 103225485 \
    -f datasets/wikitext-103 \
    -a ctxt,freq,lm_ent,lm_max,fert \
    -u ${cutoff} \
    -h ${ar_ckpt} \
