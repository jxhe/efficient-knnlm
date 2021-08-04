#! /bin/bash
#
# eval_knnlm.nprob.sh
# Copyright (C) 2021-05-12 Junxian <He>
#
# Distributed under terms of the MIT license.
#


declare -a k_list=(2 4 8 16 32 64 128 256 512)

for i in "${k_list[@]}"
do
    bash knnlm_scripts/utils_cmd/eval_knnlm.sh -d wikitext-103 -s valid -p dstore/dstore_size103225485_embed1024_fp16 -i dstore/knn.default.index -c knnlm_ckpt/wt103_checkpoint_best.pt -n 103225485 -k ${i}
done

