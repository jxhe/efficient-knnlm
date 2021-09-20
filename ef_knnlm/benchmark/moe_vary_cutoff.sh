#! /bin/bash
#
# moe_vary_cutoff.sh
# Copyright (C) 2021-05-17 Junxian <He>
#
# Distributed under terms of the MIT license.
#


declare -a cutoff_list=(10 30 70 90)



for i in "${cutoff_list[@]}"
do
    bash knnlm_scripts/utils_cmd/eval_knnlm.sh -d wikitext-103 -s test -p dstore/dstore_size103225485_embed1024_fp16 -i dstore/knn.default.index -c knnlm_ckpt/wt103_checkpoint_best.pt -n 103225485 -f datasets/wikitext-103 -a ctxt,lm_ent,lm_max,fert -u ${i} -h /projects/tir4/users/junxianh/projects/knnlmXS/checkpoint/wikitext-103/20210516/moe/mlp.l1.0.05.ngram0.hid128.nl5.bs64.drop0.2.ftctxt_lm_ent_lm_max_fert.seed927.jobid368056.taskid0/checkpoint_best.pt
done
