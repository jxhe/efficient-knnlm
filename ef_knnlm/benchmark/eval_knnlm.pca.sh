#! /bin/bash
#
# eval_knnlm.nprob.sh
# Copyright (C) 2021-05-12 Junxian <He>
#
# Distributed under terms of the MIT license.
#


declare -a temp_list=(0.1 0.5 1)
declare -a index_list=("dstore/knn.103225485.pca16.m8.index" \
    "dstore/knn.103225485.pca32.m16.index" \
    "dstore/knn.103225485.pca64.m32.index" \
    "dstore/knn.103225485.pca128.m64.index" \
    "dstore/knn.103225485.pca512.m64.index" \
    "dstore/knn.103225485.pca256.index")

dstore_size=103225485
dstore_file="dstore/dstore_size103225485_embed1024_fp16"
index_file=dstore/knn.103225485.pca256.index
dataset=wikitext-103
ckpt="knnlm_ckpt/wt103_checkpoint_best.pt"
split="valid"
temp=${temp_list[$taskid]}

temp=1
index_file=${index_list[$taskid]}

for id in "${index_list[@]}"
do
    echo "evaluate knnlm with temperature ${temp}, index ${id}"
    bash knnlm_scripts/utils_cmd/eval_knnlm.sh -n ${dstore_size} -p ${dstore_file} -i ${id} -d ${dataset} -c ${ckpt} -e ${temp} -s ${split}
done

