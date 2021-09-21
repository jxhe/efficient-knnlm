#!/bin/bash

dstore_prefix="dstore/dstore_size103225485_embed1024_fp16"
dstore_size=103225485
index="dstore/knn.103225485.pca512.m64.index"
k=30
dimension=1024
# interval=10000
save_dir="dstore/greedy_merge"


start=0
save="${save_dir}/retrieve_results_start${start}"

echo "traverse from position ${start}"

mkdir -p ${save_dir}

python knnlm_scripts/dstore_compression/save_retrieval_results.py \
    --dstore-prefix ${dstore_prefix} \
    --dstore-size ${dstore_size} \
    --actual-dstore-size ${actual_dstore_size} \
    --index ${index} \
    --k $k \
    --dstore-fp16 \
    --dimension ${dimension} \
    --save ${save} \
    --start-point ${start} \
