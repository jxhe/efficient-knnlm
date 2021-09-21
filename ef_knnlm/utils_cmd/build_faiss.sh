#!/bin/bash


size=$1
actual_size=$2
keys=$3
vals=$4
index_name=$5
dimension=${6:-1024}
dstore_dir=${7:-""}
pca=${8:-512}
infer_prefix=${9:-"dstore"}


python build_dstore.py \
    --dstore_keys ${keys} \
    --dstore_vals ${vals} \
    --dstore_size ${size} \
    --dstore_dir ${dstore_dir} \
    --faiss_index ${index_name} \
    --actual_dstore_size ${actual_size} \
    --dimension ${dimension} \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 \
    --infer_prefix ${infer_prefix} \
    --pca ${pca} \
    --dstore_fp16
