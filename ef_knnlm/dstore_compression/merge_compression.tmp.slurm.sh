#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-2%3
#SBATCH --mem=32g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --nodelist=compute-0-3
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15

taskid=${SLURM_ARRAY_TASK_ID}

declare -a k_list=(30 45 60)

dstore_prefix="dstore/dstore_size103225485_embed1024_fp16"
dstore_size=103225485
dim=1024
retrieval_dir="dstore/filter_compress"
save_dir="/projects/tir3/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress"
k=${k_list[$taskid]}


python knnlm_scripts/dstore_compression/merge_compression.py \
    --dstore-prefix ${dstore_prefix} \
    --dstore-size ${dstore_size} \
    --dimension ${dim} \
    --dstore-fp16 \
    --retrieval-dir ${retrieval_dir} \
    --save-dir ${save_dir} \
    --k ${k} \


bash knnlm_scripts/utils_cmd/build_faiss.sh \
    0 \
    0 \
    none \
    none \
    ${save_dir}/knn.merge${k}.index \
    ${dim} \
    ${save_dir} \
    dstore_merge${k} \


