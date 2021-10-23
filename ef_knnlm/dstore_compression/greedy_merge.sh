#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-5%6
#SBATCH --mem=50g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --nodelist=compute-0-3
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15

# declare -a k_list=(1 3 7 9 12 15)

dstore_prefix="dstore/dstore_size103225485_embed1024_fp16"
dstore_size=103225485
dim=1024
retrieval_dir="dstore/greedy_merge"
save_dir="dstore/wikitext-103/greedy_merge"
k=9


python knnlm_scripts/dstore_compression/greedy_merge.py \
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
    ${save_dir}/knn.merge${k}.pca512.m64.index \
    ${dim} \
    ${save_dir} \
    512 \
    dstore_merge${k}_ \


