#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-5%6
#SBATCH --mem=50g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --nodelist=compute-0-3
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15

taskid=${SLURM_ARRAY_TASK_ID}

# declare -a k_list=(1 3 7 9 12 15)

dstore_prefix="dstore/dstore_size103225485_embed1024_fp16"
dstore_size=103225485
dim=1024
retrieval_dir="dstore/filter_compress"
save_dir="dstore/wikitext-103/merge_compress"
k=${k_list[$taskid]}

# declare -a k_list=(1 5 9 15 30 60)

# dstore_prefix="dstore/law/dstore_size19068709_embed1536_fp16"
# dstore_size=19068709
# dim=1536
# retrieval_dir="/projects/tir5/users/junxianh/knnlmXS/dstore/law/merge_compression"
# save_dir="/projects/tir5/users/junxianh/knnlmXS/dstore/law/merge_compression"
# k=${k_list[$taskid]}


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
    dstore_merge${k}_ \


