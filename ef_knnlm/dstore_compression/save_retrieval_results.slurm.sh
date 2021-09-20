#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-9%10
#SBATCH --mem=16g
#SBATCH --cpus-per-task=8
#SBATCH -t 0
##SBATCH --nodelist=compute-0-3
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15

taskid=${SLURM_ARRAY_TASK_ID}
# taskid=1

dstore_prefix="dstore/dstore_size103225485_embed1024_fp16"
dstore_size=103225485
index="dstore/knn.default.index"
k=30
interval=10000000
# interval=10000
save_dir="dstore/merge_compression"

# declare -a start_list=($(seq 0 ${interval} ${dstore_size}))

# start=${start_list[$taskid]}
# save="${save_dir}/retrieve_results_start${start}"

# dstore_prefix="dstore/law/dstore_size19068709_embed1536_fp16"
# dstore_size=19068709
# index="dstore/law/knn.19048862.index"
# actual_dstore_size=19048862
# k=512
# interval=2000000
# dimension=1536
# # interval=10000
# save_dir="/projects/tir5/users/junxianh/knnlmXS/dstore/law/merge_compression"

declare -a start_list=($(seq 0 ${interval} ${dstore_size}))

start=${start_list[$taskid]}
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
    --num ${interval} \
