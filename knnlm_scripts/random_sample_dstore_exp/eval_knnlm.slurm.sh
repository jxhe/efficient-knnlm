#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

jobid=${SLURM_ARRAY_JOB_ID}
taskid=${SLURM_ARRAY_TASK_ID}

dim=1024
dstore_size=103225485

fraction_list=(0.2 0.4 0.6 0.8)

fraction=${fraction_list[$taskid]}
num=$(printf %.0f $(echo "${dstore_size}*${fraction}" | bc -l))
output_dir="dstore/random_sample"

dstore_size=${num}
dstore_file="${output_dir}/sample_dstore_size${num}_dim${dim}"
index_file="${output_dir}/knn.${num}.index"
dataset="wikitext-103"
ckpt="knnlm_ckpt/wt103_checkpoint_best.pt"

echo "evalute dstore size ${num}"

bash knnlm_scripts/utils_cmd/eval_knnlm.sh ${dstore_size} ${dstore_file} ${index_file} ${dataset} ${ckpt}

