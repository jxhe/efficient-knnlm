#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --mem=50g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

jobid=${SLURM_ARRAY_JOB_ID}
taskid=${SLURM_ARRAY_TASK_ID}

declare -a split_list=("valid" "test")

split=${split_list[$taskid]}
ckpt='wmtnc_lm_ckpt/wmt19.en/model.pt'
dataset='law'
index_file='dstore/law/knn.19048862.index'
dstore_file='dstore/law/dstore_size19068709_embed1536_fp16'
dstore_size=19068709
max_tokens=2048
ctxt=0
lmbda=0.25
mode="eos"

bash knnlm_scripts/utils_cmd/save_moe_feature.sh \
    -d ${dataset} \
    -c ${ckpt} \
    -s ${split} \
    -p ${dstore_file} \
    -i ${index_file} \
    -n ${dstore_size} \
    -l ${lmbda} \
    -m ${max_tokens} \
    -t ${ctxt} \
    -o ${mode}

