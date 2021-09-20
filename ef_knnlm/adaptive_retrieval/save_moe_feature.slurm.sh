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

taskid=1

declare -a split_list=("train" "valid")

split=${split_list[$taskid]}
ckpt='knnlm_ckpt/wt103_checkpoint_best.pt'
dataset='wikitext-103-heldval'
index_file='dstore/knn.default.index'
dstore_file='dstore/dstore_size103225485_embed1024_fp16'
dstore_size=103225485
max_tokens=3072
ctxt=2560
lmbda=0.25
mode="complete"

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

