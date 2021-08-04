#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-0%1
#SBATCH --gres=gpu:1
#SBATCH --mem=50g
#SBATCH --cpus-per-task=10
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

jobid=${SLURM_ARRAY_JOB_ID}
taskid=${SLURM_ARRAY_TASK_ID}


dataset="law"
train="dstore/law/valid"
val="dstore/law/test"
l1=0.05

bash knnlm_scripts/utils_cmd/train_moe.sh \
    ${dataset} \
    ${train} \
    ${val} \
    ${l1} \

