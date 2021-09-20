#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:1
#SBATCH --mem=50g
#SBATCH --cpus-per-task=10
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

jobid=${SLURM_ARRAY_JOB_ID}
taskid=${SLURM_ARRAY_TASK_ID}

# declare -a f_list=("all" "ctxt" "ctxt,freq,lm_ent,lm_max" "ctxt,lm_ent,lm_max" "ctxt,lm_ent,lm_max,fert" "freq" "fert" "lm_ent,lm_max" "freq,lm_ent,lm_max" "lm_ent,lm_max,fert")
declare -a f_list=("ctxt,fert" "ctxt,freq,fert" "ctxt,freq")

dataset="wikitext-103"
train="datasets/${dataset}/valid_feat"
val="datasets/${dataset}/test_feat"
l1=0.05
# feature="ctxt,freq,lm_ent,lm_max"
feature=${f_list[$taskid]}

bash knnlm_scripts/utils_cmd/train_moe.sh ${dataset} ${train} ${val} ${l1} ${feature}

