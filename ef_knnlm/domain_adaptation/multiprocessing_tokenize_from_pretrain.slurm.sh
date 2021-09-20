#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-2%3
#SBATCH --mem=10g
#SBATCH --cpus-per-task=10
#SBATCH -t 0
##SBATCH --nodelist=compute-0-3
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15

taskid=${SLURM_ARRAY_TASK_ID}


datadir="datasets/law"
declare -a split_list=("train" "dev" "test")

split=${split_list[$taskid]}
inputs="${datadir}/${split}.en"
outputs="${datadir}/${split}.en.tokenized"
bpe="wmtnc_lm_ckpt/wmt19.en/bpecodes"

python knnlm_scripts/multiprocessing_tokenize_from_pretrain.py --inputs ${inputs} --outputs ${outputs} --keep-empty --bpe-codes ${bpe}

