#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0%1
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=16
#SBATCH --exclude=compute-0-31,compute-0-36
#SBATCH -t 0
##SBATCH —nodelist=compute-0-31,compute-0-30

taskid=${SLURM_ARRAY_TASK_ID}

DATE=`date +%Y%m%d`

declare -a lmbda_list=(0.3 0.25 0.2 0.1 0.05)
# arglen=${#fold_list[@]}
# i=$(( taskid%arglen ))
lmbda=${lmbda_list[$taskid]}

dstore_size=101010281


python eval_lm.py data-bin/wikitext-103-heldout600 \
    --path /projects/tir4/users/junxianh/projects/knnlmXS/checkpoint/wikitext-103-lm/20210419/wikitext-103-heldout600/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap dstore/heldout600/dstore --knn-keytype 'last_ffn_input' \
    --dstore-size ${dstore_size} --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16

