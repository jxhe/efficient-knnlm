#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=1-1%1
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-36
##SBATCH —nodelist=compute-0-31,compute-0-30

taskid=${SLURM_ARRAY_TASK_ID}

DATE=`date +%Y%m%d`

declare -a lmbda_list=(0.3 0.25 0.2 0.1 0.05)
# arglen=${#fold_list[@]}
# i=$(( taskid%arglen ))
lmbda=${lmbda_list[$taskid]}
lmbd=0.25

declare -a split_list=("valid" "test")

dstore_size=101010281
dstore_file="dstore/heldout600/dstore_size101010281_embed1024_fp16"
index_file="dstore/heldout600/knn.101010281.index"
dataset="wikitext-103-heldout600"
ckpt="/projects/tir4/users/junxianh/projects/knnlmXS/checkpoint/wikitext-103-lm/20210419/wikitext-103-heldout600/checkpoint_best.pt"
# dstore_size=101010281
# dstore_file="dstore/heldout600/dstore_size101010281_embed1024_fp16"
# index_file="dstore/heldout600/knn.101010281.index"
# dataset="wikitext-103-heldout600"
# ckpt="knnlm_ckpt/wt103_checkpoint_best.pt"
split=${split_list[$taskid]}

python eval_lm.py data-bin/${dataset} \
    --path ${ckpt} \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset ${split} --dstore-filename ${dstore_file} \
    --indexfile ${index_file} \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda ${lmbda} --dstore-size ${dstore_size} --knn-keytype last_ffn_input \
    --probe 32 --fp16 --dstore-fp16 --no-load-keys --knn-sim-func "do_not_recomp_l2" \
    # --write-distribution dstore/heldout600/${split}_full_lm --moe-feat-cache datasets/wikitext-103/heldout600 \
    # --knnlm \
    # --dstore-weight \

