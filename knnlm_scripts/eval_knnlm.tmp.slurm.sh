#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-0%1
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=compute-1-13
#SBATCH --exclude=compute-1-7
#SBATCH -t 0

taskid=${SLURM_ARRAY_TASK_ID}

DATE=`date +%Y%m%d`

declare -a lmbda_list=(0.25 0.1 0.05)
# arglen=${#fold_list[@]}
# i=$(( taskid%arglen ))
lmbda=${lmbda_list[$taskid]}

dstore_size=103225485
dstore_file="dstore/dstore_size103225485_embed1024_fp16"
index_file="dstore/knn.default.index"
lmbda=0.25
# dstore_size=20292283
# dstore_file="dstore/sample/dstore_sample_size20292283_dim1024_fp16"
# index_file="dstore/sample/knn.sample20292283.index"

python eval_lm.py data-bin/wikitext-103 \
    --path knnlm_ckpt/wt103_checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset test --dstore-filename ${dstore_file} \
    --indexfile ${index_file} \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda ${lmbda} --dstore-size ${dstore_size} --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 --no-load-keys --knn-sim-func "do_not_recomp_l2" \
    # --dstore-weight
