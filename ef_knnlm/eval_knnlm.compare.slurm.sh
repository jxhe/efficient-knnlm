#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-0%1
#SBATCH --gres=gpu:3090:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=32
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

taskid=${SLURM_ARRAY_TASK_ID}

DATE=`date +%Y%m%d`

# declare -a fold_list=(0 1 2 3 4)
# arglen=${#fold_list[@]}
# i=$(( taskid%arglen ))

dstore_size=103225485
dstore_file="dstore/dstore_size103225485_embed1024_fp16"
index_file="dstore/knn.default.index"

python eval_lm.py data-bin/wikitext-103 \
    --path knnlm_ckpt/wt103_checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset test --dstore-filename ${dstore_file} \
    --indexfile ${index_file} \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.05 --dstore-size ${dstore_size} --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 --no-load-keys --knn-sim-func "do_not_recomp_l2" \
    --write-distribution dstore/kmeans/test.full_dstore.baseline --analyze-knn --moe-feat-cache datasets/wikitext-103 \
    # --dstore-weight
