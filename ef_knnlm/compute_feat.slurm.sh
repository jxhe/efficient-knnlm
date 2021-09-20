#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-0%1
#SBATCH --gres=gpu:1
#SBATCH --mem=50g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

jobid=${SLURM_ARRAY_JOB_ID}
taskid=${SLURM_ARRAY_TASK_ID}

DATE=`date +%Y%m%d`

split='test'
dstore='dstore/kmeans/new_no_sampling/dstore_kmeans_size20700000_embed1024_fp16'
ckpt='knnlm_ckpt/wt103_checkpoint_best.pt'

python eval_lm.py data-bin/wikitext-103  \
    --path ${ckpt} \
    --sample-break-mode complete --max-tokens 3072  --context-window 2560  --softmax-batch 1024 \
    --gen-subset ${split} --dstore-filename ${dstore}  --indexfile dstore/kmeans/new_no_sampling/knn.kmeans20342283.index \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}"  --k 1024 \
    --lmbda 0.25 --dstore-size 20700000 --knn-keytype last_ffn_input --probe 32 --knnlm \
    --dstore-fp16 --fp16 --knn-sim-func do_not_recomp_l2 \
    --write-distribution dstore/kmeans/new_no_sampling/${split}_feat  \
    --moe-feat-cache datasets/wikitext-103 \
    --dstore-weight \
    # --drop-top1 \

