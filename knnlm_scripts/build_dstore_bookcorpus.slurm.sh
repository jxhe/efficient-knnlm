#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=3-3%1
#SBATCH --gres=gpu:3090:1
#SBATCH --mem=20g
#SBATCH --cpus-per-task=10
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

taskid=${SLURM_ARRAY_TASK_ID}

DATE=`date +%Y%m%d`

declare -a context_list=(2 4 10 100)
arglen=${#context_list[@]}
i=$(( taskid%arglen ))


# ctxt=${context_list[$i]}
# ctxt=10
# tokens_per_sample=$(( ctxt+1 ))

ctxt=1536
tokens_per_sample=1536
dstore_size=584548313

python eval_lm.py data-bin/bookcorpus \
    --path wmtnc_lm_ckpt/wmt19.en/model.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window ${ctxt} --tokens-per-sample ${tokens_per_sample} \
    --dstore-mmap dstore/bookcorpus --knn-keytype 'last_ffn_input' \
    --dstore-size ${dstore_size}  \
    --log-interval 100 \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --fp16 --dstore-fp16 \
    --save-knnlm-dstore \

