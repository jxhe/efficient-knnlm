#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-7%8
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

jobid=${SLURM_ARRAY_JOB_ID}
taskid=${SLURM_ARRAY_TASK_ID}

taskid=2

declare -a temp_list=(0.1 0.5 1 5.0)
declare -a k_list=(256 1024)
declare -a weight_list=("True" "False")

dstore_size=68683408
# dstore_file="dstore/dstore_size103225485_embed1024_fp16"
dstore_file="/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/dstore_merge5_size68683408_embed1024"
# index_file=dstore/knn.default.index
index_file="/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/knn.merge5.index"
dataset=wikitext-103
ckpt="knnlm_ckpt/wt103_checkpoint_best.pt"

i=$(( taskid % 4 ))
j=$(( taskid / 4 ))

temp=${temp_list[$i]}
k=${k_list[$j]}
dstore_weight=${weight_list[$j]}

k=1024

echo "evaluate knnlm with temperature ${temp}"

bash knnlm_scripts/utils_cmd/eval_knnlm.sh \
    -n ${dstore_size} \
    -p ${dstore_file} \
    -i ${index_file} \
    -d ${dataset} \
    -c ${ckpt} \
    -e ${temp} \
    -k ${k} \
    -w ${dstore_weight} \

