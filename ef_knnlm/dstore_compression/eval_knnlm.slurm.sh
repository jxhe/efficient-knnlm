#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-5%4
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

jobid=${SLURM_ARRAY_JOB_ID}
taskid=${SLURM_ARRAY_TASK_ID}


# declare -a temp_list=(0.1 0.5 1 5.0)
# declare -a size_list=(58623957 56119915 74504638 64795663 61900130)
# declare -a merge_k_list=(12 15 3 7 9)
# declare -a k_list=(256 1024)
# declare -a weight_list=("True" "False")

# i=$(( taskid % 5 ))
# j=$(( taskid / 5 ))

# dstore_size=${size_list[$i]}
# merge_k=${merge_k_list[$i]}
# # dstore_file="dstore/dstore_size103225485_embed1024_fp16"
# dstore_file="/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/dstore_merge${merge_k}_size${dstore_size}_embed1024"
# # index_file=dstore/knn.default.index
# index_file="/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/knn.merge${merge_k}.index"
# dataset=wikitext-103
# ckpt="knnlm_ckpt/wt103_checkpoint_best.pt"
# split="test"
# lambda=0.25


declare -a temp_list=(0.1 0.5 1 5.0)
declare -a size_list=(9037103 14761087 7809665 11241405 6710669 10023860)
declare -a merge_k_list=(15 1 30 5 60 9)
declare -a k_list=(256 1024)
declare -a weight_list=("True" "False")

i=$(( taskid % 6 ))
j=$(( taskid / 5 ))

dstore_size=${size_list[$i]}
merge_k=${merge_k_list[$i]}
# dstore_file="dstore/dstore_size103225485_embed1024_fp16"
dstore_file="/projects/tir5/users/junxianh/knnlmXS/dstore/law/merge_compression/dstore_merge${merge_k}_size${dstore_size}_embed1536"
# index_file=dstore/knn.default.index
index_file="/projects/tir5/users/junxianh/knnlmXS/dstore/law/merge_compression/knn.merge${merge_k}.index"
dataset=law
ckpt="wmtnc_lm_ckpt/wmt19.en/model.pt"
split="valid"
lambda=0.9

# temp=${temp_list[$i]}
# k=${k_list[$j]}
dstore_weight="True"

k=1024
temp=1

echo "evaluate knnlm with temperature ${temp}"

bash knnlm_scripts/utils_cmd/eval_knnlm.sh \
    -n ${dstore_size} \
    -p ${dstore_file} \
    -i ${index_file} \
    -d ${dataset} \
    -c ${ckpt} \
    -e ${temp} \
    -l ${lambda} \
    -k ${k} \
    -s ${split} \
    -w ${dstore_weight} \

