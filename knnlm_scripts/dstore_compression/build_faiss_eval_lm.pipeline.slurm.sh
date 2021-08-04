#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=2-3%2
#SBATCH --gres=gpu:1
#SBATCH --mem=32g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

jobid=${SLURM_ARRAY_JOB_ID}
taskid=${SLURM_ARRAY_TASK_ID}

declare -a size_list=(20645097 41290194 61935291 82580388)

dataset="wikitext-103"
dstore_size=${size_list[$taskid]}
actual_dstore_size=${dstore_size}
max_tokens=3072
ctxt=2560
tokens_per_sample=1536
embed=1024
output="dstore/filter_compress"
ckpt="knnlm_ckpt/wt103_checkpoint_best.pt"
mode="complete"

# echo "save datastore to ${output}"

# bash knnlm_scripts/utils_cmd/save_dstore.sh \
#     ${dataset} \
#     ${dstore_size} \
#     ${max_tokens} \
#     ${ctxt} \
#     ${tokens_per_sample} \
#     ${output} \
#     ${ckpt} \
#     ${mode} \

prefix="${output}/dstore_scorefilter_size${dstore_size}_embed${embed}"
keys="${prefix}_keys.npy"
vals="${prefix}_vals.npy"
index_dir="/projects/tir5/users/junxianh/knnlmXS/dstore/${dataset}/filter_compress"

mkdir -p ${index_dir}

index="${index_dir}/knn.${actual_dstore_size}.index"

# echo "build datastore to ${index}"

# bash knnlm_scripts/utils_cmd/build_faiss.sh \
#     ${dstore_size} \
#     ${actual_dstore_size} \
#     ${keys} \
#     ${vals} \
#     ${index} \
#     ${embed} \

echo "evaluate knnlm"

bash knnlm_scripts/utils_cmd/eval_knnlm.sh \
    -n ${dstore_size} \
    -p ${prefix} \
    -i ${index} \
    -d ${dataset} \
    -c ${ckpt} \
    -e 1 \
    -m ${max_tokens} \
    -t ${ctxt} \
    -o ${mode} \

# echo "evaluate NLM"

# max_tokens=2048
# ctxt=0

# bash knnlm_scripts/utils_cmd/eval_lm_sent.sh \
#     ${dataset} \
    # ${ckpt} \
    # test \
    # ${max_tokens} \
    # ${ctxt} \
