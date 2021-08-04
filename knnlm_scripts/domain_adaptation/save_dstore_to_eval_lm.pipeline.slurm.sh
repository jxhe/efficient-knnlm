#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-0%1
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

jobid=${SLURM_ARRAY_JOB_ID}
taskid=${SLURM_ARRAY_TASK_ID}

dataset="law"
dstore_size=19068709
actual_dstore_size=19048862
max_tokens=2048
ctxt=0
tokens_per_sample=512
embed=1536
output="dstore/${dataset}/dstore"
ckpt="wmtnc_lm_ckpt/wmt19.en/model.pt"
mode="eos"

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

prefix="${output}_size${dstore_size}_embed${embed}_fp16"
keys="${prefix}_keys.npy"
vals="${prefix}_vals.npy"
index="dstore/${dataset}/knn.${actual_dstore_size}.index"

echo "build datastore to ${index}"

bash knnlm_scripts/utils_cmd/build_faiss.sh \
    ${dstore_size} \
    ${actual_dstore_size} \
    ${keys} \
    ${vals} \
    ${index} \
    ${embed} \

echo "evaluate knnlm"

bash knnlm_scripts/utils_cmd/eval_knnlm.sh \
    ${dstore_size} \
    ${prefix} \
    ${index} \
    ${dataset} \
    ${ckpt} \
    1 \
    ${max_tokens} \
    ${ctxt} \
    ${mode} \

echo "evaluate NLM"

max_tokens=2048
ctxt=0

bash knnlm_scripts/utils_cmd/eval_lm_sent.sh \
    ${dataset} \
    ${ckpt} \
    test \
    ${max_tokens} \
    ${ctxt} \
