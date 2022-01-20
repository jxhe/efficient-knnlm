#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=5-5%1
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

jobid=${SLURM_ARRAY_JOB_ID}
taskid=${SLURM_ARRAY_TASK_ID}

# taskid=4

# declare -a temp_list=(0.1 0.5 1)
# declare -a index_list=("dstore/knn.103225485.pca16.m8.index" \
#     "dstore/knn.103225485.pca32.m16.index" \
#     "dstore/knn.103225485.pca64.m32.index" \
#     "dstore/knn.103225485.pca128.m64.index" \
#     "dstore/knn.103225485.pca512.m64.index" \
#     "dstore/knn.103225485.pca256.index")

dstore_size=19068709
# dstore_file="dstore/law/dstore_size19068709_embed1536_fp16"
# index_file=dstore/law/knn.19048862.index
dstore_file="/projects/tir4/users/urialon/efficient-knnlm/checkpoints/law/dstore16_finetuned_size19068709_embed1536_fp16"
index_file="/projects/tir4/users/urialon/efficient-knnlm/checkpoints/law/knn_finetuned"
dataset=law
# ckpt="wmtnc_lm_ckpt/wmt19.en/model.pt"
ckpt="/projects/tir4/users/junxianh/projects/knnlmXS/checkpoint/law/20210509/checkpoint_best.pt"
split="test"
lambda=0.25
probe=32
# temp=${temp_list[$taskid]}

temp=1
# index_file=${index_list[$taskid]}

# echo "evaluate knnlm with temperature ${temp}, index ${index_file}"

bash ef_knnlm/utils_cmd/eval_knnlm.sh -n ${dstore_size} -p ${dstore_file} -i ${index_file} -d ${dataset} -c ${ckpt} -e ${temp} -s ${split} -l ${lambda} -b ${probe} -g "True"

