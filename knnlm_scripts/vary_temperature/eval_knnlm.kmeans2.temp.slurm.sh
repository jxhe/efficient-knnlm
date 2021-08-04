#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-6%4
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

jobid=${SLURM_ARRAY_JOB_ID}
taskid=${SLURM_ARRAY_TASK_ID}

declare -a temp_list=(0.1 0.5 1 5.0 10 100 1000)

dstore_size=60000000
dstore_file="dstore/kmeans/kmeans_min20_max5000/dstore_kmeans_size60000000_embed1024_fp16"
index_file="dstore/kmeans/kmeans_min20_max5000/knn.kmeans54414803.index"
dataset=wikitext-103
ckpt="knnlm_ckpt/wt103_checkpoint_best.pt"
temp=${temp_list[$taskid]}

echo "evaluate knnlm with temperature ${temp}"

bash knnlm_scripts/utils_cmd/eval_knnlm.sh ${dstore_size} ${dstore_file} ${index_file} ${dataset} ${ckpt} ${temp}

