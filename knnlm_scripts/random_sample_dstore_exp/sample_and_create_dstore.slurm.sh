#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-3%4
#SBATCH --mem=30g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

jobid=${SLURM_ARRAY_JOB_ID}
taskid=${SLURM_ARRAY_TASK_ID}

dim=1024
dstore_size=103225485

fraction_list=(0.2 0.4 0.6 0.8)

fraction=${fraction_list[$taskid]}
num=$(printf %.0f $(echo "${dstore_size}*${fraction}" | bc -l))
output_dir="dstore/random_sample"

echo "random sample datastore to size ${num}"

python knnlm_scripts/sampling_dstore_baseline.py \
    --dstore-keys dstore/dstore_size103225485_embed1024_fp16_keys.npy \
    --dstore-vals dstore/dstore_size103225485_embed1024_fp16_vals.npy \
    --dstore-size ${dstore_size} \
    --num ${num} \
    --dim ${dim} \
    --output-dir ${output_dir}


keys="${output_dir}/sample_dstore_size${num}_dim${dim}_keys.npy"
vals="${output_dir}/sample_dstore_size${num}_dim${dim}_vals.npy"
index="${output_dir}/knn.${num}.index"

echo "building datastore for size ${num}"

bash knnlm_scripts/utils_cmd/build_faiss.sh ${num} ${num} ${keys} ${vals} ${index}
