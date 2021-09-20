#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-0%1
#SBATCH --mem=20g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --nodelist=compute-0-3
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15

taskid=${SLURM_ARRAY_TASK_ID}

dstore_prefix="dstore/dstore_size103225485_embed1024_fp16"
dstore_size=103225485
dim=1024
retrieval_dir="dstore/filter_compress"


python knnlm_scripts/dstore_compression/filter_compression.py \
    --dstore-prefix ${dstore_prefix} \
    --dstore-size ${dstore_size} \
    --dimension ${dim} \
    --dstore-fp16 \
    --retrieval-dir ${retrieval_dir} \