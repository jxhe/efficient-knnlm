#!/bin/bash
#SBATCH --output=slurm_out/dbscan/slurm-%A_%a.out
#SBATCH --error=slurm_out/dbscan/slurm-%A_%a.err
#SBATCH --array=0-10%11
#SBATCH --mem=60g
#SBATCH --cpus-per-task=10
#SBATCH -t 0
##SBATCH --nodelist=compute-1-7
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15

taskid=${SLURM_ARRAY_TASK_ID}
# taskid=10

for i in $(seq 0 4)
do
    tokid=$(( i*1000+taskid ))
    python knnlm_scripts/compress_dstore_with_dbscan.py --tokid ${tokid} --dstore-keys dstore/dstore_size103225485_embed1024_fp16_keys.npy --dstore-size 103225485 --tok2pos datasets/wikitext-103/tok2pos.dict.pickle --output tok_dbscan --minpts 5 --eps 15
done

