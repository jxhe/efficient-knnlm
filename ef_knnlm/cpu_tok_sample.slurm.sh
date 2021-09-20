#!/bin/bash
#SBATCH --output=slurm_out/kmeans/slurm-%A_%a.out
#SBATCH --error=slurm_out/kmeans/slurm-%A_%a.err
#SBATCH --array=4,5,6,7,8,9,10%7
#SBATCH --mem=50g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --nodelist=compute-1-7
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15

taskid=${SLURM_ARRAY_TASK_ID}
tokid=${taskid}
# taskid=10

python knnlm_scripts/compress_dstore_with_sampling.py --tokid ${tokid} --dstore-keys dstore/dstore_size103225485_embed1024_fp16_keys.npy --dstore-size 103225485 --tok2pos datasets/wikitext-103/tok2pos.dict.pickle --output tok_sampling --rate 100

