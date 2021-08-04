#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-0%1
#SBATCH --mem=30g
#SBATCH --cpus-per-task=32
#SBATCH -t 0
##SBATCH --nodelist=compute-0-3
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15

taskid=${SLURM_ARRAY_TASK_ID}


# size=101010281
# actual_size=101010281
# keys=dstore/heldout600/dstore_size101010281_embed1024_fp16_keys.npy
# vals=dstore/heldout600/dstore_size101010281_embed1024_fp16_vals.npy
# index_name=dstore/heldout600/knn.101010281.index

size=103225485
pca=0
actual_size=103225485
keys=dstore/dstore_size103225485_embed1024_fp16_keys.npy
vals=dstore/dstore_size103225485_embed1024_fp16_vals.npy
# index_name=dstore/knn.103225485.pca${pca}.index
index_name=dstore/knn.default.index

python build_dstore.py \
    --dstore_keys ${keys} \
    --dstore_vals ${vals} \
    --dstore_size ${size} \
    --faiss_index ${index_name} \
    --actual_dstore_size ${actual_size} \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 \
    --dstore_fp16 \
    --pca ${pca} \
