#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=3-3%1
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

declare -a pca_list=(32 64 128 256 512)
declare -a m_list=(16 32 64 64 64)

size=19068709
pca=${pca_list[$taskid]}
mseg=${m_list[$taskid]}
actual_size=19048862
dimension=1536
keys=dstore/law/dstore_size19068709_embed1536_fp16_keys.npy
vals=dstore/law/dstore_size19068709_embed1536_fp16_vals.npy
index_name=dstore/law/knn.${actual_size}.pca${pca}.m${mseg}.index

python build_dstore.py \
    --dstore_keys ${keys} \
    --dstore_vals ${vals} \
    --dstore_size ${size} \
    --faiss_index ${index_name} \
    --actual_dstore_size ${actual_size} \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 \
    --dimension ${dimension} \
    --dstore_fp16 \
    --pca ${pca} \
    --code_size ${mseg} \

