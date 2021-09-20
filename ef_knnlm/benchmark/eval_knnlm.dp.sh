#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH --cpus-per-task=16
#SBATCH -t 0
##SBATCH --exclude=compute-0-31,compute-0-19,compute-0-15
##SBATCH —nodelist=compute-0-31,compute-0-30

jobid=${SLURM_ARRAY_JOB_ID}
taskid=${SLURM_ARRAY_TASK_ID}

taski=4

declare -a file_list=("/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/dstore_merge1_size85888581_embed1024" \
                    "/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/dstore_merge5_size68683408_embed1024" \
                    "/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/dstore_merge9_size61900130_embed1024" \
                    "/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/dstore_merge15_size56119915_embed1024" \
                    "/projects/tir3/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/dstore_merge30_size48639879_embed1024" \
                    "/projects/tir3/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/dstore_merge60_size41737766_embed1024" \
                    "dstore/filter_compress/dstore_scorefilter_size20645097_embed1024" \
                    "dstore/filter_compress/dstore_scorefilter_size41290194_embed1024" \
                    "dstore/filter_compress/dstore_scorefilter_size61935291_embed1024" \
                    "dstore/filter_compress/dstore_scorefilter_size82580388_embed1024" \
                    "dstore/random_sample/sample_dstore_size20645097_dim1024" \
                    "dstore/random_sample/sample_dstore_size41290194_dim1024" \
                    "dstore/random_sample/sample_dstore_size61935291_dim1024" \
                    "dstore/random_sample/sample_dstore_size82580388_dim1024" \
                    "/projects/tir5/users/junxianh/knnlmXS/dstore/kmeans/new_no_sampling/dstore_kmeans_size20700000_embed1024_fp16" \
    )

declare -a index_list=("/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/knn.merge1.index" \
                        "/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/knn.merge5.index" \
                        "/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/knn.merge9.index" \
                        "/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/knn.merge15.index" \
                        "/projects/tir3/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/knn.merge30.index" \
                        "/projects/tir3/users/junxianh/knnlmXS/dstore/wikitext-103/merge_compress/knn.merge60.index" \
                        "/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/filter_compress/knn.20645097.index" \
                        "/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/filter_compress/knn.41290194.index" \
                        "/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/filter_compress/knn.61935291.index" \
                        "/projects/tir5/users/junxianh/knnlmXS/dstore/wikitext-103/filter_compress/knn.82580388.index" \
                        "dstore/random_sample/knn.20645097.index" \
                        "dstore/random_sample/knn.41290194.index" \
                        "dstore/random_sample/knn.61935291.index" \
                        "dstore/random_sample/knn.82580388.index" \
                        "/projects/tir5/users/junxianh/knnlmXS/dstore/kmeans/new_no_sampling/knn.kmeans20342283.index" \
                        )


declare -a weight_list=("True" "True" "True" "True" "True" "True" "False" "False" "False" "False" "False" "False" "False" "False" "True")
declare -a size_list=(85888581 68683408 61900130 56119915 48639879 41737766 20645097 41290194 61935291 82580388 20645097 41290194 61935291 82580388 20700000)

dataset=wikitext-103
ckpt="knnlm_ckpt/wt103_checkpoint_best.pt"
split="valid"
k=1024
temp=1


for i in $(seq 14 14)
do
    dstore_size=${size_list[$i]}
    dstore_file=${file_list[$i]}
    index=${index_list[$i]}
    dstore_weight=${weight_list[$i]}

    echo "evaluate knnlm with ${dstore_file}"

    bash knnlm_scripts/utils_cmd/eval_knnlm.sh -n ${dstore_size} -p ${dstore_file} -i ${index} -d ${dataset} -c ${ckpt} -e ${temp} -k ${k} -s ${split} -w ${dstore_weight}
done
# done

