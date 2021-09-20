#!/bin/bash


GPU=$1
seed=927
lr=0.0005
# feature="freq,lm_ent,lm_max"
# feature="all"
feature=$2

DATE=`date +%Y%m%d`

declare -a hid_list=(256)
declare -a nl_list=(5)
# declare -a drop_list=(0.1 0.2 0.3 0.5)
declare -a activation_list=('relu')
declare -a drop_list=(0.2)
declare -a l1_list=(0.05)

feature_str=$(printf "$feature" | tr , _)


for hid in "${hid_list[@]}"; do
    for nl in "${nl_list[@]}"; do
        for activation in "${activation_list[@]}"; do
            for drop in "${drop_list[@]}"; do
                for l1_lambda in "${l1_list[@]}"; do
                        SAVE=checkpoint/wikitext_103/${DATE}/moe/mlp.l1${l1_lambda}.nh${hid}.nl${nl}.act${activation}.drop${drop}.ft${feature_str}.seed${seed}
                        rm -r ${SAVE}; mkdir -p ${SAVE}

                        CUDA_VISIBLE_DEVICES=${GPU} python knnlm_scripts/moe.py \
                            --hidden-units ${hid} \
                            --nlayers ${nl} \
                            --dropout ${drop} \
                            --seed ${seed} \
                            --output-dir ${SAVE} \
                            --activation ${activation} \
                            --feature-type ${feature} \
                            --train  /projects/junxianh/knnlmXS/datasets/wikitext-103/valid_feat \
                            --val /projects/junxianh/knnlmXS/datasets/wikitext-103/test_feat \
                            --l1 \
                            --l1-lambda ${l1_lambda} \
                            # --move-to-mem
                            # --input features.jsonl,features_test.jsonl \
                            # --debug
                            # --train datasets/wikitext-103-valtest/train_feat.jsonl \
                            # --val datasets/wikitext-103-valtest/val_feat.jsonl \
                done
            done
        done
    done
done
