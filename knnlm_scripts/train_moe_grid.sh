#!/bin/bash


GPU=$1
seed=927
lr=0.0005
feature="all"

DATE=`date +%Y%m%d`

declare -a hid_list=(128)
declare -a nl_list=(5)
# declare -a drop_list=(0.1 0.2 0.3 0.5)
declare -a activation_list=('relu')
declare -a drop_list=(0.2)


for hid in "${hid_list[@]}"; do
    for nl in "${nl_list[@]}"; do
        for activation in "${activation_list[@]}"; do
            for drop in "${drop_list[@]}"; do
                SAVE=checkpoint/wikitext_103/${DATE}/moe/mlp.nh${hid}.nl${nl}.act${activation}.drop${drop}.ft${feature}.seed${seed}
                rm -r ${SAVE}; mkdir -p ${SAVE}

                CUDA_VISIBLE_DEVICES=${GPU} python -m pdb knnlm_scripts/moe.py \
                    --hidden-units ${hid} \
                    --nlayers ${nl} \
                    --dropout ${drop} \
                    --seed ${seed} \
                    --output-dir ${SAVE} \
                    --activation ${activation} \
                    --feature-type ${feature} \
                    --input features.jsonl,features_test.jsonl \
                    # --debug
                    # --train datasets/wikitext-103-valtest/train_feat.jsonl \
                    # --val datasets/wikitext-103-valtest/val_feat.jsonl \
            done
        done
    done
done
