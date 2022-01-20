
dataset=${1}
new_data=${dataset}-valid

echo "make dir ${new_data}"
mkdir -p datasets/${new_data}

echo "holdout 10% of dataset ${dataset} for validation"
# hold out 10% for validation which is like 6 articles
python ef_knnlm/domain_adaptation/adaptive_retrieval/hold_out_train.py \
    --input datasets/${dataset}/dev.en.tokenized \
    --output datasets/${new_data}/valid

echo "binarize dataset ${new_data}"
# binarize
TEXT=datasets/${new_data}

python preprocess.py \
    --only-source \
    --trainpref $TEXT/valid.train \
    --validpref $TEXT/valid.heldout \
    --destdir data-bin/${new_data} \
    --workers 20 \
    --srcdict data-bin/${dataset}/dict.txt
