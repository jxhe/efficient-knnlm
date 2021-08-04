# Efficient Nearest Neighbor Language Models

This repository is largly based on the [knnlm](https://github.com/urvashik/knnlm) repo which is a fork of [Fairseq](https://github.com/pytorch/fairseq) repo and the exact fairseq commit that this code is based on can be found [here](https://github.com/pytorch/fairseq/tree/6a5181509aa1fa7d260985157e77211753da544b). Please use the exact commit page to determine software requirements for using this code. We encourage the readers to check the original [knnlm](https://github.com/urvashik/knnlm) repo for 


## Install Dependencies
```bash
pip install --editable .

pip install faiss
```

### A Note about Hardware

Experiments for this paper were conducted on machines that contain 100GB of RAM, NVIDIA 3090 24GB GPUs. Saving the Wikitext-103 datastore requires 400GB of disk space. The speed of saving the datastore, building the FAISS index and evaluating the nearest neighbors language model heavily depends on the amount of RAM available for each job. Some of these steps can be sped up by parallelizing, which we leave for users to do in order to best cater to their setup.

If you are working with a remote cluster, please note that we use [memmaps](https://numpy.org/doc/1.18/reference/generated/numpy.memmap.html) for saving the datastore. This allows us to keep the data on disk while accessing it by loading small chunks into memory, depending on the available RAM. This means there are a large number of disk seeks. In order to prevent slowing down your entire cluster, we suggest always reading/writing this data to/from local disks (as opposed to NFS directories), and flash storage is best for faster access.

### Prerequisite -- go through the evaluation of the kNNLM baseline and prepare the datastores

**Prepare Data**.
We share Fairseq's instructions on how to prepare the data here.

```bash
mkdir -p datasets/wikitext-103
cp examples/language_model/wikitext-103/prepare-wikitext-103.sh datasets/wikitext-103

cd datasets/wikitext-103
bash prepare-wikitext-103.sh
cd ../..

TEXT=datasets/wikitext-103
python preprocess.py \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
```

**Download the language model checkpoint pretrained on WikiText-103 (trained by facebook)**
```bash
wget https://nlp.stanford.edu/projects/knnlm/wt103_checkpoint_best.pt -P knnlm_ckpt
```



**Evaluate kNNLM**:

```bash
bash knnlm_scripts/utils_cmd/eval_knnlm.sh \
    -d wikitext-103 \
    -s valid \
    -p dstore/dstore_size103225485_embed1024_fp16 \
    -i dstore/knn.default.index \
    -n 103225485 \
```



## Efficient kNNLM
The following includes instructions to speed-up kNNLM through adaptive retrieval, datastore pruning, and dimension reduction.

### Adaptive Retrieval

We train the retrieval adaptor with validation data, thus we use 90% of the original validation data for training and the remaining for validation:

```bash
mkdir datasets/wikitext103-valid

# hold out 10% for validation which is like 6 articles
python knnlm_scripts/hold_out_train.py --input datasets/wikitext-103/wiki.valid.tokens --n 6 --output datasets/wikitext103-valid/valid

# binarize
TEXT=datasets/wikitext103-valid

python preprocess.py \
    --only-source \
    --trainpref $TEXT/valid.train \
    --validpref $TEXT/valid.heldout \
    --destdir data-bin/wikitext103-valid \
    --workers 20 \
    --srcdict data-bin/wikitext-103/dict.txt   
```

Precompute scalar features like frequency or fertility which may be used in adaptive retrieval:

```bash
# the command would save "freq_cache.pickle" and "fertility_cache.pickle" into folder [args.cache]
python knnlm_scripts/cache_freq_fertility.py --data datasets/wikitext-103/wiki.train.tokens --cache datasets/wikitext103-valid
```

Precompute all features including contextualized embeddings:

```bash
# [NOTE]: the following commands require one GPU

# The following command saves "[split]_ctxt.jsonl" (include contextualized embeddings) 
# and "[split]_others.jsonl" (include other required quantities such as 
# symbolic features, lm/knnlm scores, lm confidence, etc.)
bash knnlm_scripts/adaptive_retrieval/precompute_all_features.sh train
bash knnlm_scripts/adaptive_retrieval/precompute_all_features.sh valid
```

Train the retrieval adaptor:

```bash
# [NOTE]: the following command requires one GPU

# arguments like input paths can be changed in this script
bash knnlm_scripts/adaptive_retrieval/train_ar.sh
```

please find the saved checkpoint in `checkpoint/wikitext103-valid/[DATE]/moe/mlp.l1.0.05.ngram0.hid128.nl5.bs64.drop0.2.ftall.seed927.jobid.taskid/`

evaluate knnlm with adaptive retrieval:

```bash
bash knnlm_scripts/benchmark/benchmark_ar.sh [the retrieval adaptor .pt file]
```



### Datastore Pruning

```

```

