# Efficient Nearest Neighbor Language Models

This is implementation of the [paper](https://arxiv.org/abs/2109.04212):

```
Efficient Nearest Neighbor Language Models
Junxian He, Graham Neubig, Taylor Berg-Kirkpatrick
EMNLP 2021
```

This repo implements several techniques to speed up the evaluation of non-parametric, nearest neighbor language models. Specifically, we improve the efficiency along three axes: adaptive retrieval, datastore prunning, and dimension reduction. 



## Install Dependencies

This repository is largly based on the [knnlm](https://github.com/urvashik/knnlm) repo which is a fork of [Fairseq](https://github.com/pytorch/fairseq) (commit [da544b](https://github.com/pytorch/fairseq/tree/6a5181509aa1fa7d260985157e77211753da544b)). Please use the exact commit page to determine software requirements for using this code. 

```bash
git clone git@github.com:jxhe/efficient-knnlm.git

cd efficient-knnlm
pip install --editable .
pip install faiss
```

##### Hardware

Experiments for this paper were conducted on machines that contain 100GB of RAM, NVIDIA 3090 24GB GPUs. Saving the Wikitext-103 datastore requires 200GB of disk space.

### Running Efficient kNNLM

Note: readers can refer to the original [knnlm](https://github.com/urvashik/knnlm) repo for running the vanilla kNNLM.

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

**Compute the datastore**

```bash
python eval_lm.py data-bin/wikitext-103 \
    --path knnlm_ckpt/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap dstore/dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 103225485 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16 --dstore-fp1
```

**build faiss index (dimension reduction is performed here)** 

```bash
# the script applies PCA of dimension 512 by default 
# the PCA hyperparameter can be tuned in this script
bash ef_knnlm/build_faiss.sh
```



**Evaluate kNNLM**:

```bash
bash ef_knnlm/utils_cmd/eval_knnlm.sh \
    -d wikitext-103 \
    -s valid \
    -p dstore/dstore_size103225485_embed1024_fp16 \
    -i dstore/knn.default.index \
    -n 103225485 \
```



TODO
