#! /bin/bash
#
# binarize.sh
# Copyright (C) 2021-02-27 Junxian <He>
#
# Distributed under terms of the MIT license.
#


TEXT=datasets/wikitext-103/jackknife

for i in $(seq 0 4)
do
    python preprocess.py \
        --only-source \
        --trainpref $TEXT/train.fold$i \
        --validpref $TEXT/val.fold$i \
        --destdir data-bin/wikitext-103-fold$i \
        --workers 20 \
        --srcdict data-bin/wikitext-103/dict.txt
done
