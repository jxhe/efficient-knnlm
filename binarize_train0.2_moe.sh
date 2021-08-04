#! /bin/bash
#
# binarize.sh
# Copyright (C) 2021-02-27 Junxian <He>
#
# Distributed under terms of the MIT license.
#


TEXT=datasets/wikitext-103-train0.2

python preprocess.py \
    --only-source \
    --trainpref $TEXT/heldout150.tokens \
    --validpref examples/language_model/wikitext-103/wiki.valid1500line.tokens \
    --destdir data-bin/wikitext-103-train0.2-moe \
    --workers 20 \
    --srcdict data-bin/wikitext-103/dict.txt
