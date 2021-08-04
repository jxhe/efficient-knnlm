#! /bin/bash
#
# binarize.sh
# Copyright (C) 2021-02-27 Junxian <He>
#
# Distributed under terms of the MIT license.
#


TEXT=datasets/wikitext-103

python preprocess.py \
    --only-source \
    --trainpref $TEXT/wiki.valid.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.valtest.tokens \
    --destdir data-bin/wikitext-103-benchmark \
    --workers 20 \
    --srcdict data-bin/wikitext-103/dict.txt
