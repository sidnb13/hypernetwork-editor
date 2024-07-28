#!/bin/bash

python3 main.py \
    mode=eval \
    task=scone \
    task.domains=[alchemy] \
    task.mode=editor \
    exp_name=scone_ft \
    ++data.train_split_name=val \
    ++data.test_split_name=test \
    ++data.n_examples=128 \
    ++train.do_eval=false \
    ++train.use_ddp=false \
    ++train.log_interval=1 \
    ++train.train_batch_size=8 \
    ++train.validation_batch_size=8
