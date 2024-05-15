#!/bin/bash

python3 main.py \
    mode=train \
    task=wikipedia \
    exp_name=test \
    ++train.loss=kl \
    ++train.use_ddp=false \
    ++train.log_interval=5 \
    ++train.eval_interval=20 \
    ++train.save_interval=20\
    ++train.steps=100 \
    ++train.train_batch_size=16 \
    ++train.validation_batch_size=16 \
    ++train.do_save=true \
    ++train.lr=1e-6 \
    ++train.scheduler=cosine \
    ++wandb.enabled=false \
