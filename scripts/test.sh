#!/bin/bash

python3 main.py \
    mode=train \
    task=wikipedia \
    exp_name=test \
    ++train.loss=kl \
    ++train.use_ddp=true \
    ++train.log_interval=5 \
    ++train.eval_interval=10 \
    ++train.do_save=false \
    ++train.lr=1e-6 \
    ++train.scheduler=cosine \
    ++wandb.enabled=false \
    ++train.steps=100 \
