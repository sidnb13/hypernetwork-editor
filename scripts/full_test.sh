#!/bin/bash

python3 main.py \
    mode=train \
    task=wikipedia \
    exp_name=wikipedia-full \
    ++train.loss=kl \
    ++train.use_ddp=true \
    ++train.log_interval=10 \
    ++train.eval_interval=500 \
    ++train.save_interval=10000 \
    ++train.do_save=true \
    ++train.lr=1e-6 \
    ++train.scheduler=cosine \
    ++wandb.enabled=true \
