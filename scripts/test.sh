#!/bin/bash

python3 main.py \
    mode=train \
    task=wikipedia \
    exp_name=test \
    ++model.edit_channel_multiply_factor=1 \
    ++model.edit_dampening_factor=1.0 \
    ++train.loss=kl \
    ++data.n_examples=1 \
    ++train.use_ddp=false \
    ++train.log_interval=5 \
    ++train.eval_interval=100 \
    ++train.do_save=true \
    ++train.steps=3000 \
    ++train.train_batch_size=1 \
    ++train.validation_batch_size=1 \
    ++train.lr=1e-4 \
    ++train.scheduler=constant \
    ++wandb.enabled=true \
