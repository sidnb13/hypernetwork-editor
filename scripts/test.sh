#!/bin/bash

python3 main.py \
    mode=train \
    task=wikipedia \
    exp_name=test \
    ++train.loss=kl \
    ++train.use_ddp=true \
    ++train.log_interval=5 \
    ++train.eval_interval=20 \
    ++train.save_interval=100\
    ++train.train_batch_size=16 \
    ++train.validation_batch_size=16 \
    ++train.do_save=true \
    ++train.lr=1e-6 \
    ++train.scheduler=cosine \
    ++wandb.enabled=true \
    ++wandb.run_id="vgwg1sby" \
    ++train.steps=200 \
    ++resume_ckpt="/home/sidnbaskaran/hypernetwork-editor/assets/checkpoints/test_20240512_163553/step-100"
