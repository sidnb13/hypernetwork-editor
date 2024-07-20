#!/bin/bash

python3 main.py \
    mode=finetune_sft \
    task=scone \
    task.domains=[alchemy] \
    exp_name=scone_ft \
    ++train.use_ddp=true \
    ++train.log_interval=10 \
    ++train.train_batch_size=64 \
    ++train.validation_batch_size=64 \
    ++train.eval_interval=50 \
    ++train.save_interval=500 \
    ++train.do_save=true \
    ++train.lr=3e-5 \
    ++train.n_epochs=1 \
    ++train.warmup_steps=0.1 \
    ++train.scheduler=cosine \
    debug=false \
    ++wandb.enabled=true
