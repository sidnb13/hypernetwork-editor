#!/bin/bash

python3 main.py \
    mode=finetune_sft \
    task=scone \
    task.domains=[alchemy] \
    exp_name=scone_ft \
    ++train.use_ddp=false \
    ++train.log_interval=10 \
    ++train.train_batch_size=16 \
    ++train.validation_batch_size=16 \
    ++train.eval_interval=500 \
    ++train.save_interval=4000 \
    ++train.do_save=true \
    ++train.lr=3e-4 \
    ++train.scheduler=cosine \
    debug=true \
    ++wandb.enabled=true \
