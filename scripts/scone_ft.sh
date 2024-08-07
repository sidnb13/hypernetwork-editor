#!/bin/bash

python3 main.py \
    --multirun \
    mode=finetune_sft \
    task=scone \
    task.domains=[alchemy] \
    task.mode=sft \
    exp_name=scone_ft \
    ++data.train_split_name=val \
    ++data.test_split_name=test \
    ++data.n_examples=128 \
    ++train.do_eval=false \
    ++train.use_ddp=false \
    ++train.log_interval=1 \
    ++train.train_batch_size=8 \
    ++train.validation_batch_size=16 \
    ++train.eval_interval=50 \
    ++train.save_interval=500 \
    ++train.do_save=true \
    ++train.lr=3e-5 \
    ++train.n_epochs=5 \
    ++train.warmup_steps=0.1 \
    ++train.scheduler=cosine \
    debug=false \
    ++wandb.enabled=true \
