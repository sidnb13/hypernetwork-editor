#!/bin/bash

python3 main.py \
    mode=train \
    task=wikipedia \
    exp_name=wikipedia-full \
    ++model.use_layerwise_embeddings=false \
    ++model.num_editing_heads=768 \
    ++model.compute_position_ids=false \
    ++train.loss=kl \
    ++train.use_ddp=true \
    ++train.log_interval=10 \
    ++train.train_batch_size=32 \
    ++train.validation_batch_size=32 \
    ++train.eval_interval=500 \
    ++train.save_interval=10000 \
    ++train.do_save=true \
    ++train.lr=1e-4 \
    ++train.scheduler=cosine \
    ++wandb.enabled=true \
