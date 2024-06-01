#!/bin/bash

python3 main.py \
    mode=train \
    task=wikipedia \
    exp_name=wikipedia-full \
    ++model.use_layerwise_embeddings=false \
    ++model.num_editing_heads=6144 \
    ++model.edit_channel_multiply_factor=16 \
    ++model.compute_position_ids=false \
    ++model.use_ghost_token=true \
    ++train.loss=kl \
    ++train.use_ddp=false \
    ++train.log_interval=100 \
    ++train.train_batch_size=16 \
    ++train.validation_batch_size=16 \
    ++train.eval_interval=500 \
    ++train.save_interval=10000 \
    ++train.do_save=true \
    ++train.lr=3e-4 \
    ++train.scheduler=cosine \
    ++wandb.enabled=true \
