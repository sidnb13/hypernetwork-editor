#!/bin/bash

python3 main.py \
    mode=train \
    task=wikipedia \
    exp_name=wikipedia-full \
    ++model.use_layerwise_embeddings=false \
    ++model.num_editing_heads=6144 \
    ++model.edit_channel_multiply_factor=16 \
    ++model.compute_position_ids=false \
    ++model.use_ghost_token=false \
    ++model.chop_editor_at_layer=12 \
    ++train.loss=kl \
    ++train.use_ddp=true \
    ++train.log_interval=10 \
    ++train.train_batch_size=64 \
    ++train.validation_batch_size=64 \
    ++train.eval_interval=500 \
    ++train.save_interval=10000 \
    ++train.do_save=true \
    ++train.n_epochs=1 \
    ++train.lr=3e-4 \
    ++train.scheduler=cosine \
    ++wandb.enabled=true \
    ++wandb.group=wikipedia-pretrain-hypertuning \
    ckpt_dir=assets/checkpoints/wikipedia-pretrain-hypertuning
