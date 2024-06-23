#!/bin/bash

python3 main.py \
    mode=train \
    task=counterfact \
    exp_name=counterfact_full_run \
    ++task.name_or_path=assets/data/processed_counterfact_full_data \
    ++model.cross_attn_layers=[] \
    ++model.chop_editor_at_layer=0 \
    ++model.use_layerwise_embeddings=false \
    ++model.num_editing_heads=6144 \
    ++model.edit_channel_multiply_factor=16 \
    ++model.compute_position_ids=false \
    ++model.use_ghost_token=false \
    ++train.loss=kl \
    ++train.use_ddp=false \
    ++train.log_interval=10 \
    ++train.train_batch_size=64 \
    ++train.validation_batch_size=64 \
    ++train.eval_interval=500 \
    ++train.save_interval=4000 \
    ++train.do_save=true \
    ++train.lr=3e-4 \
    ++train.scheduler=cosine \
    ++wandb.enabled=true \
    ++wandb.notes="only embedding layer"
