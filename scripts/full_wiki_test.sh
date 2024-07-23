#!/bin/bash

python3 main.py \
    --multirun \
    mode=train_editor \
    task=wikipedia \
    exp_name=wikipedia-full \
    debug=true \
    ++model.use_layerwise_embeddings=false \
    ++model.num_editing_heads=6144 \
    ++model.edit_channel_multiply_factor=16 \
    ++model.compute_position_ids=false \
    ++model.use_ghost_token=false \
    ++model.chop_editor_at_layer=12 \
    ++train.loss=ce \
    ++train.use_ddp=false \
    ++train.log_interval=1 \
    ++train.train_batch_size=4 \
    ++train.validation_batch_size=4 \
    ++train.eval_interval=1 \
    ++train.save_interval=8 \
    ++train.do_save=true \
    ++train.n_epochs=1 \
    ++data.n_examples=32 \
    ++train.lr=3e-4 \
    ++train.scheduler=cosine \
    ++wandb.enabled=false \
    ++wandb.group=test
