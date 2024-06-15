#!/bin/bash
n_examples=100000

python3 \
    main.py \
    mode=data_synthetic \
    ++data.n_examples=$n_examples \
    ++data.target_generation_tokens=50

python3 \
    main.py \
    mode=data_continuation \
    ++data.n_examples=$n_examples \
    ++data.target_generation_tokens=50
