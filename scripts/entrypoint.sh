#!/bin/bash

accelerate launch run.py \
    --cmd=train \
    --device=cuda \
    --batch_size=256 \
    --epochs=50 \
    --seq_len=30 \
    --reward_scale=1.0 \
    --eval_episodes=10 \
    --num_workers=10 \
    --dataset_name=halfcheetah-medium-expert-v2 \
    --dataset_type=d4rl_across
