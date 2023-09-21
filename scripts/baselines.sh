#!/bin/bash

datasets=(
    "halfcheetah-medium-v2"
    "halfcheetah-medium-expert-v2"
    "hopper-medium-v2"
    "walker2d-medium-v2"
    )

i=1
for dataset in "${datasets[@]}"; do
    cuda_device=$((i % 2))
    echo "Running dataset: $dataset with CUDA device: $cuda_device"


    CUDA_VISIBLE_DEVICES=$cuda_device python run.py --cmd=baseline --device=cuda--num_layers=4 --num_heads=4 --embedding_dim=128 --batch_size=128 --epochs=30 --seq_len=20 --num_workers=10 --eval_episodes=10 --eval_before_train=True --save_model=True --exp_name=baseline-$dataset --dataset_name=$dataset --log.name=baseline-$dataset --log.mode=online > output/logs/baseline-$dataset.log 2>&1 &

    # also run the experiment with tokenized actions
    CUDA_VISIBLE_DEVICES=$cuda_device python run.py --cmd=train --device=cuda --eval_output_sequential=False --modal_embed.action_embed_class=ActionTokenizedEmbedding --modal_embed.tokenize_action=True --num_layers=4 --num_heads=4 --embedding_dim=128 --batch_size=128 --epochs=30 --seq_len=20 --secondary_loss_scale=0.1 --num_workers=10 --eval_episodes=10 --eval_before_train=True --save_model=True --exp_name=tokenized-$dataset --dataset_name=$dataset --log.name=tokenized-$dataset --log.mode=online > output/logs/tokenized-$dataset.log 2>&1 &

    # might want date in log output
    # >output/logs/${dataset}_$(date '+%m-%d-%H-%M-%S').txt 2>&1 &

    i=$((i + 1))
done
