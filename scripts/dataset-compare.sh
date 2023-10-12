#!/bin/bash

datasets=("halfcheetah-random-v2"
    "halfcheetah-medium-v2"
    "halfcheetah-expert-v2"
    "halfcheetah-medium-replay-v2"
    "halfcheetah-full-replay-v2"
    "halfcheetah-medium-expert-v2")

i=1
for dataset in "${datasets[@]}"; do
    cuda_device=$((i % 2))
    echo "Running dataset: $dataset with CUDA device: $cuda_device"

    CUDA_VISIBLE_DEVICES=$cuda_device python run.py --cmd=train --device=cuda \
        --batch_size=64 \
        --epochs=10 \
        --seq_len=30 \
        --reward_scale=0.001 \
        --eval_episodes=10 \
        --num_workers=8 \
        --dataset_type=d4rl_across \
        --dataset_name=$dataset \
        --exp_name=${dataset}-across \
        --log.mode=online >output/logs/${dataset}.txt 2>&1 &
    # --log.mode=online >output/logs/${dataset}_$(date '+%m-%d-%H-%M-%S').txt 2>&1 &

    i=$((i + 1))
done


# compare the Pos/AltPos/MAltPos
python inctxdt/run.py --cmd=train --device=cuda --config_path=conf/corl/dt/walker2d/medium_expert_v2.yaml --train_seed=10 --seed=10 --batch_size=256 --log.group=walker2d-medium_expert_v2 --modal_embed.per_action_encode=False --modal_embed.tokenize_action=True --modal_embed.action_embed_class=PosActionTokenizedSpreadEmbedding --log.job_type=PosActionTokenizedSpreadEmbedding --log.mode=online > /dev/null 2>&1 &


python inctxdt/run.py --cmd=train --device=cuda --config_path=conf/corl/dt/walker2d/medium_expert_v2.yaml --train_seed=11 --seed=11 --batch_size=256 --log.group=walker2d-medium_expert_v2 --modal_embed.per_action_encode=False --modal_embed.tokenize_action=True --modal_embed.action_embed_class=AltPosActionTokenizedSpreadEmbedding --log.job_type=AltPosActionTokenizedSpreadEmbedding --log.mode=online > /dev/null 2>&1 &

python inctxdt/run.py --cmd=train --device=cuda --config_path=conf/corl/dt/walker2d/medium_expert_v2.yaml --train_seed=10 --seed=10 --batch_size=256 --log.group=walker2d-medium_expert_v2 --modal_embed.per_action_encode=False --modal_embed.tokenize_action=True --modal_embed.action_embed_class=MAltPosActionTokenizedSpreadEmbedding --log.job_type=MAltPosActionTokenizedSpreadEmbedding --log.mode=online > /dev/null 2>&1 &