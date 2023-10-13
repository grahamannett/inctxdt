#!/bin/bash

DEVICE="${DEVICE:-0}"

cmd="docker run -d --gpus all -v /home/graham/.d4rl:/root/.d4rl -v /home/graham/code/inctxdt/data:/workspace/data -e CUDA_VISIBLE_DEVICES=$DEVICE inctxdt/base:latest python $@"

echo running cmd $cmd
eval $cmd

# docker run -d --gpus all -v /home/graham/.d4rl:/root/.d4rl -v /home/graham/code/inctxdt/data:/workspace/data -e CUDA_VISIBLE_DEVICES=1 -e WANDB_DOCKER="inctxdt/base" -e WANDB_API_KEY="0e4ff52e96c334d262eda801fab3fc233cfb7c50" inctxdt/base:latest python inctxdt/run.py --cmd=train --config_path=conf/corl/dt/walker2d/medium_expert_v2.yaml --train_seed=20 --seed=20 --batch_size=256 --log.group=TRAIN/walker2d-medium_expert_v2 --modal_embed.per_action_encode=True --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --eval_output_sequential=True --log.job_type=ActionTokenizedSpreadEmbedding --log.mode=online --update_steps=40000

# wandb job create --project "inctxdt" -e "graham" --name "job-test" code "inctxdt/run.py"
