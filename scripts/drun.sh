#!/bin/bash

DEVICE="${DEVICE:-0}"

cmd="docker run -d --gpus all -v /home/graham/.d4rl:/root/.d4rl -v /home/graham/code/inctxdt/data:/workspace/data -e CUDA_VISIBLE_DEVICES=$DEVICE inctxdt/base:latest python $@"

# inctxdt/run.py --cmd=downstream --device=cuda --config_path=conf/corl/dt/halfcheetah/medium_expert_v2.yaml --downstream.config_path=conf/corl/dt/antmaze/umaze_v2.yaml --num_layers=3 --num_heads=1 --modal_embed.per_action_encode=True --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=25000 --downstream.patch_actions=True --downstream.update_optim_actions=True --downstream.patch_states=True --downstream.update_optim_states=True --downstream.optim_only_patched=True --downstream.eval_every=500 --log.group=DownstreamAntmaze-TrainedHalfcheetah --log.job_type=OnlyPatch-ActionTokenizedSpreadEmbedding --log.mode=online --seed=11

echo 'command is'
eval $cmd