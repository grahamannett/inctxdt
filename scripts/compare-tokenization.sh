#!/bin/bash

CUDA_DEVICE=1

# action tokenized per action
log_name='DT-sep-action-tokenized-antmaze-umaze-v2'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python inctxdt/run.py --cmd=train --device=cuda --modal_embed.per_action_encode=True --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedEmbedding --config_path=conf/corl/dt/antmaze/umaze_v2.yaml --log.mode=online --log.name=$log_name  > output/logs/$log_name.log 2>&1 &


# action tokenized all actions
log_name='DT-together-action-antmaze-umaze-v2'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python inctxdt/run.py --cmd=train --device=cuda --modal_embed.per_action_encode=False --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedEmbedding --config_path=conf/corl/dt/antmaze/umaze_v2.yaml--log.mode=online --log.name=$log_name > output/logs/$log_name.log 2>&1 &

# corl baseline and my baseline should be very similar

# baseline
log_name='DT-baseline-antmaze-umaze-v2'
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python inctxdt/run.py --cmd=train --device=cuda --modal_embed.action_embed_class=ActionEmbedding --config_path=conf/corl/dt/antmaze/umaze_v2.yaml --log.mode=online --log.name=$log_name > output/logs/$log_name.log 2>&1 &

# CORL Baseline
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python baseline/corl_dt.py --config_path=conf/corl/dt/antmaze/umaze_v2.yaml --mode=online --name="CORL-DT" --checkpoints_path="output/corl" > output/logs/corl-antmaze-umaze.log 2>&1 &

