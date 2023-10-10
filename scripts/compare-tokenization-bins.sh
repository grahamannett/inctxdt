#!/bin/bash

DEVICE=0
SEED=11

# CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --config_path=conf/corl/dt/antmaze/umaze_v2.yaml --cmd=train --device=cuda --num_layers=3 --num_heads=1 --update_steps=40000 --batch_size=128 --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --modal_embed.tokenize_action=True --modal_embed.per_action_encode=True --modal_embed.num_bins=100 --log.group=Antmaze-NBin-Compare --log.mode=online --seed=$SEED --log.job_type=N100ActionTokenizedSpreadEmbedding > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --config_path=conf/corl/dt/antmaze/umaze_v2.yaml --cmd=train --device=cuda --num_layers=3 --num_heads=1 --update_steps=40000 --batch_size=128 --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --modal_embed.tokenize_action=True --modal_embed.per_action_encode=True --modal_embed.num_bins=500 --log.group=Antmaze-NBin-Compare --log.mode=online --seed=$SEED --log.job_type=N500ActionTokenizedSpreadEmbedding > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --config_path=conf/corl/dt/antmaze/umaze_v2.yaml --cmd=train --device=cuda --num_layers=3 --num_heads=1 --update_steps=40000 --batch_size=128 --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --modal_embed.tokenize_action=True --modal_embed.per_action_encode=True --modal_embed.num_bins=1000 --log.group=Antmaze-NBin-Compare --log.mode=online --seed=$SEED --log.job_type=N1000ActionTokenizedSpreadEmbedding > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --config_path=conf/corl/dt/antmaze/umaze_v2.yaml --cmd=train --device=cuda --num_layers=3 --num_heads=1 --update_steps=40000 --batch_size=128 --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --modal_embed.tokenize_action=True --modal_embed.per_action_encode=True --modal_embed.num_bins=2000 --log.group=Antmaze-NBin-Compare --log.mode=online --seed=$SEED --log.job_type=N2000ActionTokenizedSpreadEmbedding > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --config_path=conf/corl/dt/antmaze/umaze_v2.yaml --cmd=train --device=cuda --num_layers=3 --num_heads=1 --update_steps=40000 --batch_size=128 --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --modal_embed.tokenize_action=True --modal_embed.per_action_encode=True --modal_embed.num_bins=3000 --log.group=Antmaze-NBin-Compare --log.mode=online --seed=$SEED --log.job_type=N3000ActionTokenizedSpreadEmbedding > /dev/null 2>&1 &

# # baseline

# CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --config_path=conf/corl/dt/antmaze/umaze_v2.yaml --cmd=train --device=cuda --num_layers=3 --num_heads=1 --update_steps=40000 --batch_size=128 --modal_embed.action_embed_class=ActionEmbedding --modal_embed.tokenize_action=False --log.group=Antmaze-NBin-Compare --log.mode=online --seed=$SEED --log.job_type=ActionEmbedding > /dev/null 2>&1 &

# -----

CONFIG_PATH=conf/corl/dt/hopper/medium_expert_v2.yaml
GROUP=Bin-Compare-Hopper-MediumExpert

CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --config_path=$CONFIG_PATH --cmd=train --device=cuda --num_layers=3 --num_heads=1 --update_steps=40000 --eval_every=500 --batch_size=128 --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --modal_embed.tokenize_action=True --modal_embed.per_action_encode=True --modal_embed.num_bins=100 --log.group=$GROUP --log.mode=online --seed=$SEED --log.job_type=N100ActionTokenizedSpreadEmbedding > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --config_path=$CONFIG_PATH --cmd=train --device=cuda --num_layers=3 --num_heads=1 --update_steps=40000 --eval_every=500 --batch_size=128 --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --modal_embed.tokenize_action=True --modal_embed.per_action_encode=True --modal_embed.num_bins=500 --log.group=$GROUP --log.mode=online --seed=$SEED --log.job_type=N500ActionTokenizedSpreadEmbedding > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --config_path=$CONFIG_PATH --cmd=train --device=cuda --num_layers=3 --num_heads=1 --update_steps=40000 --eval_every=500 --batch_size=128 --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --modal_embed.tokenize_action=True --modal_embed.per_action_encode=True --modal_embed.num_bins=1000 --log.group=$GROUP --log.mode=online --seed=$SEED --log.job_type=N1000ActionTokenizedSpreadEmbedding > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --config_path=$CONFIG_PATH --cmd=train --device=cuda --num_layers=3 --num_heads=1 --update_steps=40000 --eval_every=500 --batch_size=128 --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --modal_embed.tokenize_action=True --modal_embed.per_action_encode=True --modal_embed.num_bins=2000 --log.group=$GROUP --log.mode=online --seed=$SEED --log.job_type=N2000ActionTokenizedSpreadEmbedding > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --config_path=$CONFIG_PATH --cmd=train --device=cuda --num_layers=3 --num_heads=1 --update_steps=40000 --eval_every=500 --batch_size=128 --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --modal_embed.tokenize_action=True --modal_embed.per_action_encode=True --modal_embed.num_bins=3000 --log.group=$GROUP --log.mode=online --seed=$SEED --log.job_type=N3000ActionTokenizedSpreadEmbedding > /dev/null 2>&1 &

# baseline
CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --config_path=$CONFIG_PATH --cmd=train --device=cuda --num_layers=3 --num_heads=1 --update_steps=40000 --eval_every=500 --batch_size=128 --modal_embed.action_embed_class=ActionEmbedding --modal_embed.tokenize_action=False --log.group=$GROUP --log.mode=online --seed=$SEED --log.job_type=ActionEmbedding > /dev/null 2>&1 &

# -----

