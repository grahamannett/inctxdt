#!/bin/bash

DEVICE="${DEVICE:-0}"
SEED="${SEED:-10}"

GROUP="${GROUP:-DOWNSTREAM/AntmazeUmaze-TrainedHalfCheetahMediumExpert}"
# CONFIG PATH is PRETRAIN
CONFIG_PATH="${CONFIG_PATH:-conf/corl/dt/halfcheetah/medium_expert_v2.yaml}"
DOWNSTREAM_CONFIG_PATH="${DOWNSTREAM_CONFIG_PATH:-conf/corl/dt/antmaze/umaze_v2.yaml}"
UPDATE_STEPS="${UPDATE_STEPS:-50000}"

NUM_LAYERS="${NUM_LAYERS:-3}"
NUM_HEADS="${NUM_HEADS:-1}"

# source $1


# DOWNSTREAM_CORNFIT_PATH is DOWNSTREAM
# update_steps=$0

# baseline with low training steps
CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --cmd=downstream --device=cuda --config_path=$CONFIG_PATH --downstream.config_path=$DOWNSTREAM_CONFIG_PATH --num_layers=$NUM_LAYERS --num_heads=$NUM_HEADS --modal_embed.per_action_encode=False --modal_embed.tokenize_action=False --modal_embed.action_embed_class=ActionEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=$UPDATE_STEPS --downstream.patch_actions=True --downstream.update_optim_actions=True --downstream.patch_states=True --downstream.update_optim_states=True --downstream.optim_only_patched=True --log.group=$GROUP --log.job_type=UpdateEmb-ActionEmbedding --log.mode=online --seed=$SEED --train_seed=$SEED  > /dev/null 2>&1 &


# spread only emb training
CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --cmd=downstream  --device=cuda --config_path=$CONFIG_PATH --downstream.config_path=$DOWNSTREAM_CONFIG_PATH --num_layers=$NUM_LAYERS --num_heads=$NUM_HEADS --modal_embed.per_action_encode=True --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=$UPDATE_STEPS --downstream.patch_actions=True --downstream.update_optim_actions=True --downstream.patch_states=True --downstream.update_optim_states=True --downstream.optim_only_patched=True --log.group=$GROUP --log.job_type=UpdateEmb-ActionTokenizedSpreadEmbedding  --log.mode=online --seed=$SEED --train_seed=$SEED > /dev/null 2>&1 &


# ActionTokenizedEmbedding - aka not spread, only patched
CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --cmd=downstream  --device=cuda --config_path=$CONFIG_PATH --downstream.config_path=$DOWNSTREAM_CONFIG_PATH --num_layers=$NUM_LAYERS --num_heads=$NUM_HEADS --modal_embed.per_action_encode=True --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=$UPDATE_STEPS --downstream.patch_actions=True --downstream.update_optim_actions=True --downstream.patch_states=True --downstream.update_optim_states=True --downstream.optim_only_patched=True --log.group=$GROUP --log.job_type=UpdateEmb-ActionTokenizedEmbedding --log.mode=online --seed=$SEED --train_seed=$SEED > /dev/null 2>&1 &


# baseline with NO ACTION TRAINING
CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --cmd=downstream --device=cuda --config_path=$CONFIG_PATH --downstream.config_path=$DOWNSTREAM_CONFIG_PATH --num_layers=$NUM_LAYERS --num_heads=$NUM_HEADS --modal_embed.per_action_encode=False --modal_embed.tokenize_action=False --modal_embed.action_embed_class=ActionEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=$UPDATE_STEPS --downstream.patch_actions=True --downstream.update_optim_actions=False --downstream.patch_states=True --downstream.update_optim_states=True --downstream.optim_only_patched=True --log.group=$GROUP --log.job_type=UpdateState-ActionEmbedding --log.mode=online --seed=$SEED --train_seed=$SEED  > /dev/null 2>&1 &

# ActionTokenizedSpreadEmbedding with NO ACTION TRAINING
CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --cmd=downstream --device=cuda --config_path=$CONFIG_PATH --downstream.config_path=$DOWNSTREAM_CONFIG_PATH --num_layers=$NUM_LAYERS --num_heads=$NUM_HEADS --modal_embed.per_action_encode=False --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --batch_size=128 --update_steps=$UPDATE_STEPS --downstream.patch_actions=False --downstream.update_optim_actions=True --downstream.patch_states=True --downstream.update_optim_states=True --downstream.optim_only_patched=True --log.group=$GROUP --log.job_type=UpdateState-ActionTokenizedSpreadEmbedding --log.mode=online --seed=$SEED --train_seed=$SEED  > /dev/null 2>&1 &


# DEFAULT OPTIM
# ActionTokenizedEmbedding - aka not spread, only patched
CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --cmd=downstream  --device=cuda --config_path=$CONFIG_PATH --downstream.config_path=$DOWNSTREAM_CONFIG_PATH --num_layers=$NUM_LAYERS --num_heads=$NUM_HEADS --modal_embed.per_action_encode=True --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=$UPDATE_STEPS --downstream.patch_actions=True --downstream.patch_states=True --downstream.optim_use_default=True --log.group=$GROUP --log.job_type=DefaultOptim-ActionTokenizedEmbedding --log.mode=online --seed=$SEED --train_seed=$SEED > /dev/null 2>&1 &

# spread with all training
CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --cmd=downstream  --device=cuda --config_path=$CONFIG_PATH --downstream.config_path=$DOWNSTREAM_CONFIG_PATH --num_layers=$NUM_LAYERS --num_heads=$NUM_HEADS --modal_embed.per_action_encode=True --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=$UPDATE_STEPS --downstream.patch_actions=False --downstream.patch_states=True --downstream.optim_use_default=True --log.group=$GROUP --log.job_type=DefaultOptim-ActionTokenizedSpreadEmbedding --log.mode=online --seed=$SEED --train_seed=$SEED > /dev/null 2>&1 &


# ActionEmbedding - aka not spread, only patched
CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --cmd=downstream  --device=cuda --config_path=$CONFIG_PATH --downstream.config_path=$DOWNSTREAM_CONFIG_PATH --num_layers=$NUM_LAYERS --num_heads=$NUM_HEADS --modal_embed.tokenize_action=False --modal_embed.action_embed_class=ActionEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=$UPDATE_STEPS --downstream.patch_actions=True --downstream.patch_states=True --downstream.optim_use_default=True --log.group=$GROUP --log.job_type=DefaultOptim-ActionEmbedding --log.mode=online --seed=$SEED --train_seed=$SEED > /dev/null 2>&1 &