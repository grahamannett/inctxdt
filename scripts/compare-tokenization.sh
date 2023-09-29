#!/bin/bash
DEFAULT_CUDA_DEVICE=0
DEFAULT_TRAIN_SEED=10

# ARGS TO COME IN
TRAIN_SEED="${1:-$DEFAULT_TRAIN_SEED}"
CUDA_DEVICE="${2:-$DEFAULT_CUDA_DEVICE}"

#
batch_size=128
CONFIG_BASE=conf/corl/dt
# CONFIG_DIR=antmaze
# ENV=umaze_v2

CONFIG_DIR=halfcheetah
ENV=medium_v2


GROUP=$CONFIG_DIR-$ENV # e.g. antmaze-medium_diverse_v2
CONFIG_PATH=$CONFIG_BASE/$CONFIG_DIR/$ENV.yaml


function echo_and_run() {
  local command=$1
  local header=$2
  echo -e "$header:\n=>$command"
  eval $command
}

# Action tokenized all actions - spread out
log_name=DT-tokenized-spread-$GROUP
command="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python inctxdt/run.py --cmd=train --device=cuda --modal_embed.per_action_encode=False --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --config_path=$CONFIG_PATH --log.mode=online --log.name=$log_name --log.group=$GROUP --train_seed=$TRAIN_SEED --log.job_type=$log_name --batch_size=$batch_size > output/logs/$log_name-$TRAIN_SEED.log 2>&1 &"
echo -e "Running[TokenSpread]:\n=>$command\n"
eval $command


# Action tokenized per action
log_name=DT-seperate-tokenized-$GROUP
command="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python inctxdt/run.py --cmd=train --device=cuda --modal_embed.per_action_encode=True --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedEmbedding --config_path=$CONFIG_PATH --log.mode=online --log.name=$log_name --log.group=$GROUP --train_seed=$TRAIN_SEED --log.job_type=$log_name --batch_size=$batch_size > output/logs/$log_name-$TRAIN_SEED.log 2>&1 &"
echo -e "Running[SepToken]:\n=>$command\n"
eval $command

# Action tokenized all actions
log_name=DT-tokenized-$GROUP
command="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python inctxdt/run.py --cmd=train --device=cuda --modal_embed.per_action_encode=False --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedEmbedding --config_path=$CONFIG_PATH --log.mode=online --log.name=$log_name --log.group=$GROUP --train_seed=$TRAIN_SEED --log.job_type=$log_name --batch_size=$batch_size > output/logs/$log_name-$TRAIN_SEED.log 2>&1 &"
echo -e "Running[Token]:\n=>$command\n"
eval $command


# Baseline
log_name=DT-baseline-$GROUP
command="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python inctxdt/run.py --cmd=train --device=cuda --modal_embed.action_embed_class=ActionEmbedding --config_path=$CONFIG_PATH --log.mode=online --log.name=$log_name --log.job_type=$log_name --log.group=$GROUP --train_seed=$TRAIN_SEED --batch_size=$batch_size > output/logs/baseline-$GROUP-$TRAIN_SEED.log 2>&1 &"
echo -e "Running[Baseline]:\n=>$command\n"
eval $command

#
# CORL Baseline
# command="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python baseline/corl_dt.py  --mode=online --name=CORL-DT-$GROUP --checkpoints_path="output/corl" --config_path=$CONFIG_PATH --group=$GROUP --job_type=corl_baseline --train_seed=$TRAIN_SEED > output/logs/corl-$GROUP-$TRAIN_SEED.log 2>&1 &"
# echo -e "Running:\n=>$command"
# eval $command

