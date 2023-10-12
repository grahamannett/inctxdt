#!/bin/bash
DEFAULT_CUDA_DEVICE=0
DEFAULT_TRAIN_SEED=10

# ARGS TO COME IN
TRAIN_SEED="${1:-$DEFAULT_TRAIN_SEED}"
CUDA_DEVICE="${2:-$DEFAULT_CUDA_DEVICE}"

#
batch_size=128
CONFIG_BASE=conf/corl/dt
CONFIG_DIR=halfcheetah
ENV=medium_v2

GROUP=$CONFIG_DIR-$ENV # e.g. antmaze-medium_diverse_v2
CONFIG_PATH=$CONFIG_BASE/$CONFIG_DIR/$ENV.yaml

function echo_and_run() {
  local header=$1
  local command=$2
  # note: echo with -e flag interprets \n and want # so if i copy it to run it, it's clear but commented out
  echo -e "$header:\n=>\t#$command\n---"
  eval $command
}

function run_experiment() {
  local log_name=$1
  local per_action_encode=$2
  local tokenize_action=$3
  local action_embed_class=$4
  local eval_output_sequential=$5
  local use_secondary_loss=$6
  local scale_state_loss=$7
  local scale_rewards_loss=$8

  command="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
           python inctxdt/run.py \
           --cmd=train \
           --device=cuda \
           --modal_embed.per_action_encode=$per_action_encode \
           --modal_embed.tokenize_action=$tokenize_action \
           --modal_embed.action_embed_class=$action_embed_class \
           --eval_output_sequential=$eval_output_sequential \
           --use_secondary_loss=$use_secondary_loss \
           --scale_state_loss=$scale_state_loss \
           --scale_rewards_loss=$scale_rewards_loss \
           --config_path=$CONFIG_PATH \
           --log.mode=online \
           --log.name=$log_name \
           --log.group=$GROUP \
           --train_seed=$TRAIN_SEED \
           --log.job_type=$log_name \
           --batch_size=$batch_size \
           > output/logs/$log_name-$TRAIN_SEED.log 2>&1 &"
  echo_and_run "Running[$log_name]" "$command"
}

run_experiment "DT-tokenized-spread-seq-2nd-loss-$GROUP" False True "ActionTokenizedSpreadEmbedding" True True 0.1 0.1
run_experiment "DT-tokenized-spread-sequential-$GROUP" False True "ActionTokenizedSpreadEmbedding" True False None None
run_experiment "DT-seperate-tokenized-$GROUP" True True "ActionTokenizedEmbedding" False False None None
run_experiment "DT-tokenized-$GROUP" False True "ActionTokenizedEmbedding" False False None None
run_experiment "DT-baseline-$GROUP" False False "ActionEmbedding" False False None None
