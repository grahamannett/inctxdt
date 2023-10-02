#!/bin/bash

# Usage:
# ./scripts/go.sh -e halfcheetah -c medium_v2 -s 0 -d 0
#   -e: Env that is a folder in conf/corl, e.g. halfcheetah, antmaze, etc
#   -c: Config file in the folder. The env and config together are the d4rl/minari dataset name (e.g. halfcheetah-medium-v2)
#   -s: Seed for training
#   -d: Device to run on
#   -b: Batch size.  A lot of the batch sizes (e.g. 4k in halfcheetah) will not work with default 24GB gpu. Avoid changing the configs so allow this to be changed here.

# Default values
SEED=0
DEVICE=0
CONFIG_BASE=conf/corl/dt
BATCH_SIZE=256
WANDB_MODE=online
DRY_RUN="${DRY_RUN:-false}"

CMD_SEP_PREFIX="==> "
CMD_SEPARATOR="--- --- ---"

# Parse options -- e and c are main/required
while getopts "e:c:b:s:d:" opt; do
  case $opt in
    e) ENV_DIR=$OPTARG ;;
    c) CONFIG_FILE=$OPTARG ;;
    b) BATCH_SIZE=$OPTARG ;;
    s) SEED=$OPTARG ;;
    d) DEVICE=$OPTARG ;;
    *) echo "Invalid option: -$OPTARG" >&2
       exit 1
       ;;
  esac
done



CONFIG_FILEPATH="$CONFIG_BASE/$ENV_DIR/$CONFIG_FILE.yaml"

if ! [[ -f $CONFIG_FILEPATH ]]; then
    echo "Config file $CONFIG_FILEPATH does not exist"
    exit 1
fi


function build_command {
  # using $cmd_prefix and $cmd_direct_output like this so we can comment out if needed for debugging
  # local JOB_NAME=$1
  # local CMD_DIRECT_OUTPUT="> output/logs/$ENV_DIR-$CONFIG_FILE-$SEED-$1.log 2>&1 &"
  local CMD_DIRECT_OUTPUT="> output/logs/$ENV_DIR-$CONFIG_FILE-$SEED-$CMD_JOB_NAME.log 2>&1 &"

  local cmd_out=""
  cmd_out="$cmd_prefix"
  # cmd_out="$cmd_out \"${@:2}\"" # skip first arg
  cmd_out="$cmd_out $@"
  cmd_out="$cmd_out $CMD_DIRECT_OUTPUT"
  echo -e $cmd_out
  # echo -e $cmd_prefix "$@" $cmd_direct_output
}


function echo_and_run {
    local header=$CMD_JOB_NAME
    local command=$1
    # note: echo with -e flag interprets \n and want # so if i copy it to run it, it's clear but commented out
    # echo -e "---"
    echo -e "\n🍉 ⬇️  === $header ===\n"
    echo -e "$command"
    echo -e "\n---"

    if [ "$DRY_RUN" = false ] ; then
      eval $command
    fi
    # eval $command
    _NUM_RUNS=$(( _NUM_RUNS + 1 ))
}

cmd_base="python \
inctxdt/run.py \
--cmd=train \
--device=cuda \
--config_path=$CONFIG_FILEPATH \
--train_seed=$SEED \
--batch_size=$BATCH_SIZE"

cmd_prefix="CUDA_VISIBLE_DEVICES=$DEVICE"
cmd_direct_output="> output/logs/$ENV_DIR-$CONFIG_FILE-$SEED.log 2>&1 &"

# not using:
secondary_loss_args="--use_secondary_loss=True --scale_state_loss=0.1 --scale_rewards_loss=0.1"

# For log args:
# We can group with job_type and group, so one of these should be the idea of multiple different runs of different types and one of these should be the idea of different runs of same type with different seeds.
# group should be all same, job type is per dt/dt+token/dt+token+spread/etc
# log args that arent shared among run are log.job_type and log.name (name can be empty)
log_args="\
--log.mode=$WANDB_MODE \
--log.group=$ENV_DIR-$CONFIG_FILE"


# HERE ARE THE COMMANDS TO RUN:
# baseline - no tokenization, no secondary loss, no sequential eval
# tokenized - tokenized actions, no secondary loss, no sequential eval
# tokenized-seperate - each action tokenized with unique encoding, no secondary loss, no sequential eval
# tokenized-spread - tokenized actions, secondary loss, sequential eval
# tokenized-spread-secondary-loss - tokenized actions, secondary loss, sequential eval, 2nd loss

# --- --- ---
# tokenized_spread_secondary_loss
# --- ---
# ---
CMD_JOB_NAME="tokenized_spread_secondary_loss"
cmd_args_tokenized_spread_secondary_loss="\
--modal_embed.per_action_encode=False \
--modal_embed.tokenize_action=True \
--modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding \
--eval_output_sequential=True \
--use_secondary_loss=True \
--scale_state_loss=0.1 \
--scale_rewards_loss=0.1 \
--log.job_type=$CMD_JOB_NAME"
cmd_tokenized_sequential_secondary_loss=$(build_command $cmd_base $log_args $cmd_args_tokenized_spread_secondary_loss)
echo_and_run "$cmd_tokenized_sequential_secondary_loss"

# --- --- ---

# --- --- ---
# tokenized_spread
# --- ---
CMD_JOB_NAME="tokenized_spread"
cmd_args_tokenized_spread="\
--modal_embed.per_action_encode=False \
--modal_embed.tokenize_action=True \
--modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding \
--eval_output_sequential=True \
--log.job_type=$CMD_JOB_NAME"
cmd_tokenized_spread=$(build_command $cmd_base $log_args $cmd_args_tokenized_spread)
echo_and_run "$cmd_tokenized_spread"
# --- --- ---

# --- --- ---
# tokenized-seperate
# --- ---
CMD_JOB_NAME="tokenized_seperate"
cmd_args_tokenized_seperate="\
--modal_embed.per_action_encode=True \
--modal_embed.tokenize_action=True \
--modal_embed.action_embed_class=ActionTokenizedEmbedding \
--log.job_type=$CMD_JOB_NAME"
cmd_tokenized_seperate=$(build_command $cmd_base $log_args $cmd_args_tokenized_seperate)
echo_and_run "$cmd_tokenized_seperate"
# --- --- ---

# --- --- ---
# tokenized
# --- ---
CMD_JOB_NAME="tokenized"
cmd_args_tokenized="\
--modal_embed.per_action_encode=False \
--modal_embed.tokenize_action=True \
--modal_embed.action_embed_class=ActionTokenizedEmbedding \
--log.job_type=$CMD_JOB_NAME"
cmd_tokenized=$(build_command $cmd_base $log_args $cmd_args_tokenized)
echo_and_run "$cmd_tokenized"
# --- --- ---

# --- --- ---
# baseline
# --- ---
CMD_JOB_NAME="baseline"
cmd_args_baseline="\
--modal_embed.action_embed_class=ActionEmbedding \
--log.job_type=$CMD_JOB_NAME"
cmd_baseline=$(build_command $cmd_base $log_args $cmd_args_baseline)
echo_and_run "$cmd_baseline"
# --- --- ---
