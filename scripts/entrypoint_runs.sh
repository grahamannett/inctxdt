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
DEVICE="${DEVICE:-0}"
CONFIG_BASE=conf/corl/dt
BATCH_SIZE=256
WANDB_MODE="${WANDB_MODE:-online}"
DRY_RUN="${DRY_RUN:-false}"



CMD_SEP_PREFIX="==> "
CMD_SEPARATOR="--- --- ---"

echo "INPUT ARGS TO entrypoint_runs.sh:" $@

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
GROUP="${GROUP:-$ENV_DIR-$CONFIG_FILE}"

if ! [[ -f $CONFIG_FILEPATH ]]; then
    echo "Config file $CONFIG_FILEPATH does not exist"
    exit 1
fi

function build_command {
  local CMD_DIRECT_OUTPUT="> output/logs/$ENV_DIR-$CONFIG_FILE-$SEED-$CMD_JOB_NAME.log 2>&1 &"
  local cmd_out="$cmd_prefix $@ $CMD_DIRECT_OUTPUT"
  echo -e $cmd_out
}

function echo_and_run {
    local header=$CMD_JOB_NAME
    local command=$1
    echo -e "\n🍉 ⬇️  === $header ===\n"
    echo -e "$command"
    echo -e "\n---"
    if [ "$DRY_RUN" = false ] ; then
      eval $command
      pids[${#pids[@]}]=$!
    fi
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

log_args="\
--log.mode=$WANDB_MODE \
--log.group=$GROUP"

declare -A job_args
declare -a pids
# old name was tokenized_spread_secondary_loss
job_args["ActionTokenizedSpreadEmbedding-SecondaryLoss"]="\
--modal_embed.per_action_encode=False \
--modal_embed.tokenize_action=True \
--eval_output_sequential=True \
--use_secondary_loss=True \
--scale_state_loss=0.1 \
--scale_rewards_loss=0.1 \
--modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding"

job_args["ActionTokenizedSpreadEmbedding-SequentialEval"]="\
--modal_embed.per_action_encode=False \
--modal_embed.tokenize_action=True \
--eval_output_sequential=True \
--modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding"

job_args["ActionTokenizedSpreadEmbedding-PerActionTokenized"]="\
--modal_embed.per_action_encode=True \
--modal_embed.tokenize_action=True \
--eval_output_sequential=True \
--modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding"

job_args["ActionTokenizedEmbedding-PerActionTokenized"]="\
--modal_embed.per_action_encode=True \
--modal_embed.tokenize_action=True \
--modal_embed.action_embed_class=ActionTokenizedEmbedding"

job_args["ActionTokenizedEmbedding"]="\
--modal_embed.per_action_encode=False \
--modal_embed.tokenize_action=True \
--modal_embed.action_embed_class=ActionTokenizedEmbedding"

job_args["Baseline"]="\
--modal_embed.tokenize_action=False \
--modal_embed.action_embed_class=ActionEmbedding"

for job_name in "${!job_args[@]}"; do
  echo job_name: $job_name
  CMD_JOB_NAME=$job_name
  cmd_args="${job_args[$job_name]} --log.job_type=$CMD_JOB_NAME"
  cmd=$(build_command $cmd_base $cmd_args $log_args)
  echo_and_run "$cmd"
done

# Wait for all child processes to finish
for pid in ${pids[*]}; do
  wait $pid
done