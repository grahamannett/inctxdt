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
BATCH_SIZE=512


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

cmd_prefix="CUDA_VISIBLE_DEVICES=$DEVICE"
cmd_direct_output="> output/logs/$ENV_DIR-$CONFIG_FILE-$SEED.log 2>&1 &"

cmd="python \
inctxdt/fake.py \
--cmd=train \
--device=cuda \
--config_path=$CONFIG_FILEPATH \
--train_seed=$SEED \
--batch_size=$BATCH_SIZE"

# For log args:
# We can group with job_type and group, so one of these should be the idea of multiple different runs of different types and one of these should be the idea of different runs of same type with different seeds.
log_args="\
--log.mode=online \
--log.name=$ENV_DIR-$CONFIG_FILE \
--log.group=$ENV_DIR-$CONFIG_FILE \
--log.job_type=$ENV_DIR-$CONFIG_FILE"

modal_embed_args="\
--modal_embed.per_action_encode=False \
--modal_embed.tokenize_action=True \
--modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding \
--eval_output_sequential=True"

secondary_loss_args="--use_secondary_loss=True --scale_state_loss=0.1 --scale_rewards_loss=0.1"

cmd_base="$cmd_prefix $cmd $cmd_direct_output"
cmd_modal="$cmd_prefix $cmd $modal_embed_args $cmd_direct_output"
echo -e "Running command:\n$cmd_modal"

# HERE ARE THE COMMANDS TO RUN:

# baseline - no tokenization, no secondary loss, no sequential eval

# with spread
# "Running[TokenSpreadSequentialSecondaryLoss]"