#!/bin/bash

# MAIN
ENV="${ENV:-hopper}"
ENV_CONFIG_FILE="${ENV_CONFIG_FILE:-medium_v2}"
WANDB_MODE="${WANDB_MODE:-online}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-100000}"
DOWNSTREAM_STEPS="${DOWNSTREAM_STEPS:-100000}"

# ACTION_EMBED_CLASS=ActionTokenizedSpreadEmbedding
# EVAL_OUTPUT_SEQUENTIAL=True

EMBED_CLASS="${EMBED_CLASS:-ActionTokenizedEmbedding}"
EVAL_OUTPUT_SEQUENTIAL="${EVAL_SEQUENTIAL:-False}"


# EXTRA STUFF - MOSTLY GENERIC
BATCH_SIZE=256
# CONFIG_FILE="${CONFIG_FILE:-$ENV_CONFIG_FILE}"
HEADER="$ENV-$ENV_CONFIG_FILE"
GROUP="${GROUP:-$HEADER}"
JOB_TYPE="${JOB_TYPE:-$HEADER}"
HEADER=TESTING-NO-PRETRAIN-DOWNSTREAM-$HEADER
# BELOW IS GENERIC
SEED="${SEED:-1}"

echo -e "RUNNING: \`$HEADER\`" | tr a-z A-Z
echo -e "   -> SEED: $SEED | DOWNSTREAM-ENV: $ENV | CONFIG: $ENV_CONFIG_FILE | PRETRAIN_STEPS: $PRETRAIN_STEPS | DOWNSTREAM_STEPS: $DOWNSTREAM_STEPS | WANDB_MODE: $WANDB_MODE"
echo -e "   -> ARGS: $@\n---\n"


# ./scripts/entrypoint_runs.sh -e $ENV -c $CONFIG_FILE -b $BATCH_SIZE -s $SEED "$@"

python inctxdt/run.py --cmd=downstream \
    --seed=$SEED --device=cuda \
    --config_path=conf/corl/dt/halfcheetah/medium_v2.yaml \
    --downstream.config_path=conf/corl/dt/$ENV/$ENV_CONFIG_FILE.yaml \
    --num_layers=4 \
    --num_heads=4 \
    --modal_embed.per_action_encode=False \
    --modal_embed.tokenize_action=True \
    --modal_embed.action_embed_class=$EMBED_CLASS \
    --eval_output_sequential=$EVAL_OUTPUT_SEQUENTIAL \
    --batch_size=$BATCH_SIZE \
    --update_steps="$PRETRAIN_STEPS" \
    --log.mode=$WANDB_MODE \
    --log.group=$GROUP \
    --log.job_type=$JOB_TYPE \
    $@ > output/logs/$HEADER-$SEED.log 2>&1 &
    # $@
    # --debug=OTHER_INFO
    # --downstream.update_steps=$DOWNSTREAM_STEPS > output/logs/$HEADER-$SEED.log 2>&1 &
