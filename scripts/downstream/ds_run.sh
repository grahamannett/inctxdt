#!/bin/bash

# MAIN
ENV=hopper
ENV_CONFIG_FILE=medium_v2
MODE=disabled
TRAIN_STEPS=10
DOWNSTREAM_TRAIN_STEPS=100

# EXTRA STUFF - MOSTLY GENERIC
BATCH_SIZE=256
CONFIG_FILE="${CONFIG_FILE:-ENV_CONFIG_FILE}"
HEADER="DOWNSTREAM-$ENV-$CONFIG_FILE"
# BELOW IS GENERIC
SEED="${SEED:-1}"

echo -e "RUNNING: \`$HEADER\`" | tr a-z A-Z
echo -e "   -> SEED: $SEED"
echo -e "   -> ARGS: $@\n---\n"

# ./scripts/entrypoint_runs.sh -e $ENV -c $CONFIG_FILE -b $BATCH_SIZE -s $SEED "$@"

python inctxdt/run.py --cmd=downstream \
    --seed=$SEED --device=cuda \
    --config_path=conf/corl/dt/halfcheetah/medium_v2.yaml \
    --modal_embed.per_action_encode=False \
    --modal_embed.tokenize_action=True \
    --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding \
    --eval_output_sequential=True \
    --batch_size=128 \
    --update_steps=$TRAIN_STEPS \
    --log.mode=$MODE \
    --log.group=$HEADER \
    --log.job_type=TESTING-$HEADER \
    --downstream.config_path="conf/corl/dt/$ENV/$ENV_CONFIG_FILE.yaml" \
    --downstream.update_steps=$DOWNSTREAM_TRAIN_STEPS

    # downstream_config_path: str = "conf/corl/dt/hopper/medium_v2.yaml"