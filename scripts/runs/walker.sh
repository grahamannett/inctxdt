#!/bin/bash

# MAIN
ENV=walker2d
ENV_CONFIG_FILE=medium_v2

# EXTRA STUFF - MOSTLY GENERIC
BATCH_SIZE=256
CONFIG_FILE="${CONFIG_FILE:-ENV_CONFIG_FILE}"
HEADER=$ENV-$CONFIG_FILE
# BELOW IS GENERIC
SEED="${SEED:-1}"

echo -e "RUNNING: \`$HEADER\`" | tr a-z A-Z
echo -e "   -> SEED: $SEED"
echo -e "   -> ARGS: $@\n---\n"

./scripts/train_runs_entrypoint.sh -e $ENV -c $CONFIG_FILE -b $BATCH_SIZE -s $SEED "$@"
