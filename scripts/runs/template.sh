#!/bin/bash

ENV=
ENV_CONFIG_FILE=

# EXTRA STUFF - MOSTLY GENERIC - COPY BELOW
BATCH_SIZE=256
CONFIG_FILE="${CONFIG_FILE:-$ENV_CONFIG_FILE}"
HEADER=$ENV-$CONFIG_FILE
# BELOW IS GENERIC
SEED="${SEED:-1}"

echo -e "RUNNING: \`$HEADER\`" | tr a-z A-Z
echo -e "   -> SEED: $SEED"
echo -e "   -> ARGS PASSED IN: $@\n---\n"

./scripts/train_runs_entrypoint.sh -e $ENV -c $CONFIG_FILE -b $BATCH_SIZE -s $SEED "$@"