#!/bin/bash


# MAIN
ENV=antmaze
ENV_CONFIG_FILE=umaze_v2

# EXTRA STUFF - MOSTLY GENERIC
BATCH_SIZE=256
CONFIG_FILE="${CONFIG_FILE:-ENV_CONFIG_FILE}"
HEADER=$ENV-$CONFIG_FILE
# BELOW IS GENERIC
SEED="${SEED:-1}"

echo -e "RUNNING: \`$HEADER\`" | tr a-z A-Z
echo -e "   -> SEED: $SEED"
echo -e "   -> ARGS: $@\n---\n"

./scripts/entrypoint_runs.sh -e $ENV -c $CONFIG_FILE -b $BATCH_SIZE -s $SEED "$@"
