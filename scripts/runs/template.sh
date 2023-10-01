#!/bin/bash

HEADER="template"

ENV=
CONFIG_FILE=
BATCH_SIZE=

# BELOW IS GENERIC
NUM_SEEDS=1
SEEDS=($(seq 1 $NUM_SEEDS))

echo -e "RUNNING: \`$HEADER\` - SEEDS: $SEEDS\n---\n"

# iterate over seeds and run each with runs_entrypoint
for SEED in "${SEEDS[@]}"
do
    ./scripts/runs_entrypoint.sh -e $ENV -c $CONFIG_FILE -b $BATCH_SIZE -s $SEED "$@"
done