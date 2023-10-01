#!/bin/bash

HEADER="halfcheetah runs"

ENV=halfcheetah
CONFIG_FILE=medium_v2
BATCH_SIZE=256
# how many times to run it, generally 1 for debugging, 5 for real runs
NUM_SEEDS=1
SEEDS=($(seq 1 $NUM_SEEDS))

echo -e "RUNNING: \`$HEADER\` - SEEDS: $SEEDS\n---\n"

# iterate over seeds and run each with runs_entrypoint
for SEED in "${SEEDS[@]}"
do
    ./scripts/entrypoint_runs.sh -e $ENV -c $CONFIG_FILE -b $BATCH_SIZE -s $SEED "$@"
done
