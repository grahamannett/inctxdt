#!/bin/bash


ENV=walker2d
CONFIG_FILE=medium_v2
BATCH_SIZE=128

HEADER=$ENV-$CONFIG_FILE
# BELOW IS GENERIC
SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-1}"
SEEDS=($(seq $SEED_START $SEED_END))

echo -e "RUNNING: \`$HEADER\`" | tr a-z A-Z
echo -e "   -> SEEDS: ${SEEDS[@]}\n---\n"

export _NUM_RUNS=0

# iterate over seeds and run each with runs_entrypoint
for SEED in "${SEEDS[@]}"
do
    ./scripts/entrypoint_runs.sh -e $ENV -c $CONFIG_FILE -b $BATCH_SIZE -s $SEED "$@"
done