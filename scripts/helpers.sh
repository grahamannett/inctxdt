#!/bin/bash

function run_info() {
    echo -e "RUNNING: \`$1\`" | tr a-z A-Z
    echo -e "   -> SEEDS: ${SEEDS[@]}\n---\n"
}

export _NUM_RUNS=0

# iterate over seeds and run each with runs_entrypoint
for SEED in "${SEEDS[@]}"
do
    ./scripts/entrypoint_runs.sh -e $ENV -c $CONFIG_FILE -b $BATCH_SIZE -s $SEED "$@"
done