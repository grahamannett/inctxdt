#!/bin/bash

HEADER="halfcheetah runs"

ENV=halfcheetah
CONFIG_FILE=medium_v2
BATCH_SIZE=256

SEEDS=(1 2 3 4)

echo -e "RUNS: $HEADER\n--- --- ---"

./scripts/go.sh -e $ENV -c $CONFIG_FILE -b $BATCH_SIZE "$@"