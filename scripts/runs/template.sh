#!/bin/bash

HEADER="template"

ENV=
CONFIG_FILE=
BATCH_SIZE=

SEEDS=(1 2 3 4)

echo -e "RUNS: $HEADER\n--- --- ---"

./scripts/go.sh -e $ENV -c $CONFIG_FILE -b $BATCH_SIZE "$@"