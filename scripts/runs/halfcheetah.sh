#!/bin/bash
# to run with docker:
# docker run -d --gpus all -v /home/graham/.d4rl:/root/.d4rl -v /home/graham/code/inctxdt/data:/workspace/data -e GROUP=Train-halfcheetah-medium-v2 -e DEVICE=1 -e SEED=50 inctxdt/base:latest scripts/runs/halfcheetah.sh

ENV=halfcheetah
ENV_CONFIG_FILE=medium_v2

# EXTRA STUFF - MOSTLY GENERIC
BATCH_SIZE=256
CONFIG_FILE="${CONFIG_FILE:-$ENV_CONFIG_FILE}"
HEADER=$ENV-$CONFIG_FILE
# BELOW IS GENERIC
SEED="${SEED:-1}"

echo -e "RUNNING: \`$HEADER\`" | tr a-z A-Z
echo -e "   -> SEED: $SEED"
echo -e "   -> ARGS PASSED IN: $@\n---\n"

./scripts/train_runs_entrypoint.sh -e $ENV -c $CONFIG_FILE -b $BATCH_SIZE -s $SEED "$@"
