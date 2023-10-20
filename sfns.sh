#!/bin/bash

DEVICE="${DEVICE:-0}"
SEED="${SEED:-10}"

GROUP="${GROUP:-DOWNSTREAM/AntmazeUmaze-TrainedHalfCheetahMediumExpert}"
# CONFIG PATH is PRETRAIN
CONFIG_PATH="${CONFIG_PATH:-conf/corl/dt/halfcheetah/medium_expert_v2.yaml}"
DOWNSTREAM_CONFIG_PATH="${DOWNSTREAM_CONFIG_PATH:-conf/corl/dt/antmaze/umaze_v2.yaml}"
UPDATE_STEPS="${UPDATE_STEPS:-50000}"

NUM_LAYERS="${NUM_LAYERS:-3}"
NUM_HEADS="${NUM_HEADS:-1}"


run_docker() {
    echo running docker
}


run_debug() {
    echo running debug
}