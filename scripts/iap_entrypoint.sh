#!/bin/bash

function run_and_wait_for_pids() {
    declare -a pids
    source scripts/compare/run_original_compare.sh $1
    echo 'waiting for jobs...'
    for pid in ${pids[*]}; do
        wait $pid
    done
}

CONFIG_PATH=conf/corl/dt/antmaze/umaze_v2.yaml
GROUP=TRAIN/antmaze-umaze_v2

run_and_wait_for_pids 10
run_and_wait_for_pids 11
run_and_wait_for_pids 12
run_and_wait_for_pids 13


CONFIG_PATH=conf/corl/dt/maze2d/umaze_v1.yaml
GROUP=TRAIN/maze2d-umaze_v1

run_and_wait_for_pids 10
run_and_wait_for_pids 11
run_and_wait_for_pids 12
run_and_wait_for_pids 13


CONFIG_PATH=conf/corl/dt