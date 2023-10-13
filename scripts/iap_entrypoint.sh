#!/bin/bash

source scripts/compare/run_original_compare.sh $@

echo 'waiting for jobs...'
for pid in ${pids[*]}; do
    wait $pid
done