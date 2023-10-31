#!/bin/bash

DEVICE="${DEVICE:-0}"

cmd="docker run -d --gpus all -v /home/graham/.d4rl:/root/.d4rl -v /home/graham/code/inctxdt/data:/workspace/data -e CUDA_VISIBLE_DEVICES=$DEVICE inctxdt/base:latest python $@"

echo running cmd $cmd
eval $cmd
