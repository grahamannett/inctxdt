#!/bin/bash

# need singularity image to run on clusters so process would be:
# 1. build docker image locally
# 2. build singularity image from docker image
# 3. transfer singularity image to cluster with scp
# 4. run singularity image on cluster


SIF_FILE_BUILD_PATH=/data/graham/simages/inctxdt.sif
DOCKER_IMAGE_NAME=inctxdt/base:latest

# borah related
CLUSTER_SIF_DIR=/bsuhome/gannett/scratch/simages

# first just build the docker image locally
docker build --build-arg="WANDB_API_KEY=$(cat secrets/wandb)" -t $DOCKER_IMAGE_NAME -f dockerfile .

# build singularity image locally and put in the singularity images folder
singularity build $SIF_FILE_BUILD_PATH docker-daemon://$DOCKER_IMAGE_NAME

# transfer to borah cluster in the ~/scratch folder
scp $SIF_FILE_BUILD_PATH borah:$CLUSTER_SIF_DIR