#!/bin/bash

# Set default values for variables
ENV="${ENV:-hopper}"
ENV_CONFIG_FILE="${ENV_CONFIG_FILE:-medium_v2}"

PRETRAIN_CONFIG_PATH="${PRETRAIN_CONFIG_PATH:-'conf/corl/dt/halfcheetah/medium_v2.yaml'}"
WANDB_MODE="${WANDB_MODE:-online}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-10000}"
DOWNSTREAM_STEPS="${DOWNSTREAM_STEPS:-100000}"
EVAL_OUTPUT_SEQUENTIAL="${EVAL_SEQUENTIAL:-False}"
HEADER="$ENV-$ENV_CONFIG_FILE"
GROUP="${GROUP:-DOWNSTREAM-$HEADER}"
JOB_TYPE="${JOB_TYPE:-$HEADER}"

SEED="${SEED:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-1}"
BATCH_SIZE=256

# Print out the configuration details
echo -e "RUNNING: \`$HEADER\`" | tr a-z A-Z
echo -e "   -> SEED: $SEED | DOWNSTREAM-ENV: $ENV | CONFIG: $ENV_CONFIG_FILE | PRETRAIN_STEPS: $PRETRAIN_STEPS | DOWNSTREAM_STEPS: $DOWNSTREAM_STEPS | WANDB_MODE: $WANDB_MODE"
echo -e "   -> ARGS: $@\n---\n"


function run_downstream() {
    # Args:
    #   action_embed_class: The class of action embedding to use.
    #   tokenize_action: Whether to tokenize the action or not.
    #   patch_actions: Whether to patch the actions or not.
    #   eval_output_sequential: Whether to evaluate the output sequentially or not.
    #   job_type: The type of job to run.
    #   extra_arg: An extra argument to be added.

    local action_embed_class=$1
    local job_type=$2
    local tokenize_action=$3
    local patch_actions=$4
    local eval_output_sequential=$5
    local extra_arg=$6

    local cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python inctxdt/run.py --cmd=downstream \
        --seed=$SEED --device=cuda \
        --config_path=$PRETRAIN_CONFIG_PATH \
        --downstream.config_path=conf/corl/dt/$ENV/$ENV_CONFIG_FILE.yaml \
        --num_layers=4 \
        --num_heads=4 \
        --modal_embed.per_action_encode=False \
        --modal_embed.tokenize_action=$tokenize_action \
        --modal_embed.action_embed_class=$action_embed_class \
        --eval_output_sequential=$eval_output_sequential \
        --batch_size=$BATCH_SIZE \
        --update_steps=$PRETRAIN_STEPS \
        --downstream.patch_actions=$patch_actions \
        --log.mode=$WANDB_MODE \
        --log.group=$GROUP \
        --log.job_type=$job_type \
        $extra_arg \
        > output/logs/downstream-$job_type-$HEADER-$SEED.log 2>&1 &"

    echo $cmd
    eval $cmd
}

# format is
# run_downstream [action_embed_class]           [job_type]                              [tokenize_action] [patch_actions]   [eval_seq]  extra_arg
# ---
run_downstream "ActionTokenizedEmbedding"       "ActionTokenizedEmbedding"                          "True"      "False"     "False"
run_downstream "ActionTokenizedEmbedding"       "PatchedActionTokenizedEmbedding"                   "True"      "True"      "False"
run_downstream "ActionTokenizedSpreadEmbedding" "PatchedActionTokenizedSpreadEmbedding"             "True"      "True"      "False"
run_downstream "ActionTokenizedSpreadEmbedding" "PatchedActionTokenizedSpreadEmbeddingSequential"   "True"      "True"      "True"
run_downstream "ActionTokenizedSpreadEmbedding" "ActionTokenizedSpreadEmbedding"                    "True"      "False"     "False"
run_downstream "ActionEmbedding"                "PatchedActionEmbedding"                            "False"     "True"      "False" "--downstream.optim_use_defaultizer=False"
run_downstream "ActionEmbedding"                "ReusedOptimPatchedActionEmbedding"                 "False"     "True"      "False" "--downstream.optim_use_defaultizer=True"



