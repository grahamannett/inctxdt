#!/bin/bash

# Set default values for variables
ENV="${ENV:-hopper}"
ENV_CONFIG_FILE="${ENV_CONFIG_FILE:-medium_v2}"
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
    local tokenize_action=$2
    local patch_actions=$3
    local eval_output_sequential=$4
    local job_type=$5
    local extra_arg=$6

    local cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python inctxdt/run.py --cmd=downstream \
        --seed=$SEED --device=cuda \
        --config_path=conf/corl/dt/halfcheetah/medium_v2.yaml \
        --downstream.config_path=conf/corl/dt/$ENV/$ENV_CONFIG_FILE.yaml \
        --num_layers=4 \
        --num_heads=4 \
        --modal_embed.per_action_encode=False \
        --modal_embed.tokenize_action=$tokenize_action \
        --modal_embed.action_embed_class=$action_embed_class \
        --eval_output_sequential=$eval_output_sequential \
        --batch_size=$BATCH_SIZE \
        --update_steps=$PRETRAIN_STEPS \
        --downstream._patch_actions=$patch_actions \
        --log.mode=$WANDB_MODE \
        --log.group=$GROUP \
        --log.job_type=$job_type \
        $extra_arg \
        > output/logs/downstream-$job_type-$HEADER-$SEED.log 2>&1 &"

    echo $cmd
    eval $cmd
}

# format is
# run_downstream [action_embed_class]  [tokenize_action] [patch_actions] eval_output_sequential job_type extra_arg
# ---
run_downstream "ActionTokenizedEmbedding" "True" "False" "False" "ActionTokenizedEmbedding"
run_downstream "ActionTokenizedEmbedding" "True" "True" "False" "PatchedActionTokenizedEmbedding"
run_downstream "ActionTokenizedSpreadEmbedding" "True" "True" "False" "PatchedActionTokenizedSpreadEmbedding"
run_downstream "ActionTokenizedSpreadEmbedding" "True" "True" "True" "PatchedActionTokenizedSpreadEmbeddingSequential"
run_downstream "ActionTokenizedSpreadEmbedding" "True" "False" "False" "ActionTokenizedSpreadEmbedding"
run_downstream "ActionEmbedding" "False" "True" "False" "ReusedOptimPatchedActionEmbedding" "--downstream._reuse_optimizer=True"
# if ActionEmbedding you need to patch actions
# run_downstream "ActionEmbedding" "False" "False" "False" "ActionEmbedding" "--downstream._reuse_optimizer=True"


