#!/bin/bash

# MAIN
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-1}"
ENV="${ENV:-hopper}"
ENV_CONFIG_FILE="${ENV_CONFIG_FILE:-medium_v2}"
WANDB_MODE="${WANDB_MODE:-online}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-10000}"
DOWNSTREAM_STEPS="${DOWNSTREAM_STEPS:-100000}"

EVAL_OUTPUT_SEQUENTIAL="${EVAL_SEQUENTIAL:-False}"


# EXTRA STUFF - MOSTLY GENERIC
BATCH_SIZE=256
# CONFIG_FILE="${CONFIG_FILE:-$ENV_CONFIG_FILE}"
HEADER="$ENV-$ENV_CONFIG_FILE"
GROUP="${GROUP:-DOWNSTREAM-$HEADER}"
JOB_TYPE="${JOB_TYPE:-$HEADER}"

# BELOW IS GENERIC
SEED="${SEED:-1}"

echo -e "RUNNING: \`$HEADER\`" | tr a-z A-Z
echo -e "   -> SEED: $SEED | DOWNSTREAM-ENV: $ENV | CONFIG: $ENV_CONFIG_FILE | PRETRAIN_STEPS: $PRETRAIN_STEPS | DOWNSTREAM_STEPS: $DOWNSTREAM_STEPS | WANDB_MODE: $WANDB_MODE"
echo -e "   -> ARGS: $@\n---\n"



# ActionTokenizedEmbedding
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python inctxdt/run.py --cmd=downstream \
    --seed=$SEED --device=cuda \
    --config_path=conf/corl/dt/halfcheetah/medium_v2.yaml \
    --downstream.config_path=conf/corl/dt/$ENV/$ENV_CONFIG_FILE.yaml \
    --num_layers=4 \
    --num_heads=4 \
    --modal_embed.per_action_encode=False \
    --modal_embed.tokenize_action=True \
    --modal_embed.action_embed_class=ActionTokenizedEmbedding \
    --eval_output_sequential=False \
    --batch_size=$BATCH_SIZE \
    --update_steps=$PRETRAIN_STEPS \
    --downstream._patch_actions=False \
    --log.mode=$WANDB_MODE \
    --log.group=$GROUP \
    --log.job_type=ActionTokenizedEmbedding \
    $@ > output/logs/downstream-ActionTokenizedEmbedding-$HEADER-$SEED.log 2>&1 &

# PatchedActionTokenizedEmbedding
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python inctxdt/run.py --cmd=downstream \
    --seed=$SEED --device=cuda \
    --config_path=conf/corl/dt/halfcheetah/medium_v2.yaml \
    --downstream.config_path=conf/corl/dt/$ENV/$ENV_CONFIG_FILE.yaml \
    --num_layers=4 \
    --num_heads=4 \
    --modal_embed.per_action_encode=False \
    --modal_embed.tokenize_action=True \
    --modal_embed.action_embed_class=ActionTokenizedEmbedding \
    --eval_output_sequential=False \
    --batch_size=$BATCH_SIZE \
    --update_steps=$PRETRAIN_STEPS \
    --downstream._patch_actions=True \
    --log.mode=$WANDB_MODE \
    --log.group=$GROUP \
    --log.job_type=PatchedActionTokenizedEmbedding \
    $@ > output/logs/downstream-PatchedActionTokenizedEmbedding-$HEADER-$SEED.log 2>&1 &

#  PatchedActionTokenizedSpreadEmbedding
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python inctxdt/run.py --cmd=downstream \
    --seed=$SEED --device=cuda \
    --config_path=conf/corl/dt/halfcheetah/medium_v2.yaml \
    --downstream.config_path=conf/corl/dt/$ENV/$ENV_CONFIG_FILE.yaml \
    --num_layers=4 \
    --num_heads=4 \
    --modal_embed.per_action_encode=False \
    --modal_embed.tokenize_action=True \
    --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding \
    --eval_output_sequential=False \
    --batch_size=$BATCH_SIZE \
    --update_steps=$PRETRAIN_STEPS \
    --downstream._patch_actions=True \
    --log.mode=$WANDB_MODE \
    --log.group=$GROUP \
    --log.job_type=PatchedActionTokenizedSpreadEmbedding \
    $@ > output/logs/downstream-PatchedActionTokenizedSpreadEmbedding-$HEADER-$SEED.log 2>&1 &

#  PatchedActionTokenizedSpreadEmbeddingSequential
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python inctxdt/run.py --cmd=downstream \
    --seed=$SEED --device=cuda \
    --config_path=conf/corl/dt/halfcheetah/medium_v2.yaml \
    --downstream.config_path=conf/corl/dt/$ENV/$ENV_CONFIG_FILE.yaml \
    --num_layers=4 \
    --num_heads=4 \
    --modal_embed.per_action_encode=False \
    --modal_embed.tokenize_action=True \
    --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding \
    --eval_output_sequential=True \
    --batch_size=$BATCH_SIZE \
    --update_steps=$PRETRAIN_STEPS \
    --downstream._patch_actions=True \
    --log.mode=$WANDB_MODE \
    --log.group=$GROUP \
    --log.job_type=PatchedActionTokenizedSpreadEmbeddingSequential \
    $@ > output/logs/downstream-PatchedActionTokenizedSpreadEmbeddingSequential-$HEADER-$SEED.log 2>&1 &

# ActionTokenizedSpreadEmbedding
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python inctxdt/run.py --cmd=downstream \
    --seed=$SEED --device=cuda \
    --config_path=conf/corl/dt/halfcheetah/medium_v2.yaml \
    --downstream.config_path=conf/corl/dt/$ENV/$ENV_CONFIG_FILE.yaml \
    --num_layers=4 \
    --num_heads=4 \
    --modal_embed.per_action_encode=False \
    --modal_embed.tokenize_action=True \
    --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding \
    --eval_output_sequential=False \
    --batch_size=$BATCH_SIZE \
    --update_steps=$PRETRAIN_STEPS \
    --downstream._patch_actions=False \
    --log.mode=$WANDB_MODE \
    --log.group=$GROUP \
    --log.job_type=ActionTokenizedSpreadEmbedding \
    $@ > output/logs/downstream-ActionTokenizedSpreadEmbedding-$HEADER-$SEED.log 2>&1 &

# PatchedActionEmbedding
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python inctxdt/run.py --cmd=downstream \
    --seed=$SEED --device=cuda \
    --config_path=conf/corl/dt/halfcheetah/medium_v2.yaml \
    --downstream.config_path=conf/corl/dt/$ENV/$ENV_CONFIG_FILE.yaml \
    --num_layers=4 \
    --num_heads=4 \
    --modal_embed.per_action_encode=False \
    --modal_embed.tokenize_action=True \
    --modal_embed.action_embed_class=ActionEmbedding \
    --eval_output_sequential=False \
    --batch_size=$BATCH_SIZE \
    --update_steps=$PRETRAIN_STEPS \
    --downstream._patch_actions=True \
    --log.mode=$WANDB_MODE \
    --log.group=$GROUP \
    --log.job_type=PatchedActionEmbedding \
    $@ > output/logs/downstream-PatchedActionEmbedding-$HEADER-$SEED.log 2>&1 &

# ActionEmbedding
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python inctxdt/run.py --cmd=downstream \
    --seed=$SEED --device=cuda \
    --config_path=conf/corl/dt/halfcheetah/medium_v2.yaml \
    --downstream.config_path=conf/corl/dt/$ENV/$ENV_CONFIG_FILE.yaml \
    --num_layers=4 \
    --num_heads=4 \
    --modal_embed.per_action_encode=False \
    --modal_embed.tokenize_action=False \
    --modal_embed.action_embed_class=ActionEmbedding \
    --eval_output_sequential=False \
    --batch_size=$BATCH_SIZE \
    --update_steps=$PRETRAIN_STEPS \
    --downstream._patch_actions=False \
    --log.mode=$WANDB_MODE \
    --log.group=$GROUP \
    --log.job_type=ActionEmbedding \
    $@ > output/logs/downstream-ActionEmbedding-$HEADER-$SEED.log 2>&1 &
