#!/bin/bash

SEED="${SEED:-1}"

# python inctxdt/run.py --config_path=conf/corl/dt/halfcheetah/medium_expert_v2.yaml --cmd=train --device=cuda  --num_layers=4 --num_heads=4 --batch_size=128 --modal_embed.action_embed_class=ActionEmbedding --modal_embed.tokenize_action=False --log.group=DEBUG-Spread --log.mode=online --seed=$SEED --log.job_type=ActionEmbedding-Baseline > /dev/null 2>&1 &

# # trying to figure out why ActionTokenizedSpreadEmbedding isnt doing as well as I thought it would
# python inctxdt/run.py --config_path=conf/corl/dt/halfcheetah/medium_expert_v2.yaml --cmd=train --device=cuda  --num_layers=4 --num_heads=4 --batch_size=128 --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --modal_embed.tokenize_action=True --log.group=DEBUG-Spread --log.mode=online --seed=$SEED --log.job_type=ActionTokenizedSpreadEmbedding

# same as above but would allow better tokenization
python inctxdt/run.py --config_path=conf/corl/dt/halfcheetah/medium_expert_v2.yaml --cmd=train --device=cuda  --num_layers=4 --num_heads=4 --batch_size=128 --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --modal_embed.tokenize_action=True --modal_embed.per_action_encode=True --modal_embed.num_bins=2000 --log.group=DEBUG-Spread --log.mode=online --seed=$SEED --log.job_type=PerActionActionTokenizedSpreadEmbedding
