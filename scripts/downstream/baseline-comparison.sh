#!/bin/bash


GROUP="${GROUP:-Testing-Downstream-BaselineComparison}"


# this is the baseline that should intentionally NOT learn the new embeddings/task
python inctxdt/run.py --cmd=downstream --seed=10 --device=cuda --config_path=conf/corl/dt/hopper/medium_v2.yaml --downstream.config_path=conf/corl/dt/halfcheetah/medium_v2.yaml --num_layers=4 --num_heads=4 --modal_embed.per_action_encode=False --modal_embed.tokenize_action=False --modal_embed.action_embed_class=ActionEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=10000 --downstream.patch_actions=True --downstream.update_optim_actions=False --downstream.patch_states=True --downstream.update_optim_states=False --log.group=$GROUP --log.job_type=ActionEmbedding-PoorResult --downstream.optim_use_default=True --log.mode=online > /dev/null 2>&1 &# this

# this is the baseline that should be marginally better than above
python inctxdt/run.py --cmd=downstream --seed=10 --device=cuda --config_path=conf/corl/dt/hopper/medium_v2.yaml --downstream.config_path=conf/corl/dt/halfcheetah/medium_v2.yaml --num_layers=4 --num_heads=4 --modal_embed.per_action_encode=False --modal_embed.tokenize_action=False --modal_embed.action_embed_class=ActionEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=10000 --downstream.patch_actions=True --downstream.update_optim_actions=True --downstream.patch_states=True --downstream.update_optim_states=True --log.group=$GROUP --log.job_type=ActionEmbedding-GoodResult --downstream.optim_use_default=True --log.mode=online > /dev/null 2>&1 &# this is

# and then if we just start with fresh optim/scheduler
python inctxdt/run.py --cmd=downstream --seed=10 --device=cuda --config_path=conf/corl/dt/hopper/medium_v2.yaml --downstream.config_path=conf/corl/dt/halfcheetah/medium_v2.yaml --num_layers=4 --num_heads=4 --modal_embed.per_action_encode=False --modal_embed.tokenize_action=False --modal_embed.action_embed_class=ActionEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=10000 --downstream.patch_actions=True --downstream.update_optim_actions=True --downstream.patch_states=True --downstream.update_optim_states=True --log.group=$GROUP --log.job_type=ActionEmbedding-NewOptim --downstream.optim_use_default=False --log.mode=online > /dev/null 2>&1 &