
DEVICE=0
SEED=10

# update_steps=10000


# baseline with low training steps
CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --cmd=downstream --device=cuda --config_path=conf/corl/dt/hopper/medium_v2.yaml --downstream.config_path=conf/corl/dt/halfcheetah/medium_v2.yaml --num_layers=3 --num_heads=1 --modal_embed.per_action_encode=False --modal_embed.tokenize_action=False --modal_embed.action_embed_class=ActionEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=1000 --downstream.patch_actions=True --downstream.update_optim_actions=True --downstream.patch_states=True --downstream.update_optim_states=True --downstream.optim_only_patched=True --log.group=Downstream-HalfCheetah --log.job_type=OnlyPatch-ActionEmbedding --log.mode=online --seed=$SEED  > /dev/null 2>&1 &

# baseline with NO ACTION TRAINING
CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --cmd=downstream --device=cuda --config_path=conf/corl/dt/hopper/medium_v2.yaml --downstream.config_path=conf/corl/dt/halfcheetah/medium_v2.yaml --num_layers=3 --num_heads=1 --modal_embed.per_action_encode=False --modal_embed.tokenize_action=False --modal_embed.action_embed_class=ActionEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=1000 --downstream.patch_actions=True --downstream.update_optim_actions=False --downstream.patch_states=True --downstream.update_optim_states=True --downstream.optim_only_patched=True --log.group=Downstream-HalfCheetah --log.job_type=OnlyStates-ActionEmbedding --log.mode=online --seed=$SEED  > /dev/null 2>&1 &


# spread with low training steps
CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --cmd=downstream  --device=cuda --config_path=conf/corl/dt/hopper/medium_v2.yaml --downstream.config_path=conf/corl/dt/halfcheetah/medium_v2.yaml --num_layers=3 --num_heads=1 --modal_embed.per_action_encode=True --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=1000 --downstream.patch_actions=True --downstream.update_optim_actions=True --downstream.patch_states=True --downstream.update_optim_states=True --downstream.optim_only_patched=True --log.group=Downstream-HalfCheetah --log.job_type=Low-OnlyPatch-ActionTokenizedSpreadEmbedding  --log.mode=online --seed=$SEED > /dev/null 2>&1 &

# spread with good
CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --cmd=downstream  --device=cuda --config_path=conf/corl/dt/hopper/medium_v2.yaml --downstream.config_path=conf/corl/dt/halfcheetah/medium_v2.yaml --num_layers=3 --num_heads=1 --modal_embed.per_action_encode=True --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=25000 --downstream.patch_actions=True --downstream.update_optim_actions=True --downstream.patch_states=True --downstream.update_optim_states=True --downstream.optim_only_patched=True --log.group=Downstream-HalfCheetah --log.job_type=OnlyPatch-ActionTokenizedSpreadEmbedding  --log.mode=online --seed=$SEED > /dev/null 2>&1 &

# spread with NO ACTION TRAINING
CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --cmd=downstream  --device=cuda --config_path=conf/corl/dt/hopper/medium_v2.yaml --downstream.config_path=conf/corl/dt/halfcheetah/medium_v2.yaml --num_layers=3 --num_heads=1 --modal_embed.per_action_encode=True --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=25000 --downstream.patch_actions=True --downstream.update_optim_actions=False --downstream.patch_states=True --downstream.update_optim_states=True --downstream.optim_only_patched=True --log.group=Downstream-HalfCheetah --log.job_type=OnlyStates-ActionTokenizedSpreadEmbedding  --log.mode=online --seed=$SEED > /dev/null 2>&1 &


# spread with ALL
CUDA_VISIBLE_DEVICES=$DEVICE python inctxdt/run.py --cmd=downstream  --device=cuda --config_path=conf/corl/dt/hopper/medium_v2.yaml --downstream.config_path=conf/corl/dt/halfcheetah/medium_v2.yaml --num_layers=3 --num_heads=1 --modal_embed.per_action_encode=True --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=25000 --downstream.patch_actions=True --downstream.update_optim_actions=True --downstream.patch_states=True --downstream.update_optim_states=True --downstream.optim_only_patched=False --log.group=Downstream-HalfCheetah --log.job_type=ActionTokenizedSpreadEmbedding  --log.mode=online --seed=$SEED > /dev/null 2>&1 &


# --downstream.optim_use_default=True


docker run -d --gpus all -v /home/graham/.d4rl:/root/.d4rl -v /home/graham/code/inctxdt/data:/workspace/data inctxdt/base:latest python inctxdt/run.py --cmd=downstream  --device=cuda --config_path=conf/corl/dt/hopper/medium_v2.yaml --downstream.config_path=conf/corl/dt/halfcheetah/medium_v2.yaml --num_layers=3 --num_heads=1 --modal_embed.per_action_encode=True --modal_embed.tokenize_action=True --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --eval_output_sequential=False --batch_size=128 --update_steps=25000 --downstream.patch_actions=True --downstream.update_optim_actions=True --downstream.patch_states=True --downstream.update_optim_states=True --downstream.optim_only_patched=False --log.group=Downstream-HalfCheetah --log.job_type=ActionTokenizedSpreadEmbedding  --log.mode=online --seed=10