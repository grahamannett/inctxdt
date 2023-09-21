#!/bin/bash

# python run.py --cmd=train --device=cuda--num_layers=4 --num_heads=4 --embedding_dim=128 --batch_size=128 --epochs=25 --seq_len=10 --num_workers=10 --eval_episodes=10 --eval_before_train=True --eval_output_sequential=False --modal_embed.action_embed_class=ActionTokenizedEmbedding --modal_embed.tokenize_action=True

python run.py --cmd=train --device=cuda--num_layers=4 --num_heads=4 --embedding_dim=128 --batch_size=128 --epochs=25 --seq_len=10 --num_workers=10 --eval_episodes=10 --eval_before_train=True --eval_output_sequential=True --modal_embed.action_embed_class=ActionTokenizedSpreadEmbedding --modal_embed.tokenize_action=False


# --save_model=True --exp_name=tokenized-$dataset --dataset_name=$dataset --log.name=tokenized-$dataset --log.mode=online > output/logs/tokenized-$dataset.log 2>&1 &