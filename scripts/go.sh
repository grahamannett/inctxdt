python run.py --cmd=train --device=cuda --batch_size=512 --epochs=500 --seq_len=30 --eval_episodes=10 --debug_note="FORGOT TO USE ACTIVATION OUTPUT" >output/run_latest.log
accelerate launch run.py --cmd=train --device=cuda --batch_size=512 --epochs=500 --seq_len=30 --eval_episodes=10 --debug_note="FORGOT TO USE ACTIVATION OUTPUT" >output/run_latest.log

# run

python run.py
