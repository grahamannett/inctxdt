#!/bin/bash
#SBATCH -J sft_test         # job name
#SBATCH -o log_slurm.o%j    # log file name (%j expands to jobID)
#SBATCH -n 1                 # total number of tasks requested
#SBATCH -N 1                 # number of nodes you want to run on
#SBATCH --cpus-per-task 48
#SBATCH --gres=gpu:8        # request 8 gpu
#SBATCH -p nam-bio            # queue (partition)
#SBATCH -t 06:00:00         # run time (hh:mm:ss)

# Activate the conda environment
. ~/.bashrc

# Load the cudnn module
# module load cudnn8.4-cuda11.4
module load conda
module load cuda11.7/toolkit/11.7.1
module load cudnn8.5-cuda11.7/8.5.0.96

conda activate ll

export WANDB_PROJECT=sft_test
export HF_HOME=$HOME/scratch/huggingface
# Run your python code
# Replace MYSCRIPT.py with the path to your python script
cd $HOME/scratch/llama-tests
echo "STARTING...===$(pwd)"
echo "WITH HF_HOME: $HF_HOME"
echo "WITH PYTHON: $(which python)"

# accelerate launch sft_llama2.py --group_by_length=False
python sft_llama2.py --group_by_length=False --model_name="gpt2"