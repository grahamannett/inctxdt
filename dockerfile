
# NOTE: believe i based this off of the dockerfile from smartrl if i need to fix anything
#
# to build:
# > docker build --build-arg="WANDB_API_KEY=$(cat secrets/wandb)" -t inctxdt/base:latest -f dockerfile .
#
# to run:
#  - in interactive mode:
#       > docker run --gpus all -v /home/graham/.d4rl:/root/.d4rl -v /home/graham/code/inctxdt/data:/workspace/data -it inctxdt:latest bash
#
#  - in non-interactive mode (e.g. experiments/runs):
#       > docker run --gpus all -v /home/graham/.d4rl:/root/.d4rl -v /home/graham/code/inctxdt/data:/workspace/data -it inctxdt/base:latest scripts/runs/halfcheetah.sh

# notes:
# - add the volumes for data (i.e. ~/.d4rl) otherwise will take very long or image may be too large
# -----

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
ARG WANDB_API_KEY=none

WORKDIR /workspace

# APT stuff here
RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake wget vim
RUN apt-get install -y libosmesa6-dev


RUN mkdir -p /root/.mujoco \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz \
    && tar -xvzf mujoco210.tar.gz -C /root/.mujoco \
    && rm mujoco210.tar.gz

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/.mujoco/mujoco210/bin

RUN pip install wandb omegaconf stable-baselines3[extra] transformers dataset "mujoco-py<2.2,>=2.1" "cython<3"
RUN pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
RUN pip install git+https://github.com/Farama-Foundation/Minari@main#egg=minari
RUN pip install pyrallis accelerate tensordict fast-pytorch-kmeans scikit-learn

# on first run d4rl needs to compile something
RUN python -c 'import gym; import d4rl'
# Pre-download dataset if necessary
# RUN python -c "import gym; import d4rl; [gym.make(f'{game}-{level}-v2').unwrapped.get_dataset() for level in \
#     ['medium', 'medium-replay', 'medium-expert', 'expert'] for game in ['halfcheetah', 'hopper', 'walker2d']];"


ADD conf /workspace/conf
ADD inctxdt /workspace/inctxdt
ADD scripts /workspace/scripts
RUN chmod +x /workspace/scripts/*.sh



ADD pyproject.toml /workspace/pyproject.toml
RUN pip install -e .

# set keys for logging
ENV WANDB_API_KEY=$WANDB_API_KEY

# use cmd instead of entrypoint so we can use bash if needed for debugging
# CMD [ "/workspace/scripts/entrypoint.sh" ]