
# NOTE: SMARTRL has a good dockerfile it seemed if i need tol
# RUN WITH SOMETHING LIKE
# docker run --gpus all -v /home/graham/.d4rl:/root/.d4rl -v /home/graham/code/incontext-trajectory-transformer/data:/workspace/data -it inctx:latest bash
# -----


FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /workspace

# APT stuff here
RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake wget vim
RUN apt-get install -y libosmesa6-dev


RUN mkdir -p /root/.mujoco \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz \
    && tar -xvzf mujoco210.tar.gz -C /root/.mujoco \
    && rm mujoco210.tar.gz

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/.mujoco/mujoco210/bin

RUN pip install wandb omegaconf stable-baselines3[extra] transformers dataset 'mujoco-py<2.2,>=2.1'
RUN pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
RUN pip install git+https://github.com/Farama-Foundation/Minari@main#egg=minari

# on first run d4rl needs to compile something
RUN python -c 'import gym; import d4rl'
# Pre-download dataset if necessary
# RUN python -c "import gym; import d4rl; [gym.make(f'{game}-{level}-v2').unwrapped.get_dataset() for level in \
#     ['medium', 'medium-replay', 'medium-expert', 'expert'] for game in ['halfcheetah', 'hopper', 'walker2d']];"


ADD conf /workspace/conf
ADD inctxdt /workspace/inctxdt
ADD scripts /workspace/scripts

ADD setup.py /workspace/setup.py
RUN pip install -e .