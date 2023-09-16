install related:

`pip install "gymnasium[all]" minari gymnasium-robotics`

then

`pip install -e .`

if you need mujoco:
`wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && tar -xf mujoco210-linux-x86_64.tar.gz && mkdir ~/.mujoco && mv mujoco210 ~/.mujoco/`

you likely need `pip install "cython<3"` unless the cython bug is fixed


apt related things you probably need:

sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev patchelf


# to run the baseline

`python run.py --batch_size=400 --epochs=10 --seq_len=10 --num_workers=10 --eval_episodes=10


# to run with docker

docker run -v /home/graham/.d4rl:/root/.d4rl -v /home/graham/.cache:/root/.cache

docker run --rm --ipc=host --gpus all -it -v $HOME/.d4rl:/root/.d4rl -v $HOME/.cache:/root/.cache inctxdt/base:latest /bin/bash


# to run on hpc:
follow dockerfile for setting up mujoco and python libraries. since you cant install with apt you will also need to use the following `conda install mesalib glew glfw patchelf -y`.  Also make sure you set the LD_LIBRARY_PATH
