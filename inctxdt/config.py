import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


@dataclass
class EnvSpec:
    episode_len: int
    seq_len: int
    env_name: str

    state_dim: int  # = None
    action_dim: int  # = None

    @classmethod
    def from_env(cls, env, episode_len: int, seq_len: int):
        return cls(
            episode_len=episode_len,
            seq_len=seq_len,
            env_name=env.env_name,
            action_dim=env.action_dim,
            state_dim=env.state_dim,
        )


@dataclass
class CentroidConfig:
    n_clusters: int = 100
    mode: str = "cosine"

    # number of batches to train from centroids
    n_batches: int = 10


@dataclass
class EvalConfig:
    model_dir: str = None
    reward_scale: float = None

    def __post_init__(self):
        if self.reward_scale is None:
            self.reward_scale = Config.reward_scale


@dataclass
class LogConfig:
    # init params
    name: str = None
    project: str = "inctxdt"
    group: str = "train"
    mode: str = "disabled"
    tags: list[str] = field(default_factory=list)  #

    # logging related
    log_every: int = 100

    job_type: str = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class ModalEmbedConfig:
    tokenize_action: bool = False
    action_embed_class: str = "ActionEmbedding"
    num_bins: int = 3000
    strategy: str = "quantile"
    per_action_encode: bool = True
    EmbedClass: str = "SequentialAction"

    def __post_init__(self):
        if self.tokenize_action and self.action_embed_class == "ActionEmbedding":
            raise ValueError("ActionEmbedding doesnt work if tokenzing actions")


@dataclass
class Config:
    # LEAVE THESE TO BE ABLE TO USE THE CONFIGS FROM CORL
    # --- NOT USED ---
    # env_name: str = None
    name: str = None
    group: str = None
    project: str = None
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False

    # stuff to be able to use the config from corl
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1

    train_seed: int = 10
    eval_seed: int = 10

    max_action: float = 1.0

    # dataset_name: str = "pointmaze-umaze-v1"
    # dataset_type: str = "minari"
    env_name: Union[list[str], str] = "halfcheetah-medium-v2"
    dataset_type: str = "d4rl"
    dataset_min_length: int = None

    device: str = "cpu"
    cmd: str = "train"

    exp_root: str = "output"
    exp_name: str = "latest"
    save_model: bool = False

    update_steps: int = 100_000
    epochs: int = 1
    num_workers: int = 8
    batch_size: int = 32
    shuffle: bool = True
    n_batches: int = -1

    # dataset
    seq_len: int = 20
    episode_len: int = 1000
    max_num_episodes: int = 2

    num_layers: int = 3
    num_heads: int = 1
    embedding_dim: int = 128

    adist: bool = False  # accelerate distributed
    dist: bool = False  # pytorch distributed

    modal_embed: ModalEmbedConfig = field(default_factory=ModalEmbedConfig)
    log: LogConfig = field(default_factory=LogConfig)
    centroids: CentroidConfig = field(default_factory=CentroidConfig)

    # optim
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    warmup_steps: int = 1_000  # 10000

    # loss related
    use_secondary_loss: bool = False
    loss_reduction: str = "none"  # "mean"
    state_loss_scale: float = 0.1
    rewards_loss_scale: float = 0.1

    clip_grad: Optional[float] = 0.25

    # eval
    reward_scale: float = 1  # was 0.001
    target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    eval_every: int = 1_000
    eval_episodes: int = 5
    eval_before_train: bool = False
    eval_output_sequential: bool = False

    debug: str = None
    seed: int = 42

    def __post_init__(self):
        self._check_dataset()

    def _check_dataset(self):
        dataset_type = self.dataset_type.split("_")[0]
        assert dataset_type in ["minari", "d4rl"], f"dataset_type: {dataset_type} not supported"

    def _get_dataset_type(self):
        import minari

        if self.dataset_name in minari.list_remote_datasets():
            return "minari"

        return "d4rl"

    @property
    def exp_dir(self) -> str:
        if self.log.name:
            return f"{self.exp_root}/{self.log.name}"

        return f"{self.exp_root}/{self.exp_name}"

    @property
    def console_info(self) -> str:
        from pyrallis import dump

        return dump(self)

    @classmethod
    def use_accelerate(cls, accelerator) -> None:
        cls._accelerator = accelerator

    def __repr__(self):
        output_str = ""
        if self.debug:  # allow extra debug message to be printed
            output_str += f"\n\n==>🤠|> {self.debug.upper()} <|🤠<==\n\n"

        output_str += f"=== Config ===\n"
        output_str += self.console_info
        return output_str
