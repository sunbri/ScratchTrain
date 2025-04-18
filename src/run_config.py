from dataclasses import dataclass

@dataclass
class GeneralConfig:
    input_train: str
    input_val: str
    checkpoint_dir: str
    model: str
    batch_size: int
    max_iters: int
    device: str
    eval_interval: int
    eval_iters: int
    load_from_checkpoint: bool

@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: int

@dataclass
class OptimizerConfig:
    max_learning_rate: float
    min_learning_rate: float
    warmup_iters: int
    cosine_cycle_iters: int
    weight_decay: float
    beta_1: float
    beta_2: float
    eps_adam: float
    max_norm: float