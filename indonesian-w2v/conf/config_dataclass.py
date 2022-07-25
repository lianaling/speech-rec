from dataclasses import dataclass

@dataclass
class TrainingParam:
    output_dir: str
    best_output_dir: str
    group_by_length: bool
    per_device_train_batch_size: int
    evaluation_strategy: str
    num_train_epochs: int
    fp16: bool
    gradient_checkpointing: bool
    save_steps: int
    eval_steps: int
    logging_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    save_total_limit: int

@dataclass
class Model:
    training_param: TrainingParam

@dataclass
class Dataset:
    iium_clean_train: str
    iium_clean_dev: str
    iium_clean_test: str
    malay_youtube: str

@dataclass
class TokenizerParam:
    feature_size: int
    sampling_rate: int
    padding_value: float
    do_normalize: bool
    return_attention_mask: bool
    save_directory: str

@dataclass
class Tokenizer:
    vocab_dict: str
    param: TokenizerParam

@dataclass
class Path:
    root_dir: str
    data_dir: str

@dataclass
class Checkpoint:
    model: str
    tokenizer: str

@dataclass
class Eval:
    model: str
    tokenizer: str
    result_dir: str
    dataset: str

@dataclass
class Finetune:
    model: Model
    tokenizer: Tokenizer
    checkpoint: Checkpoint

@dataclass
class Wav2Vec2Config:
    finetune: Finetune
    eval: Eval
    path: Path
    dataset: Dataset