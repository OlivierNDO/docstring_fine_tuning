"""Configuration management for the fine-tuning pipeline."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    model_path: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir: str = "./tiny_llama_finetuned"
    local_files_only: bool = False
    use_quantization: bool = True
    
    # Training hyperparameters
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data parameters
    max_length: int = 512
    test_size: float = 0.2
    random_state: int = 42
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Logging
    logging_steps: int = 1
    save_total_limit: int = 3
    early_stopping_patience: int = 999


def setup_environment():
    """Setup environment variables and logging."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    
    import logging
    logging.basicConfig(level=logging.INFO)
    
    import transformers
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_warning()