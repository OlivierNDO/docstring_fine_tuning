"""Model setup and configuration utilities."""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from typing import Tuple
from src.config import TrainingConfig


def setup_model_and_tokenizer(config: TrainingConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Setup model and tokenizer with LoRA configuration."""
    
    print(f"Loading model from: {config.model_path}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup quantization if needed
    bnb_config = None
    if config.use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, 
        local_files_only=config.local_files_only,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens if needed
    if not (hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template):
        special_tokens = ["<|user|>", "<|assistant|>", "<|end|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if config.use_quantization else torch.float16,
        device_map="auto",
        local_files_only=config.local_files_only,
        trust_remote_code=True
    )
    
    # Resize embeddings if we added special tokens
    if not (hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template):
        model.resize_token_embeddings(len(tokenizer))
    
    # Prepare for LoRA
    if config.use_quantization:
        model = prepare_model_for_kbit_training(model)
        
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer