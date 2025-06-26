"""Data processing utilities for the fine-tuning pipeline."""

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Callable, Optional
from transformers import AutoTokenizer


class SimpleDataset(Dataset):
    """Generic dataset for instruction-response fine-tuning."""
    
    def __init__(self, data: List[Dict], tokenizer: AutoTokenizer, 
                 prompt_formatter: Callable[[str], str], max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_formatter = prompt_formatter
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt = self.prompt_formatter(item['instruction'])
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": item['response']}
        ]
        
        full_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        
        # Mask padding tokens
        labels[attention_mask == 0] = -100
        
        # Mask the prompt (only train on assistant response)
        assistant_start_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        assistant_start_tokens = self.tokenizer(
            assistant_start_text, 
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length
        )["input_ids"]
        
        mask_length = len(assistant_start_tokens)
        if mask_length < len(input_ids):
            labels[:mask_length] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def filter_valid_examples(examples: List[Dict], tokenizer: AutoTokenizer, 
                         prompt_formatter: Callable[[str], str], max_length: int = 512) -> List[Dict]:
    """Filter examples that are too long for the model."""
    def is_valid_example(example: Dict) -> bool:
        prompt = prompt_formatter(example['instruction'])
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": example['response']}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return len(tokenizer.tokenize(text)) <= max_length
    
    valid_examples = [ex for ex in examples if is_valid_example(ex)]
    print(f"Filtered {len(examples)} -> {len(valid_examples)} examples")
    return valid_examples


def prepare_datasets(
    examples: List[Dict], 
    tokenizer: AutoTokenizer,
    prompt_formatter: Callable[[str], str],
    test_size: float = 0.2,
    random_state: int = 42,
    max_length: int = 512
) -> Tuple[SimpleDataset, SimpleDataset]:
    """Prepare train and validation datasets."""
    
    # Filter valid examples
    valid_examples = filter_valid_examples(examples, tokenizer, prompt_formatter, max_length)
    
    # Split data
    train_data, val_data = train_test_split(
        valid_examples, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = SimpleDataset(train_data, tokenizer, prompt_formatter, max_length)
    val_dataset = SimpleDataset(val_data, tokenizer, prompt_formatter, max_length)
    
    return train_dataset, val_dataset