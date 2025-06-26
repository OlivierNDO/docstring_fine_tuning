"""Inference utilities for fine-tuned models."""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Callable, Optional


class ModelInference:
    """Generic inference class for both base and fine-tuned models."""
    
    def __init__(self, model_path: str, prompt_formatter: Callable[[str], str],
                 use_quantization: bool = True, local_files_only: bool = False):
        self.model_path = model_path
        self.prompt_formatter = prompt_formatter
        self.use_quantization = use_quantization
        self.local_files_only = local_files_only
        
        # Check if this is a fine-tuned model (has adapter_config.json)
        is_finetuned = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        if is_finetuned:
            self._load_finetuned_model()
        else:
            self._load_base_model()
        
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.eos_token_id = self.tokenizer.eos_token_id

    def _load_finetuned_model(self):
        """Load fine-tuned model with LoRA adapter."""
        print(f"Loading fine-tuned model with LoRA adapter from: {self.model_path}")
        
        # Assume base model path - you might want to read this from adapter_config.json
        base_model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Setup quantization if needed
        bnb_config = None
        if self.use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=self.local_files_only,
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, 
            local_files_only=self.local_files_only,
            trust_remote_code=True
        )

    def _load_base_model(self):
        """Load base model without LoRA."""
        print(f"Loading base model from: {self.model_path}")
        
        # Setup quantization if needed
        bnb_config = None
        if self.use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=self.local_files_only,
            trust_remote_code=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            local_files_only=self.local_files_only,
            trust_remote_code=True
        )

    def format_prompt(self, instruction: str) -> str:
        """Format instruction into a proper prompt."""
        prompt = self.prompt_formatter(instruction)
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate_response(self, instruction: str, max_new_tokens: int = 512) -> str:
        """Generate response for given instruction."""
        input_text = self.format_prompt(instruction)
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt"
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id
        )

        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return generated.strip()


def compare_models(base_model_path: str, finetuned_model_path: str,
                  prompt_formatter: Callable[[str], str], test_cases: list):
    """Compare outputs between base and fine-tuned models."""
    
    print("Loading models for comparison...")
    base_model = ModelInference(base_model_path, prompt_formatter)
    finetuned_model = ModelInference(finetuned_model_path, prompt_formatter)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"TEST CASE {i}")
        print(f"{'='*50}")
        print("Input:")
        print(test_case)
        
        print(f"\n{'-'*20}")
        print("BASE MODEL OUTPUT:")
        print(f"{'-'*20}")
        base_output = base_model.generate_response(test_case)
        print(base_output)
        
        print(f"\n{'-'*20}")
        print("FINE-TUNED MODEL OUTPUT:")
        print(f"{'-'*20}")
        finetuned_output = finetuned_model.generate_response(test_case)
        print(finetuned_output)