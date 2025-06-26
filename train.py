"""Main training script for fine-tuning language models."""

from dotenv import load_dotenv
load_dotenv()

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)

from src.config import TrainingConfig, setup_environment
from src.model_setup import setup_model_and_tokenizer
from src.data_processing import prepare_datasets
from src.callbacks import NaNDetectionCallback, GradientMonitor, EpochProgressLogger
from src.inference import compare_models

# Import your data
from data.docstring_dataset import DOCSTRING_EXAMPLES


def create_prompt_formatter():
    """Create a prompt formatter function for your specific use case."""
    def format_prompt(instruction: str) -> str:
        return f"Generate a Python docstring for the following function:\n\n{instruction}"
    return format_prompt


def main():
    """Main training function."""
    # Setup environment
    setup_environment()
    
    # Load configuration
    config = TrainingConfig()
    
    print(f"Records in training set: {len(DOCSTRING_EXAMPLES)}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Create prompt formatter
    prompt_formatter = create_prompt_formatter()
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(
        DOCSTRING_EXAMPLES,
        tokenizer,
        prompt_formatter,
        test_size=config.test_size,
        random_state=config.random_state,
        max_length=config.max_length
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        disable_tqdm=False,
        save_total_limit=config.save_total_limit,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        
        # Stability settings
        bf16=False,
        fp16=False,
        max_grad_norm=config.max_grad_norm,
        
        dataloader_pin_memory=False,
        dataloader_drop_last=True,
        report_to=None,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Add callbacks
    trainer.add_callback(EpochProgressLogger(
        train_dataset_size=len(train_dataset),
        val_dataset_size=len(val_dataset),
        batch_size=training_args.per_device_train_batch_size,
        grad_accum=training_args.gradient_accumulation_steps,
        num_epochs=training_args.num_train_epochs
    ))
    
    trainer.add_callback(GradientMonitor())
    trainer.add_callback(NaNDetectionCallback())
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving best model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Test the model
    test_cases = [
        "def has_new_orleans_area_code(phone_number: str):\n    return phone_number[:3] == '504'",
        "def calculate_area(radius):\n    return 3.14159 * radius ** 2",
        "def get_user_by_id(user_id):\n    return database.query(f'SELECT * FROM users WHERE id = {user_id}')",
    ]
    
    print("\n" + "="*60)
    print("COMPARING MODEL OUTPUTS")
    print("="*60)
    
    compare_models(
        base_model_path=config.model_path,
        finetuned_model_path=config.output_dir,
        prompt_formatter=prompt_formatter,
        test_cases=test_cases
    )


if __name__ == "__main__":
    main()