"""Training callbacks for monitoring and debugging."""

import math
import numpy as np
import torch
from transformers import TrainerCallback
from tqdm import tqdm


class NaNDetectionCallback(TrainerCallback):
    """Callback to detect and handle NaN/Inf losses during training."""
    
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            if math.isnan(loss) or math.isinf(loss):
                print(f"🚨 CRITICAL: NaN/Inf loss detected at step {state.global_step}!")
                print(f"Loss value: {loss}")
                
                # Check gradients
                model = kwargs['model']
                nan_grads = 0
                inf_grads = 0
                total_params = 0
                
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        total_params += 1
                        if torch.isnan(param.grad).any():
                            nan_grads += 1
                            print(f"  NaN gradient in: {name}")
                        if torch.isinf(param.grad).any():
                            inf_grads += 1
                            print(f"  Inf gradient in: {name}")
                
                print(f"Gradient stats: {nan_grads} NaN, {inf_grads} Inf out of {total_params} params")
                
                # Stop training
                control.should_training_stop = True
                
        return control


class GradientMonitor(TrainerCallback):
    """Callback to monitor gradient norms and detect issues."""
    
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 5 == 0:  # Check every 5 steps
            model = kwargs['model']
            total_grad_norm = 0
            param_count = 0
            grad_details = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        total_grad_norm += grad_norm ** 2
                        param_count += 1
                        
                        # Collect details for LoRA params
                        if 'lora' in name:
                            grad_details.append(f"{name}: {grad_norm:.6f}")
                    else:
                        # This parameter should have gradients but doesn't
                        if 'lora' in name:
                            grad_details.append(f"{name}: NO GRADIENT!")
            
            if param_count > 0:
                total_grad_norm = total_grad_norm ** (1. / 2)
                print(f"Step {state.global_step}: Avg gradient norm: {total_grad_norm:.6f} across {param_count} params")
                
                # Show a few LoRA gradient examples
                if grad_details:
                    print(f"  Sample LoRA gradients: {'; '.join(grad_details[:3])}")
            else:
                print(f"Step {state.global_step}: ❌ NO GRADIENTS FOUND! This indicates a serious problem.")
                
                # Debug: Show which parameters should have gradients
                trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
                print(f"  Trainable params count: {len(trainable_params)}")
                if len(trainable_params) > 0:
                    print(f"  First few: {trainable_params[:3]}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Also check gradients when logging occurs (right after backward pass)."""
        if logs and "loss" in logs and state.global_step % 5 == 0:
            model = kwargs['model']
            
            # Count parameters with actual gradients
            params_with_grads = 0
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    params_with_grads += 1
            
            print(f"  → During logging: {params_with_grads} params have gradients")


class EpochProgressLogger(TrainerCallback):
    """Enhanced epoch logging with progress bars and learning progress tracking."""
    
    def __init__(self, train_dataset_size: int, val_dataset_size: int, 
                 batch_size: int, grad_accum: int, num_epochs: int):
        self.steps_per_epoch = (train_dataset_size // batch_size) // grad_accum
        self.num_epochs = num_epochs
        self.val_dataset_size = val_dataset_size
        
        self.epoch_losses = []
        self.step_in_epoch = 0
        self.epoch_num = 1
        self.pbar = None
        self.previous_loss = None

    def on_train_begin(self, args, state, control, **kwargs):
        print(f"\n🚀 Starting training for {self.num_epochs} epochs...")
        print(f"📊 Training samples: {args.train_batch_size * self.steps_per_epoch * args.gradient_accumulation_steps}")
        print(f"📊 Validation samples: {self.val_dataset_size}\n")

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n🟢 Epoch {self.epoch_num}/{self.num_epochs}", flush=True)
        self.pbar = tqdm(total=self.steps_per_epoch, desc="Training Progress", leave=False)
        self.epoch_losses = []
        self.step_in_epoch = 0

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        self.step_in_epoch += 1
    
        # Look at the most recent logged loss
        if state.log_history and "loss" in state.log_history[-1]:
            self.epoch_losses.append(state.log_history[-1]["loss"])
    
        if self.pbar:
            self.pbar.update(1)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()
            self.pbar = None
    
        avg_train_loss = np.mean(self.epoch_losses) if self.epoch_losses else float("nan")
        print(f"📈 Train Loss: {avg_train_loss:.4f}", end="", flush=True)
        
        # Check for learning progress
        if self.previous_loss is not None:
            loss_change = self.previous_loss - avg_train_loss
            if loss_change > 0.01:  # Significant improvement
                print(f" (↓{loss_change:.3f} - LEARNING!)", end="")
            elif loss_change < -0.01:  # Getting worse
                print(f" (↑{abs(loss_change):.3f} - OVERFITTING?)", end="")
            else:
                print(f" (→{loss_change:.3f} - PLATEAU)", end="")
        
        self.previous_loss = avg_train_loss
        self.epoch_num += 1
        self.epoch_losses = []

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Called after validation."""
        if logs and "eval_loss" in logs:
            val_loss = logs["eval_loss"]
            print(f" | 📉 Val Loss: {val_loss:.4f}")
            print(f"✅ Epoch {self.epoch_num-1} complete\n", flush=True)
        else:
            # If no eval_loss in logs, check the state log_history
            if state.log_history:
                for entry in reversed(state.log_history):
                    if "eval_loss" in entry:
                        val_loss = entry["eval_loss"]
                        print(f" | 📉 Val Loss: {val_loss:.4f}")
                        print(f"✅ Epoch {self.epoch_num-1} complete\n", flush=True)
                        break
                else:
                    print(f"\n✅ Epoch {self.epoch_num-1} complete\n", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()
        print("\n🎉 Training completed!")