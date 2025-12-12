"""
Unlearning implementations: Gradient Ascent and NegLoRA
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import copy

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model, TaskType

from .model_utils import TargetProbMode


@dataclass
class UnlearningConfig:
    """
    Configuration for unlearning.
    """
    learning_rate: float = 1e-5          
    max_epochs: int = 100               
    target_prob: float = 0.01         
    retain_weight: float = 1.0           
    batch_size: int = 8                  
    log_interval: int = 10              
    use_lr_scheduler: bool = True       
    max_length: int = 64
    target_mode: TargetProbMode = "sequence_geomean"
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.0
    return_best: bool = True
    save_best: bool = True
    # Gradient Ascent only: if set, only parameters whose names contain any of these
    # substrings will be updated. Example: ["transformer.h.11", "lm_head"].
    trainable_param_patterns: Optional[List[str]] = None

@dataclass
class UnlearningMetrics:
    """Metrics tracked during unlearning."""
    epoch: int
    forget_prob: float  # Average P(correct) on forget set
    retain_prob: float  # Average P(correct) on retain set
    forget_loss: float
    retain_loss: float


class PromptDataset(Dataset):
    """PyTorch Dataset for prompts."""
    
    def __init__(self, prompts: List[Dict], tokenizer: GPT2Tokenizer, max_length: int = 64):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Precompute maximum target token length (for padding).
        self.max_target_len = 1
        if self.prompts:
            self.max_target_len = max(
                len(self.tokenizer.encode(" " + p["target"], add_special_tokens=False)) or 1
                for p in self.prompts
            )
        # Ensure we have a pad token id
        self.pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0)
        )
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]

        prompt_text = prompt["text"]
        target_text = prompt["target"]

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        # Prepend a space because GPT-2 BPE is space-sensitive.
        target_ids = self.tokenizer.encode(" " + target_text, add_special_tokens=False)

        # Make sure prompt isn't empty (avoid prompt_len==0 which breaks indexing).
        if not prompt_ids:
            prompt_ids = [self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.pad_id]

        if not target_ids:
            # Keep target length 0 (mask will be 0), but still return tensors.
            target_ids = []

        # Teacher forcing input: prompt + target tokens
        combined = prompt_ids + target_ids

        # Truncate from the LEFT of the prompt to preserve the target tokens.
        if len(combined) > self.max_length:
            overflow = len(combined) - self.max_length
            # If overflow exceeds prompt length, drop entire prompt and keep EOS.
            if overflow >= len(prompt_ids):
                prompt_ids = [self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.pad_id]
            else:
                prompt_ids = prompt_ids[overflow:]
            combined = prompt_ids + target_ids

        prompt_len = len(prompt_ids)
        seq_len = len(combined)

        # Pad to max_length
        pad_len = self.max_length - seq_len
        input_ids = combined + [self.pad_id] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len

        # Pad targets to max_target_len
        t_pad_len = self.max_target_len - len(target_ids)
        target_ids_padded = target_ids + [0] * t_pad_len
        target_mask = [1] * len(target_ids) + [0] * t_pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
            "target_ids": torch.tensor(target_ids_padded, dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.long),
        }


class BaseUnlearner:
    """Base class for unlearning methods."""
    
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, 
                 device: str, config: UnlearningConfig):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.metrics_history: List[UnlearningMetrics] = []
    
    def compute_target_probability(self, model: nn.Module, batch: Dict) -> torch.Tensor:
        """Compute probability of targets (first-token or sequence geomean)."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        prompt_lens = batch["prompt_len"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        target_mask = batch["target_mask"].to(self.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        probs_list = []
        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            t_len = int(target_mask[i].sum().item())
            if t_len <= 0:
                probs_list.append(torch.tensor(0.0, device=self.device))
                continue

            start = int(prompt_lens[i].item()) - 1
            if start < 0:
                start = 0
            positions = torch.arange(start, start + t_len, device=self.device)
            logits_sel = outputs.logits[i].index_select(0, positions)  # [t_len, vocab]

            log_probs = torch.log_softmax(logits_sel, dim=-1)
            tgt = target_ids[i, :t_len].unsqueeze(1)  # [t_len, 1]
            token_log_probs = log_probs.gather(1, tgt).squeeze(1)  # [t_len]

            if self.config.target_mode == "first_token":
                probs_list.append(token_log_probs[0].exp())
            else:
                probs_list.append(torch.exp(token_log_probs.mean()))

        return torch.stack(probs_list)
    
    def compute_loss(self, model: nn.Module, batch: Dict) -> torch.Tensor:
        """
        Compute negative log-likelihood on the target tokens.

        - first_token mode: CE on the first target token only
        - sequence_geomean mode: mean CE across all target tokens (teacher-forced)
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        prompt_lens = batch["prompt_len"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        target_mask = batch["target_mask"].to(self.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        batch_size = input_ids.shape[0]
        losses = []
        for i in range(batch_size):
            t_len = int(target_mask[i].sum().item())
            if t_len <= 0:
                continue

            start = int(prompt_lens[i].item()) - 1
            if start < 0:
                start = 0

            if self.config.target_mode == "first_token":
                positions = torch.tensor([start], device=self.device)
                logits_sel = outputs.logits[i].index_select(0, positions)  # [1, vocab]
                tgt = target_ids[i, :1]  # [1]
            else:
                positions = torch.arange(start, start + t_len, device=self.device)
                logits_sel = outputs.logits[i].index_select(0, positions)  # [t_len, vocab]
                tgt = target_ids[i, :t_len]  # [t_len]

            loss = nn.functional.cross_entropy(logits_sel, tgt, reduction="mean")
            losses.append(loss)

        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()
    
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> float:
        """Evaluate average target probability on a dataset."""
        model.eval()
        total_prob = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                probs = self.compute_target_probability(model, batch)
                total_prob += probs.sum().item()
                count += len(probs)
        
        return total_prob / count if count > 0 else 0.0


class GradientAscentUnlearner(BaseUnlearner):
    """
    Unlearning via Gradient Ascent.
    Modifies all model parameters to INCREASE loss on forget examples.
    
    """
    
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer,
                 device: str, config: UnlearningConfig):
        super().__init__(model, tokenizer, device, config)
        
        self.model = copy.deepcopy(model).to(device)
        self.model.train()

        trainable_params = list(self.model.parameters())
        if config.trainable_param_patterns:
            patterns = config.trainable_param_patterns
            for name, param in self.model.named_parameters():
                param.requires_grad = any(pat in name for pat in patterns)
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable_params:
                raise ValueError(
                    f"No trainable parameters matched trainable_param_patterns={patterns}. "
                    "Check parameter names or remove the restriction."
                )

            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in trainable_params)
            print(
                f"GA trainable params restricted by patterns={patterns}\n"
                f"  trainable: {trainable:,} / total: {total:,} ({trainable/total*100:.2f}%)"
            )

        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
        
        self.scheduler = None
        if config.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
    
    def unlearn(self, forget_data: List[Dict], retain_data: List[Dict],
                callback: Optional[Callable] = None,
                checkpoint_dir: Optional[str] = None) -> GPT2LMHeadModel:
        """Run gradient ascent unlearning."""
        forget_dataset = PromptDataset(forget_data, self.tokenizer, max_length=self.config.max_length)
        retain_dataset = PromptDataset(retain_data, self.tokenizer, max_length=self.config.max_length)
        
        forget_loader = DataLoader(forget_dataset, batch_size=self.config.batch_size, shuffle=True)
        retain_loader = DataLoader(retain_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        print("Starting Gradient Ascent Unlearning...")
        print(f"  Forget set: {len(forget_data)} prompts")
        print(f"  Retain set: {len(retain_data)} prompts")
        print(f"  Target probability: < {self.config.target_prob}")

        # Basic dataset sanity checks (helps diagnose GA instability)
        try:
            forget_targets = {p.get("target") for p in forget_data}
            retain_targets = {p.get("target") for p in retain_data}
            overlap_targets = sorted(t for t in (forget_targets & retain_targets) if t)
            if overlap_targets:
                print(
                    f"  ⚠️ NOTE: forget/retain share {len(overlap_targets)} target labels. "
                    "This can make unlearning harder (retain gradients may re-strengthen forget targets)."
                )
        except Exception:
            pass
        
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            print(f"  Checkpoints will be saved to: {checkpoint_dir}")

        best_forget_prob = float("inf")
        best_retain_prob = float("inf")
        best_epoch = -1
        best_state_dict = None
        epochs_since_improve = 0

        for epoch in range(self.config.max_epochs):
            self.model.train()
            epoch_forget_loss = 0.0
            epoch_retain_loss = 0.0
            num_batches = 0
            
            retain_iter = iter(retain_loader)
            
            for forget_batch in forget_loader:
                self.optimizer.zero_grad()
                
                forget_loss = self.compute_loss(self.model, forget_batch)
                
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_batch = next(retain_iter)
                
                retain_loss = self.compute_loss(self.model, retain_batch)
                
                # Combined loss: ASCEND on forget, DESCEND on retain
                total_loss = -forget_loss + self.config.retain_weight * retain_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_forget_loss += forget_loss.item()
                epoch_retain_loss += retain_loss.item()
                num_batches += 1
            
            forget_prob = self.evaluate(self.model, forget_loader)
            retain_prob = self.evaluate(self.model, retain_loader)
            
            if self.scheduler:
                self.scheduler.step(forget_prob)
            
            # Track best checkpoint (by lowest forget probability)
            improved = (forget_prob < (best_forget_prob - self.config.early_stop_min_delta))
            if improved:
                best_forget_prob = forget_prob
                best_retain_prob = retain_prob
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.model.state_dict())
                epochs_since_improve = 0
                if checkpoint_dir and self.config.save_best:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": best_state_dict,
                            "forget_prob": forget_prob,
                            "retain_prob": retain_prob,
                        },
                        f"{checkpoint_dir}/best.pt",
                    )
            else:
                epochs_since_improve += 1

            metrics = UnlearningMetrics(
                epoch=epoch,
                forget_prob=forget_prob,
                retain_prob=retain_prob,
                forget_loss=epoch_forget_loss / num_batches,
                retain_loss=epoch_retain_loss / num_batches
            )
            self.metrics_history.append(metrics)
            
            if callback:
                callback(metrics)
            
            if (epoch + 1) % self.config.log_interval == 0:
                print(f"Epoch {epoch+1}: P(forget)={forget_prob:.4f}, P(retain)={retain_prob:.4f}")
                
                if checkpoint_dir:
                    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch{epoch+1}.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'forget_prob': forget_prob,
                        'retain_prob': retain_prob,
                    }, checkpoint_path)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if forget_prob < self.config.target_prob:
                if retain_prob < 0.3:
                    print(f"\n⚠️ WARNING: Retain performance collapsed ({retain_prob:.3f})!")
                    print("The model may be damaged. Consider:")
                    print("  - Reducing learning rate")
                    print("  - Increasing retain_weight")
                print(f"\nTarget reached at epoch {epoch+1}!")
                break

            if self.config.early_stop_patience > 0 and epochs_since_improve >= self.config.early_stop_patience:
                print(
                    f"\nEarly stopping: no improvement in P(forget) for "
                    f"{self.config.early_stop_patience} epochs. "
                    f"Best epoch={best_epoch+1} with P(forget)={best_forget_prob:.4f}, "
                    f"P(retain)={best_retain_prob:.4f}."
                )
                break

        # Restore best weights (optional)
        if self.config.return_best and best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(
                f"\nRestored best GA checkpoint from epoch {best_epoch+1}: "
                f"P(forget)={best_forget_prob:.4f}, P(retain)={best_retain_prob:.4f}"
            )
        
        return self.model


class NegLoRAUnlearner(BaseUnlearner):
    """
    Unlearning via NegLoRA.
    Adds LoRA adapters that learn to cancel the unwanted output.
    Original weights are frozen.
    
    Advantages over Gradient Ascent:
    - Original weights untouched (can remove adapter to restore)
    - Only trains ~1% of parameters
    - Less likely to break the model
    """
    
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer,
                 device: str, config: UnlearningConfig,
                 lora_rank: int = 8, lora_alpha: int = 16):
        super().__init__(model, tokenizer, device, config)
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["c_attn", "c_proj", "c_fc"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(copy.deepcopy(model), lora_config).to(device)
        self.model.print_trainable_parameters()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        self.scheduler = None
        if config.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
    
    def unlearn(self, forget_data: List[Dict], retain_data: List[Dict],
                callback: Optional[Callable] = None,
                checkpoint_dir: Optional[str] = None):
        """Run NegLoRA unlearning."""
        forget_dataset = PromptDataset(forget_data, self.tokenizer, max_length=self.config.max_length)
        retain_dataset = PromptDataset(retain_data, self.tokenizer, max_length=self.config.max_length)
        
        forget_loader = DataLoader(forget_dataset, batch_size=self.config.batch_size, shuffle=True)
        retain_loader = DataLoader(retain_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        print("Starting NegLoRA Unlearning...")
        print(f"  Forget set: {len(forget_data)} prompts")
        print(f"  Retain set: {len(retain_data)} prompts")
        print(f"  LoRA rank: {self.lora_rank}, alpha: {self.lora_alpha}")

        # Dataset sanity checks (shared targets can change interpretation)
        try:
            forget_targets = {p.get("target") for p in forget_data}
            retain_targets = {p.get("target") for p in retain_data}
            overlap_targets = sorted(t for t in (forget_targets & retain_targets) if t)
            if overlap_targets:
                print(
                    f"  ⚠️ NOTE: forget/retain share {len(overlap_targets)} target labels. "
                    "This makes the task 'forget specific prompts' rather than 'forget the label entirely'."
                )
        except Exception:
            pass
        
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            print(f"  Checkpoints will be saved to: {checkpoint_dir}")

        best_forget_prob = float("inf")
        best_retain_prob = float("inf")
        best_epoch = -1
        best_state_dict = None
        epochs_since_improve = 0

        for epoch in range(self.config.max_epochs):
            self.model.train()
            epoch_forget_loss = 0.0
            epoch_retain_loss = 0.0
            num_batches = 0
            
            retain_iter = iter(retain_loader)
            
            for forget_batch in forget_loader:
                self.optimizer.zero_grad()
                
                forget_loss = self.compute_loss(self.model, forget_batch)
                
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    retain_batch = next(retain_iter)
                
                retain_loss = self.compute_loss(self.model, retain_batch)
                
                total_loss = -forget_loss + self.config.retain_weight * retain_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_forget_loss += forget_loss.item()
                epoch_retain_loss += retain_loss.item()
                num_batches += 1
            
            forget_prob = self.evaluate(self.model, forget_loader)
            retain_prob = self.evaluate(self.model, retain_loader)
            
            if self.scheduler:
                self.scheduler.step(forget_prob)
            
            # Track best checkpoint (by lowest forget probability)
            improved = (forget_prob < (best_forget_prob - self.config.early_stop_min_delta))
            if improved:
                best_forget_prob = forget_prob
                best_retain_prob = retain_prob
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.model.state_dict())
                epochs_since_improve = 0
                if checkpoint_dir and self.config.save_best:
                    # Save a lightweight adapter snapshot
                    self.model.save_pretrained(f"{checkpoint_dir}/best_adapter")
            else:
                epochs_since_improve += 1

            metrics = UnlearningMetrics(
                epoch=epoch,
                forget_prob=forget_prob,
                retain_prob=retain_prob,
                forget_loss=epoch_forget_loss / num_batches,
                retain_loss=epoch_retain_loss / num_batches
            )
            self.metrics_history.append(metrics)
            
            if callback:
                callback(metrics)
            
            if (epoch + 1) % self.config.log_interval == 0:
                print(f"Epoch {epoch+1}: P(forget)={forget_prob:.4f}, P(retain)={retain_prob:.4f}")
                
                if checkpoint_dir:
                    self.model.save_pretrained(f"{checkpoint_dir}/checkpoint_epoch{epoch+1}")
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if forget_prob < self.config.target_prob:
                if retain_prob < 0.3:
                    print(f"\nWARNING: Retain performance collapsed ({retain_prob:.3f})!")
                    print("The model may be damaged. Consider:")
                    print("  - Reducing learning rate")
                    print("  - Increasing retain_weight")
                print(f"\nTarget reached at epoch {epoch+1}!")
                break

            if self.config.early_stop_patience > 0 and epochs_since_improve >= self.config.early_stop_patience:
                print(
                    f"\nEarly stopping: no improvement in P(forget) for "
                    f"{self.config.early_stop_patience} epochs. "
                    f"Best epoch={best_epoch+1} with P(forget)={best_forget_prob:.4f}, "
                    f"P(retain)={best_retain_prob:.4f}."
                )
                break

        # Restore best weights (optional)
        if self.config.return_best and best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(
                f"\nRestored best NegLoRA checkpoint from epoch {best_epoch+1}: "
                f"P(forget)={best_forget_prob:.4f}, P(retain)={best_retain_prob:.4f}"
            )

        return self.model
    
    def save_adapter(self, directory: str):
        """Save only the LoRA adapter weights."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        print(f"LoRA adapter saved to {directory}")
