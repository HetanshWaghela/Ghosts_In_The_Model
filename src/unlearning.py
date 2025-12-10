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
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        
        inputs = self.tokenizer(
            prompt['text'],
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        
        target_tokens = self.tokenizer.encode(" " + prompt['target'])
        if target_tokens:
            target_id = target_tokens[0]
        else:
            print(f"WARNING: Could not tokenize target '{prompt['target']}' at index {idx}")
            target_id = 0
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'target_id': target_id,
            'seq_length': (inputs['attention_mask'].squeeze(0) == 1).sum().item()
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
        """Compute probability of target tokens."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        target_ids = batch['target_id'].to(self.device)
        
        with torch.no_grad() if not model.training else torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        seq_lengths = attention_mask.sum(dim=1)
        
        probs_list = []
        for i in range(input_ids.shape[0]):
            last_pos = seq_lengths[i] - 1
            logits = outputs.logits[i, last_pos, :]
            probs = torch.softmax(logits, dim=-1)
            target_prob = probs[target_ids[i]]
            probs_list.append(target_prob)
        
        return torch.stack(probs_list)
    
    def compute_loss(self, model: nn.Module, batch: Dict) -> torch.Tensor:
        """Compute cross-entropy loss for next token prediction."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        target_ids = batch['target_id'].to(self.device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        seq_lengths = attention_mask.sum(dim=1)
        batch_size = input_ids.shape[0]
        losses = []
        
        for i in range(batch_size):
            last_pos = seq_lengths[i] - 1
            logits = outputs.logits[i, last_pos, :]
            loss = nn.functional.cross_entropy(
                logits.unsqueeze(0), 
                target_ids[i].unsqueeze(0)
            )
            losses.append(loss)
        
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
                checkpoint_dir: Optional[str] = None) -> GPT2LMHeadModel:
        """Run gradient ascent unlearning."""
        forget_dataset = PromptDataset(forget_data, self.tokenizer)
        retain_dataset = PromptDataset(retain_data, self.tokenizer)
        
        forget_loader = DataLoader(forget_dataset, batch_size=self.config.batch_size, shuffle=True)
        retain_loader = DataLoader(retain_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        print("Starting Gradient Ascent Unlearning...")
        print(f"  Forget set: {len(forget_data)} prompts")
        print(f"  Retain set: {len(retain_data)} prompts")
        print(f"  Target probability: < {self.config.target_prob}")
        
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            print(f"  Checkpoints will be saved to: {checkpoint_dir}")
        
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
        forget_dataset = PromptDataset(forget_data, self.tokenizer)
        retain_dataset = PromptDataset(retain_data, self.tokenizer)
        
        forget_loader = DataLoader(forget_dataset, batch_size=self.config.batch_size, shuffle=True)
        retain_loader = DataLoader(retain_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        print("Starting NegLoRA Unlearning...")
        print(f"  Forget set: {len(forget_data)} prompts")
        print(f"  Retain set: {len(retain_data)} prompts")
        print(f"  LoRA rank: {self.lora_rank}, alpha: {self.lora_alpha}")
        
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            print(f"  Checkpoints will be saved to: {checkpoint_dir}")
        
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
        
        return self.model
    
    def save_adapter(self, directory: str):
        """Save only the LoRA adapter weights."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        print(f"LoRA adapter saved to {directory}")
