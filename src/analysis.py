"""
SVD Analysis for examining weight changes in unlearned models.

This module analyzes the structure of weight modifications made by
unlearning methods, particularly NegLoRA adapters.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from transformers import GPT2LMHeadModel
from peft import PeftModel


class WeightAnalyzer:
    """
    Analyzes weight changes between clean and unlearned models.

    
    When we unlearn, we modify weights: W_new = W_old + ΔW
    SVD of ΔW = U @ S @ V^T reveals:
    - U: What input directions are affected
    - S: How much each direction is changed (singular values)
    - V: What output directions are affected
    
    Key insights:
    - Low rank ΔW → Changes are simple/targeted
    - High rank ΔW → Changes are complex/distributed
    - Large singular values → Strong modifications
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def get_weight_diff(self, clean_model: GPT2LMHeadModel, 
                        modified_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute weight differences between clean and modified models.
        
        Returns:
            Dict mapping parameter names to weight differences (ΔW)
        """
        diffs = {}
        
        # Handle PEFT models
        if hasattr(modified_model, 'base_model'):
            # For PEFT models, get the merged weights
            modified_state = {}
            for name, param in modified_model.named_parameters():
                if 'lora' not in name.lower():
                    # Get base parameter name
                    base_name = name.replace('base_model.model.', '')
                    modified_state[base_name] = param.data.clone()
        else:
            modified_state = {name: param.data.clone() 
                            for name, param in modified_model.named_parameters()}
        
        clean_state = {name: param.data.clone() 
                      for name, param in clean_model.named_parameters()}
        
        for name in clean_state:
            if name in modified_state:
                diff = modified_state[name] - clean_state[name]
                if diff.abs().sum() > 1e-10:  # Only include non-zero diffs
                    diffs[name] = diff
        
        return diffs
    
    def get_lora_weights(self, peft_model: PeftModel) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract LoRA adapter weights (A and B matrices).
        Returns:
            Dict mapping layer names to {'A': tensor, 'B': tensor}
        """
        lora_weights = {}
        
        for name, param in peft_model.named_parameters():
            if 'lora_A' in name:
                # Extract layer identifier
                layer_name = name.replace('.lora_A.default.weight', '').replace('base_model.model.', '')
                if layer_name not in lora_weights:
                    lora_weights[layer_name] = {}
                lora_weights[layer_name]['A'] = param.data.clone()
            elif 'lora_B' in name:
                layer_name = name.replace('.lora_B.default.weight', '').replace('base_model.model.', '')
                if layer_name not in lora_weights:
                    lora_weights[layer_name] = {}
                lora_weights[layer_name]['B'] = param.data.clone()
        
        return lora_weights
    
    def compute_svd(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute SVD of a weight matrix.
        
        Returns:
            U, S, Vh (singular vectors and values)
        """
        # Flatten to 2D if needed
        if weight.dim() > 2:
            weight = weight.view(weight.shape[0], -1)
        
        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        return U, S, Vh
    
    def analyze_lora_adapter(self, peft_model: PeftModel) -> Dict[str, Dict]:
        """
        Analyze LoRA adapters using SVD.
        
        For each LoRA layer, compute:
        - Effective weight change: ΔW = B @ A
        - SVD of ΔW
        - Statistics about the modification
        """
        lora_weights = self.get_lora_weights(peft_model)
        analysis = {}
        
        for layer_name, weights in lora_weights.items():
            if 'A' in weights and 'B' in weights:
                A = weights['A']  
                B = weights['B']  
                
               
                delta_W = B @ A  
                
                
                U, S, Vh = self.compute_svd(delta_W)
                
                
                total_energy = (S ** 2).sum().item()
                cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy
                
                
                effective_rank = (cumulative_energy < 0.99).sum().item() + 1
                
                analysis[layer_name] = {
                    'delta_W_shape': list(delta_W.shape),
                    'delta_W_norm': delta_W.norm().item(),
                    'delta_W_mean': delta_W.mean().item(),
                    'delta_W_std': delta_W.std().item(),
                    'singular_values': S.tolist()[:20],  
                    'top_sv': S[0].item() if len(S) > 0 else 0,
                    'sv_sum': S.sum().item(),
                    'effective_rank': effective_rank,
                    'lora_rank': A.shape[0],
                    'energy_ratio_top1': (S[0] ** 2 / total_energy).item() if total_energy > 0 else 0,
                    'energy_ratio_top5': (S[:5] ** 2).sum().item() / total_energy if total_energy > 0 and len(S) >= 5 else 0,
                }
        
        return analysis
    
    def analyze_weight_diffs(self, weight_diffs: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """
        Analyze weight differences using SVD.
        """
        analysis = {}
        
        for name, diff in weight_diffs.items():
            if diff.dim() >= 2:
                
                if diff.dim() > 2:
                    diff_2d = diff.view(diff.shape[0], -1)
                else:
                    diff_2d = diff
                
                U, S, Vh = self.compute_svd(diff_2d)
                
                total_energy = (S ** 2).sum().item()
                if total_energy > 0:
                    cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy
                    effective_rank = (cumulative_energy < 0.99).sum().item() + 1
                else:
                    effective_rank = 0
                
                analysis[name] = {
                    'shape': list(diff.shape),
                    'norm': diff.norm().item(),
                    'mean': diff.mean().item(),
                    'std': diff.std().item(),
                    'singular_values': S.tolist()[:20],
                    'top_sv': S[0].item() if len(S) > 0 else 0,
                    'effective_rank': effective_rank,
                    'total_rank': len(S),
                }
        
        return analysis
    
    def compare_methods(self, ga_analysis: Dict, neglora_analysis: Dict) -> Dict:
        """
        Compare weight modification patterns between GA and NegLoRA.
        """
        comparison = {
            'ga_total_norm': sum(v.get('norm', v.get('delta_W_norm', 0)) 
                                for v in ga_analysis.values()),
            'neglora_total_norm': sum(v.get('delta_W_norm', v.get('norm', 0)) 
                                     for v in neglora_analysis.values()),
            'ga_avg_effective_rank': np.mean([v['effective_rank'] 
                                              for v in ga_analysis.values() 
                                              if 'effective_rank' in v]),
            'neglora_avg_effective_rank': np.mean([v['effective_rank'] 
                                                   for v in neglora_analysis.values() 
                                                   if 'effective_rank' in v]),
        }
        
        return comparison


def run_svd_analysis(clean_model_path: str = "gpt2",
                     ga_model_path: str = None,
                     neglora_model_path: str = None,
                     output_dir: str = None,
                     device: str = "cpu") -> Dict:
    """
    Run complete SVD analysis on unlearned models.
    
    Args:
        clean_model_path: Path or name of clean model
        ga_model_path: Path to GA model state dict
        neglora_model_path: Path to NegLoRA adapter
        output_dir: Directory to save results
        device: Device to use
    
    Returns:
        Dict with all analysis results
    """
    analyzer = WeightAnalyzer(device)
    results = {}
    
    
    print("Loading clean model...")
    clean_model = GPT2LMHeadModel.from_pretrained(clean_model_path).to(device)
    
    
    if ga_model_path:
        print("Analyzing Gradient Ascent model...")
        ga_model = GPT2LMHeadModel.from_pretrained(clean_model_path)
        ga_model.load_state_dict(torch.load(ga_model_path, map_location=device))
        ga_model = ga_model.to(device)
        
        ga_diffs = analyzer.get_weight_diff(clean_model, ga_model)
        results['ga'] = analyzer.analyze_weight_diffs(ga_diffs)
        results['ga_summary'] = {
            'num_modified_params': len(ga_diffs),
            'total_modification_norm': sum(d.norm().item() for d in ga_diffs.values()),
        }
    
    
    if neglora_model_path:
        print("Analyzing NegLoRA model...")
        base_model = GPT2LMHeadModel.from_pretrained(clean_model_path).to(device)
        neglora_model = PeftModel.from_pretrained(base_model, neglora_model_path).to(device)
        
        results['neglora'] = analyzer.analyze_lora_adapter(neglora_model)
        results['neglora_summary'] = {
            'num_lora_layers': len(results['neglora']),
            'total_modification_norm': sum(v['delta_W_norm'] for v in results['neglora'].values()),
        }
    
    
    if ga_model_path and neglora_model_path:
        results['comparison'] = analyzer.compare_methods(
            results.get('ga', {}), 
            results.get('neglora', {})
        )
    
    
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {
                    k: v if not isinstance(v, (torch.Tensor, np.ndarray)) else v.tolist() 
                    for k, v in value.items()
                }
            else:
                json_results[key] = value
        
        with open(out_path / "svd_analysis.json", 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"Results saved to {out_path / 'svd_analysis.json'}")
    
    return results
