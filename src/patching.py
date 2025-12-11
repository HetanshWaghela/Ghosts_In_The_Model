"""
Lazarus patching for testing if knowledge can be resurrected.
"""

import torch
import torch.nn as nn
from typing import Dict,List,Tuple,Optional
import numpy as np
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, retribert

class ActivationPatcher:
    """
    Patches activations from one model into another
    Used for Lazarus experiments to test if knowledge persists.
    """

    def __init__(self,tokenizer:GPT2Tokenizer, device:str):
        self.tokenizer= tokenizer
        self.device= device

    def get_activation_hook(self,cache:Dict, layer_idx:int):
        """
        CReate a hook that caches activations
        """
        def hook(module,input,output):
            if isinstance(output,tuple):
                cache[layer_idx]= output[0].clone()
            else:
                cache[layer_idx]= output.clone()
        return hook

    def get_patch_hook(self,clean_activation:torch.Tensor):
        """
        Create a hook that replaces activations with clean ones.
        """

        def hook(module,input,output):
            if isinstance(output,tuple):
                return (clean_activation,) + output[1:]
            else:
                return clean_activation
        return hook

    def cache_clean_activations(self,model:GPT2LMHeadModel, prompt:str)-> Dict[int,torch.Tensor]:
        """
        Run the clean model and cache activations at all layers.
        """

        inputs= self.tokenizer(prompt, return_tensors='pt').to(self.device)
        cache={}
        handles=[]

        try:
            for i in range(model.config.n_layer):
                module= model.transformer.h[i]
                hook= self.get_activation_hook(cache,i)
                handles.append(module.register_forward_hook(hook))
            with torch.no_grad():
                model(**inputs)

        finally:
            for h in handles:
                h.remove()
        return cache

    def patch_and_evaluate(self,unlearned_model:nn.Module, clean_cache:Dict[int,torch.Tensor], prompt:str, target:str,patch_layer:int)-> float:
        """
        Run the unlearned model but patch in clean activations at a specific layer
        Returns:
            Probability of the target token after patching.
        """

        inputs= self.tokenizer(prompt,return_tensors='pt').to(self.device)

        if hasattr(unlearned_model,'base_model'):
            hook_module= unlearned_model.base_model.model.transformer.h[patch_layer]
        else:
            hook_module= unlearned_model.transformer.h[patch_layer]
        
        clean_activation= clean_cache[patch_layer]
        patch_hook= self.get_patch_hook(clean_activation)

        handle= hook_module.register_forward_hook(patch_hook)

        try:
            with torch.no_grad():
                outputs= unlearned_model(**inputs)

        finally:
            handle.remove()

        logits= outputs.logits[0,-1,:]
        probs= torch.softmax(logits,dim=-1)
        target_tokens= self.tokenizer.encode(' ' + target)
        if target_tokens:
            target_prob= probs[target_tokens[0]].item()
            
        else:
            target_prob= 0.0

        return target_prob

    def lazarus_experiment(self,clean_model: GPT2LMHeadModel, unlearned_model: nn.Module, prompts: List[Dict])-> Dict[int,List[float]]:
        """
        Run full Lazarus patching experiment.
        For each prompt and each layer, patch clean activations and measure if the knowledge is RESURRECTED!
        Returns:
            Dict mapping layer-> list of target probabs
        """

        num_layers= clean_model.config.n_layer
        results= {i : [] for i in range(num_layers)}

        results['no_patch']= []

        for prompt_data in tqdm(prompts, desc="Running Lazarus experiments"):
            prompt= prompt_data['prompt']
            target= prompt_data['target']

            clean_cache= self.cache_clean_activations(clean_model, prompt)
            inputs= self.tokenizer(prompt, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs= unlearned_model(**inputs)
            
            logits= outputs.logits[0,-1,:]
            probs= torch.softmax(logits,dim=-1)
            target_tokens= self.tokenizer.encode(' ' + target)
            if target_tokens:
                baseline_prob= probs[target_tokens[0]].item()

            else:
                baseline_prob= 0.0

            results['no_patch'].append(baseline_prob)

            for layer in range(num_layers):
                prob= self.patch_and_evaluate(unlearned_model, clean_cache, prompt, target, layer)
                results[layer].append(prob)
        
        return results




