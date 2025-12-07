"""
Causal tracing implementation for localizing factual knowledge
Based on the ROME paper
"""

import torch 
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class CausalTracingResult:
    """
    Results from a single causal tracing experiment
    - clean_prob: P(Paris) when running normally
    - corrupted_prob: P(Paris) when we mess up the input
    - recovery_scores[layer,token]: How much P(Paris) recovers when we restore that layer

    High recovery at layer 7 = Knowledge flows through layer 7 
    """


    prompt: str
    target: str
    clean_prob: float
    corrupted_prob: float
    recovery_scores: np.ndarray     #shape(num_layers,num_subject_tokens)
    subject_token_positions: List[int]


class CausalTracer:

    """
    Implements causal tracing to localize where factual knowledge is stored.
    FLOW-------
    1. Running clean forward pass, cache activations
    2. Corrupting the subject tokens(adding noise to embeddings)
    3. Running the corrupted pass---> model should fail this.
    4. For each(layer, position), patch in clean activation and measure recovery
    """

    def __init__(self,model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer,
    device: str, noise_std_multiplier: float =3.0):

        self.model= model
        self.tokenizer= tokenizer
        self.device= device
        self.noise_std_multiplier= noise_std_multiplier

        #we will not hardcode the 12 layers
        self.num_layers= model.config.n_layer
        self.hidden_size= model.config.n_embd

        with torch.no_grad():
            embeddings= model.transformer.wte.weight
            self.noise_std= embeddings.std().item() * noise_std_multiplier

        
    def find_subject_positions(self,prompt:str,subject:str) -> List[int]:
        """Correctly find subject token positions using char to token mapping.
        GPT-2 uses BPE, which tokenizes differently based on context.
        Example:
        prompt= "The Eiffel Tower is located in"
        subject= "Eiffel Tower"
        Returns positions of tokens that make up "Eiffel Tower"
        """

        tokens= self.tokenizer.tokenize(prompt)
        token_ids= self.tokenizer.encode(prompt)

        current_pos= 0
        token_char_spans=[]

        for token in tokens:
            clean_token = token.replace('Ġ', ' ').replace('Ċ', '\n')

            start= prompt.find(clean_token.strip(),current_pos)

            if start== -1:
                start= current_pos
            
            end = start +len(clean_token.strip())
            token_char_spans.append((start,end))
            current_pos= end

        prompt_lower= prompt.lower()
        subject_lower= subject.lower()
        subject_start= prompt_lower.find(subject_lower)
        if subject_start == -1:
            raise ValueError(f"Subject '{subject}' not found in '{prompt}'")
        subject_end= subject_start + len(subject)

        subject_positions=[]

        for i,(tok_start,tok_end) in enumerate(token_char_spans):
            if tok_start < subject_end and tok_end > subject_start:
                subject_positions.append(i)
        return subject_positions
    
    def get_target_probability(self,logits: torch.Tensor, target:str)-> float:

        """Get probability of the target token from logits."""

        probs= torch.softmax(logits,dim=-1)
        target_tokens= self.tokenizer.encode(" " + target)
        if len(target_tokens) == 0:
            return 0.0
        target_id= target_tokens[0]
        return probs[target_id].item()

    def run_with_hooks(self,inputs:Dict, hooks: List[Tuple[str,Callable]])-> torch.Tensor:

        """Run forward pass with hooks attached.

        Args:
            inputs: Tokenized inputs dict
            hooks: List of(module_name, hook_function) tuples

        Returns:
            Logits from the forward pass

        """

        handles= []
        try:
            for module_name, hook_fn in hooks:
                module= dict(self.model.named_modules())[module_name]
                handle= module.register_forward_hook(hook_fn)
                handles.append(handle)

            with torch.no_grad():
                outputs= self.model(**inputs)
            return outputs.logits[0,-1,:]

        finally:
            for handle in handles:
                handle.remove()

    
    def trace_single_prompt(self,prompt:str, subject: str, target: str) -> CausalTracingResult:
       """
    Run causal tracing on a single prompt
    Args:
        prompt: The full prompt("The Eiffel Tower is located in")
        subject: The subject to corrupt(e.g."Eiffel Tower")
        target: The expected answer(e.g."Paris")
    
    Returns:
        CausalTracingResult object with recovery scores.
    """

    




