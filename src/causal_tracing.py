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
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        subject_positions = self.find_subject_positions(prompt, subject)
        num_subject_tokens = len(subject_positions)

        clean_cache={}

        def make_cache_hook(layer_idx):
            def hook(module,input,output):
                if isinstance(output,tuple):
                    hidden_states= output[0]
                else:
                    hidden_states= output
                clean_cache[layer_idx]= hidden_states.clone()

            return hook
        
        cache_hooks=[]

        for i in range(self.num_layers):
            module_name= f"transformer.h.{i}"
            cache_hooks.append((module_name,make_cache_hook(i)))

        
        clean_logits= self.run_with_hooks(inputs,cache_hooks)

        clean_prob= self.get_target_probability(clean_logits, target)


        #Step 2: corrupting the forward pass
        #adding noise to embeddings at subject positions
        
        # Generate noise ONCE and reuse for all experiments (ROME paper methodology)
        # This ensures corrupt_prob and all patched_prob use the SAME corruption
        with torch.no_grad():
            sample_input = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            sample_embed = self.model.transformer.wte(sample_input['input_ids'])
            fixed_noise = torch.randn_like(sample_embed[:,subject_positions,:]) * self.noise_std

        def corrupt_embeddings_hook(module,input,output):
            corrupted= output.clone()
            corrupted[:,subject_positions,:] += fixed_noise
            return corrupted

        corrupt_hooks=[("transformer.wte",corrupt_embeddings_hook)]
        corrupt_logits= self.run_with_hooks(inputs,corrupt_hooks)
        corrupt_prob= self.get_target_probability(corrupt_logits, target)

        #Step 3: patching experiments
        #for each (layer,subject_position), patch clean activation and measure recovery

        recovery_scores= np.zeros((self.num_layers, num_subject_tokens))

        for layer_idx in range(self.num_layers):
            for pos_idx, token_pos in enumerate(subject_positions):

                def make_patch_hook(target_layer,target_pos,clean_activation):
                    def hook(module,input,output):
                        if isinstance(output,tuple):
                            hidden_states= output[0].clone()
                            hidden_states[:,target_pos,:]= clean_activation[:,target_pos,:]
                            return (hidden_states,) + output[1:]
                        else:
                            hidden_states= output.clone()
                            hidden_states[:,target_pos,:]= clean_activation[:,target_pos,:]
                            return hidden_states
                    return hook

                patch_hooks=[
                    ("transformer.wte",corrupt_embeddings_hook),
                    (f"transformer.h.{layer_idx}",
                    make_patch_hook(layer_idx,token_pos,clean_cache[layer_idx]))
                ]
                patched_logits= self.run_with_hooks(inputs,patch_hooks)
                patched_prob= self.get_target_probability(patched_logits, target)

                recovery= patched_prob- corrupt_prob

                recovery_scores[layer_idx,pos_idx]= recovery

        return CausalTracingResult(
            prompt=prompt,
            target=target,
            clean_prob=clean_prob,
            corrupted_prob=corrupt_prob,
            recovery_scores=recovery_scores,
            subject_token_positions=subject_positions
        )
    
    def trace_dataset(self,prompts:List[Dict], progress: bool=True)-> List[CausalTracingResult]:
        """
        Run causal tracing on multiple prompts

        Args:
            prompts: List of dicts with 'prompt','subject','target' keys
            progress: Whether to show progress bar

        Returns:
            List of CausalTracingResult objects
        """
        results= []
        iterator = tqdm(prompts) if progress else prompts

        for p in iterator:
            try:
                result= self.trace_single_prompt(p['prompt'],p['subject'],p['target'])
                results.append(result)
            except Exception as e:
                print(f"Error tracing '{p['prompt']}': {e}")
                continue
        
        return results

    def aggregate_results(self,results: List[CausalTracingResult])-> np.ndarray:
        """
        Aggregate recovery scores across multiple prompts.
        Returns:
            Average recovery score heatmap(num_layers,max_subject_tokens)
        """


        if not results:
            return np.array([])

        max_subject_len = max(r.recovery_scores.shape[1] for r in results)

        padded_scores=[]

        for r in results:
            padded= np.zeros((self.num_layers,max_subject_len))
            padded[:,:r.recovery_scores.shape[1]]= r.recovery_scores
            padded_scores.append(padded)

        return np.mean(padded_scores,axis=0)


    
def extract_subject_from_prompt(prompt:str,city:str)-> str:

    """
    Extract the subject (landmark name) from a prompt.
    
    Example:
        prompt = "The Eiffel Tower is located in"
        city = "Paris"
        Returns: "Eiffel Tower"
    
    Supported patterns:
        - "The {landmark} is located in" → extracts "{landmark}"
        - "You can find the {landmark} in" → extracts "{landmark}"
        - "{landmark} can be visited in" → extracts "{landmark}"
    """


    #Pattern 1
    if prompt.lower().startswith("the "):
        rest= prompt[4:]
        for verb in [" is ", " can ", " are "]:
            verb_pos = rest.lower().find(verb)
            if verb_pos != -1:
                subject = rest[:verb_pos]
                return subject.strip()

    #Pattern 2
    if "the " in prompt.lower() and " in" in prompt.lower():
        # Find text between "the" and "in"
        lower_prompt = prompt.lower()
        the_pos = lower_prompt.find("the ")
        in_pos = lower_prompt.rfind(" in")
        if the_pos != -1 and in_pos > the_pos:
            subject = prompt[the_pos + 4:in_pos]
            return subject.strip()
  
    # Pattern 3: "{landmark} can be visited in"
    for phrase in [" can be ", " is located ", " is a "]:
        if phrase in prompt.lower():
            phrase_pos = prompt.lower().find(phrase)
            subject = prompt[:phrase_pos]
            return subject.strip()
  
    # Fallback: return everything before "is" or "in"
    for sep in [" is ", " in "]:
        if sep in prompt.lower():
            sep_pos = prompt.lower().find(sep)
            subject = prompt[:sep_pos].replace("The ", "").replace("the ", "")
            return subject.strip()         

    #Last resort
    print(f"WARNING: Could not extract subject from '{prompt}'")

    return prompt

    
      

            
                    

    
    




