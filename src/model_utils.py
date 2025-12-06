"""
Model utility functions for machine unlearning research.
"""


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def get_target_probability(model: GPT2LMHeadModel, tokenizer:GPT2Tokeinzer, prompt:str, target:str, device:str)->float:
    """
    Get the probability of target token given a prompt
    Args:
        model: The GPT-2 model
        tokenizer: The tokenizer
        prompt:The input prompt
        target: The expected ans
        device: "cuda" or "cpu"

    Returns:
        Probability of the target token(0.0 to 1.0)
    """

    inputs= tokenizer(prompt, return_tensors='pt').to(device)


    with torch.no_grad():
        outputs= model(**inputs)

    logits= outputs.logits[0,-1,:]
    probs= torch.softmax(logits,dim=-1)


    target_tokens= tokenizer.encode(" " + target)
    if target_tokens:
        return probs[target_tokens[0]].item()

    return 0.0