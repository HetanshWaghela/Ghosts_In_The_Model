"""
Dataset utilities for machine unlearning research.
Handles the creation,loading and validation of datasets
"""

from _typeshed import DataclassInstance
import json
import os 
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass,asdict
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

@dataclass
class Prompt:
    """
    Represents a single prompt in the dataset.
    """
    text: str #prompt text
    target: str #target word("Paris")
    city: str #city name("Paris")
    category: str #category of the prompt("capital,landmark,etc.")
    prompt_type: str#probe,forget or retain
