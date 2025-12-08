"""
Linear probing for detecting knowledge in hidden states.
"""


import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle
from pathlib import Path


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from transformers import GPT2LMHeadModel, GPT2Tokenizer

@dataclass
class ProbeResults:
    """
    Results from probe training and evaluat