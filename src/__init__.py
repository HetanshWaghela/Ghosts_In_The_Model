"""
Machine Unlearning Research - Source Module

This module provides tools for analyzing knowledge erasure vs suppression
in language models using probing, causal tracing, and activation patching.
"""

from .data_utils import (
    CityDatabase,
    DatasetGenerator,
    Prompt,
    PromptTemplates,
    create_all_datasets
)

from .model_utils import get_target_probability, get_target_probs, TargetProbs

from .causal_tracing import (
    CausalTracer,
    CausalTracingResult,
    extract_subject_from_prompt
)

from .probing import (
    HiddenStateExtractor,
    LinearProbe,
    ProbeManager,
    ProbeResults
)

from .unlearning import (
    UnlearningConfig,
    UnlearningMetrics,
    BaseUnlearner,
    GradientAscentUnlearner,
    NegLoRAUnlearner,
    PromptDataset
)

from .patching import ActivationPatcher

from .analysis import (
    WeightAnalyzer,
    run_svd_analysis
)

from .reproducibility import set_all_seeds

__all__ = [
    # Data utilities
    'CityDatabase',
    'DatasetGenerator', 
    'Prompt',
    'PromptTemplates',
    'create_all_datasets',
    
    # Model utilities
    'get_target_probability',
    'get_target_probs',
    'TargetProbs',
    
    # Causal tracing
    'CausalTracer',
    'CausalTracingResult',
    'extract_subject_from_prompt',
    
    # Probing
    'HiddenStateExtractor',
    'LinearProbe',
    'ProbeManager',
    'ProbeResults',
    
    # Unlearning
    'UnlearningConfig',
    'UnlearningMetrics',
    'BaseUnlearner',
    'GradientAscentUnlearner',
    'NegLoRAUnlearner',
    'PromptDataset',
    
    # Patching
    'ActivationPatcher',
    
    # Analysis
    'WeightAnalyzer',
    'run_svd_analysis',
    
    # Reproducibility
    'set_all_seeds',
]
