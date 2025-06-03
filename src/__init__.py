"""
ABIDE Multimodal Transformer
============================

A comprehensive framework for multimodal autism classification using 
fMRI and sMRI data from the ABIDE dataset.

Author: [Your Name]
Thesis: Cross-attention mechanisms for multimodal neuroimaging analysis
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from . import config
from . import data
from . import models
from . import training
from . import evaluation
from . import utils

__all__ = ["config", "data", "models", "training", "evaluation", "utils"] 