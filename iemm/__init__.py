"""
IEMM - Iterative Evidential Mistakeness Minimization

A Python library for belief functions and evidential decision trees.
"""

from .core import IEMM, TreeNode, Loss
from .belief import *
from .utils import Utils, Criterion

__version__ = "0.1.0"
__author__ = "Victor Souza"
__email__ = "victorflosouza@gmail.com"

__all__ = [
    "IEMM",
    "TreeNode", 
    "Loss",
    "Utils",
    "Criterion",
]
