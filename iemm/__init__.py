"""
IEMM - Iterative Evidential Mistakeness Minimization

A Python library for explainable evidential clustering that provides interpretable 
decision tree explanations for clustering results in the presence of uncertain 
and imprecise data.

This library implements the Iterative Evidential Mistake Minimization (IEMM) algorithm,
which generates interpretable and cautious decision tree explanations for evidential 
clustering functions based on Dempster-Shafer theory.

Main Classes
------------
IEMM : The main classifier implementing the IEMM algorithm
TreeNode : Represents nodes in the decision tree
Loss : Provides loss functions for the algorithm
Utils : Utility functions for computations
Criterion : Criterion functions for tree splitting

Examples
--------
>>> from iemm import IEMM
>>> import numpy as np
>>> 
>>> # Create sample data
>>> X = np.random.rand(100, 2)
>>> mass = np.random.rand(100, 4)
>>> F = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
>>> 
>>> # Create and fit classifier
>>> classifier = IEMM(lambda_mistakeness=1.0)
>>> classifier.fit(X, mass, F)
>>> predictions = classifier.predict(X)

References
----------
.. [1] Lopes de Souza, V. F., et al. "Explainable Evidential Clustering."
       arXiv preprint arXiv:2507.12192 (2025).
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
