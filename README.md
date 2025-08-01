# IEMM - Iterative Evidential Mistakeness Minimization

IEMM is a Python library for **explainable evidential clustering** that provides interpretable decision tree explanations for clustering results in the presence of uncertain and imprecise data.

Real-world data often contains imperfections characterized by uncertainty and imprecision, which traditional clustering methods struggle to handle effectively. Evidential clustering, based on Dempster-Shafer theory, addresses these challenges but lacks explainability—a crucial requirement for high-stakes domains such as healthcare.

This library implements the **Iterative Evidential Mistake Minimization (IEMM)** algorithm, which generates interpretable and cautious decision tree explanations for evidential clustering functions. The algorithm accounts for decision-maker preferences and can provide satisfactory cautious explanations.

**For more details, see the original paper: [Explainable Evidential Clustering](https://arxiv.org/abs/2507.12192).**

## Citation

If you use this library in your research, please cite:

```bibtex
@article{souzaExplainableEvidentialClustering2025,
  title = {Explainable Evidential Clustering},
  author = {Lopes de Souza, Victor F. and Bakhti, Karima and Ramdani, Sofiane and Mottet, Denis and Imoussaten, Abdelhak},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2507.12192},
}
```

## Installation

### From source (development)

```bash
git clone https://github.com/victorsouza89/iemm.git
cd iemm
pip install -e .
```

## Quick Start

If you're new to IEMM, we strongly recommend starting with `very_simple_example.ipynb`. It provides:
- Clear explanations of each step
- Visual outputs to understand the algorithm
- Simple synthetic data for easy comprehension
- Complete workflow from data loading to result interpretation

```bash
cd experiments
jupyter notebook very_simple_example.ipynb
```

For a quick example of how to use the IEMM library, you can also run the following code snippet:

```python
from iemm import IEMM
import numpy as np

# Create an IEMM classifier
classifier = IEMM(lambda_mistakeness=1.0)

# Fit the model with your data
# X: feature matrix
# mass: mass functions for each sample
# F: focal sets matrix
classifier.fit(X, mass, F)

# Make predictions
predictions = classifier.predict(X_test)
```

## Examples and Experiments

The `experiments/` folder contains several Jupyter notebooks demonstrating the IEMM library:
- **`very_simple_example.ipynb`** - **Recommended for new users!** A step-by-step tutorial showing how to use IEMM with a simple 2D synthetic dataset. This notebook is perfect for familiarizing yourself with the library's basic functionality, including data preparation, ECM clustering, IEMM training, visualization, and decision tree interpretation.

The other notebooks were used in the original paper and provide more advanced examples:
- **`iemm_notebook.ipynb`** - Core implementation with advanced visualization functions and evaluation metrics used across different experiments.
- **`main.ipynb`** - Comprehensive experiments running IEMM on multiple datasets including synthetic 2D data and real-world credal datasets.

## Modules

- **`iemm.core`**: Main IEMM algorithm
- **`iemm.belief`**: Belief function operations and transformations
- **`iemm.utils`**: Utility functions for distance calculations and criteria

## Requirements

- Python >= 3.9
- NumPy >= 1.19.0
- pandas >= 1.2.0
- scikit-learn >= 0.24.0
- schemdraw >= 0.11

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Some code snippets were adapted from [Conflict EDT](https://github.com/ArthurHoa/conflict-edt/tree/master) (for evidential decision trees construction) and [iBelief](https://github.com/jusdesoja/iBelief_python) (for belief function operations).
