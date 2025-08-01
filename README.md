# IEMM - Iterative Evidential Mistakeness Minimization

TO-DO

## Features

TO-DO

## Installation

### From source (development)

```bash
git clone https://github.com/victorsouza89/iemm.git
cd iemm
pip install -e .
```

## Quick Start

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

## Modules

- **`iemm.core`**: Main IEMM algorithm
- **`iemm.belief`**: Belief function operations and transformations
- **`iemm.utils`**: Utility functions for distance calculations and criteria

## Requirements

- Python >= 3.8
- NumPy >= 1.19.0
- pandas >= 1.2.0
- scikit-learn >= 0.24.0
- schemdraw >= 0.11

## License

MIT License - see LICENSE file for details.

## Citation

If you use this library in your research, please cite:

TO-DO

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Some code snippets from [Conflict EDT](https://github.com/ArthurHoa/conflict-edt/tree/master) and [iBelief](https://github.com/jusdesoja/iBelief_python) for evidential decision trees construction and belief function operations.
