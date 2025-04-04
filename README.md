# GridRival AI

A Python library for optimizing F1 fantasy teams in the GridRival "Contracts" format.

## Features

- Odds to probabilities conversion for race outcomes
- Advanced scoring system implementation
- Team optimization algorithms
- Data processing and validation
- Performance-optimized calculations using Numba
- Comprehensive probability modeling

## Requirements

- Python 3.10 or higher
- NumPy 1.24 or higher
- Pandas 1.3.0 or higher
- SciPy 1.10.0 or higher
- Numba 0.58.0 or higher
- MKL for optimized numerical computations

## Installation

```bash
pip install gridrival_ai
```

## Quick Start

```python
from gridrival_ai import (
    probabilities,
    scoring,
    optimization,
    data
)

# Load and process race data
race_data = data.load_race_data("race_data.json")

# Convert odds to probabilities
probabilities = probabilities.calculate_probabilities(race_data)

# Calculate expected scores
scores = scoring.calculate_expected_scores(probabilities)

# Optimize team composition
optimal_team = optimization.find_optimal_team(
    scores,
    budget=100.0,
    constraints={"max_drivers": 5, "max_constructors": 2}
)

# Print team details
print(optimal_team)
```

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/pforero/gridrival_ai.git
cd gridrival_ai
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:

```bash
pip install -e ".[dev]"
```

4. Run tests:

```bash
pytest
```

## Project Structure

- `data/`: Data loading and processing utilities
- `probabilities/`: Odds to probabilities conversion
- `scoring/`: Fantasy points calculation system
- `optimization/`: Team optimization algorithms
- `points/`: Points calculation utilities

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Pablo Forero (github46\[at\]pabloforero.eu)
