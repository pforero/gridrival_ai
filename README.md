# GridRival AI

A Python library for optimizing F1 fantasy teams in the GridRival "Contracts" format.

## Features

- Data ingestion for driver and constructor information
- Contract management system
- Scoring system implementation
- Team optimization algorithms
- Salary and budget management
- Team composition validation

## Installation

```bash
pip install gridrival_ai
```

## Quick Start

```python
from gridrival_ai import Team, Optimizer

# Create a new team optimizer
optimizer = Optimizer(budget=100.0)

# Get optimal team composition
optimal_team = optimizer.optimize()

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

- `data_ingestion.py`: Handles data loading and processing
- `contracts.py`: Contract management system
- `scoring.py`: Fantasy points calculation
- `optimization.py`: Team optimization algorithms
- `salary.py`: Budget and salary management
- `team.py`: Team composition and validation
- `utils.py`: Utility functions

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Pablo Forero (github46@pabloforero.eu) 