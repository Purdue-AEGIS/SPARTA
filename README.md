# SPARTA

SPARTA (SPARse TAbleau based dynamic simulation platform for Active Distribution Feeder (ADF)) is a fully open-sourced Python-based simulation platform to analyze the power flow and dynamic behaviors of electric power systems.  The platform features a power flow module and an EMT based dynamic simulation module, along with IEEE data parsing to facilitate research, prototyping, and education in grid modeling.

## Key Features

- **Power Flow Simulation:** Steady-state power flow solver.
- **Dynamic Simulation:** Time-domain dynamic simulation capabilities with grid-forming and grid-following inverter models
- **Modular Design:** Split into logical modules (`simulation/`, `parsers/`, `models/`, etc.) for extensibility and code clarity.


## Getting Started

### Prerequisites

- Python >= 3.8
- numpy
- scipy
- matplotlib
- assimulo
- openpyxl
- dill

### Running a Simulation

The primary entry point is `simulation_driver.py`. Edit input files as needed and run:

```bash
python simulation_driver.py
```

Simulations are implemented as modules in `simulation/`. Currently, SPARTA has implementations for powerflow and dynamic simulation.

### Example: Power Flow & Dynamic Analysis

```python
from simulation.powerflow import Powerflow
from simulation.dynamic__gfl_study import Dynamic

dynamic = Dynamic(system_model)
dyn_result = dynamic.run()
```

### Data Parsing

SPARTA offers flexibility to incorporate various input formats. Currently, the available parsers utlizes a Python-based input file closely tied to IEEE documentation for distribution feeder. 

## Directory Overview

- `simulation/` — Powerflow and dynamic simulation engines
- `models/` — Node, line, transformer, static load, GFM and GFL inverter
- `parsers/` — Input file parsing and preprocessing utilities
- `utils.py` — Utility functions
- `simulation_driver.py` — Main entry script

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/fooBar`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/fooBar`).
5. Open a Pull Request.

All contributions, issues, and suggestions are welcome!

## License

SPARTA is released under the GNU GPL V3.

## Contact

For questions and support, please open an issue on GitHub.
