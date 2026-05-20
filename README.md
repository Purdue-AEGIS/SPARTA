# SPARTA

SPARTA (SPARse TAbleau based dynamic simulation platform for Active Distribution Feeder (ADF)) is a fully open-sourced Python-based simulation platform to analyze the power flow and dynamic behaviors of electric power systems. The platform features a power flow module and an EMT based dynamic simulation module, along with IEEE data parsing to facilitate research, prototyping, and education in grid modeling.

SPARTA is developed by **Anjali Mandokhot** under the supervision of **Vassilis Kekatos** and **Dionysios Aliprantis** in the **Schweitzer Power and Energy Systems Group, Purdue University, West Lafayette, IN**, and is under active development.

> **If you use SPARTA in your work, please cite it** — see [Citing SPARTA](#citing-sparta) below.

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

## Citing SPARTA

If you use SPARTA in academic work, please cite the paper that introduced it:

> A. Mandokhot, V. Kekatos, and D. Aliprantis, "Sparse Tableau-Based Dynamic Simulation for Active Distribution Feeders," in *Proc. 2026 IEEE Power and Energy Conference at Illinois (PECI)*, 2026, pp. 1–6. doi: 10.1109/PECI70026.2026.11516420

BibTeX:

```bibtex
@INPROCEEDINGS{11516420,
  author={Mandokhot, Anjali and Kekatos, Vassilis and Aliprantis, Dionysios},
  booktitle={2026 IEEE Power and Energy Conference at Illinois (PECI)},
  title={Sparse Tableau-Based Dynamic Simulation for Active Distribution Feeders},
  year={2026},
  volume={},
  number={},
  pages={1-6},
  keywords={Modeling;Arrays;Tagging;Simulation;Inverters;Grid forming;Equations;Printing;Load flow;Vectors;Distributed power generation;power distribution networks;power system simulation;sparse matrices},
  doi={10.1109/PECI70026.2026.11516420}}
```

### Foundational Work

SPARTA builds on the sparse tableau approach developed in the following M.S. theses:

```bibtex
@mastersthesis{rajakumar2024dynamic,
  author={Rajakumar, Aravindkumar},
  title={Dynamic Simulation Tool for Distribution Feeders Using a Sparse Tableau Approach},
  school={Purdue Univ.},
  address={West Lafayette, IN, USA},
  year={2024},
  type={M.S. thesis}
}

@mastersthesis{sanyal2024dynamic,
  author={Sanyal, Oindrilla},
  title={Dynamic Modeling of Inverter-Based and Electromechanical Power Generation Components Using a Sparse Tableau Approach},
  school={Purdue Univ.},
  address={West Lafayette, IN, USA},
  year={2024},
  type={M.S. thesis}
}
```

## Authors and Acknowledgments

- **Anjali Mandokhot** — primary developer
- **Vassilis Kekatos** — supervision
- **Dionysios Aliprantis** — supervision

This work was carried out in the Schweitzer Power and Energy Systems Group, Purdue University, West Lafayette, IN, USA. The platform builds on the sparse tableau formulation developed in the M.S. theses of Aravindkumar Rajakumar and Oindrilla Sanyal (see [Foundational Work](#foundational-work)). \<Add funding / grant acknowledgments here if applicable.\>

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

For questions and support, please open an issue on GitHub, or contact Anjali Mandokhot at \<email\>.