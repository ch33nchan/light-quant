# Lightning Quant

Lightning Quant is an extensible research toolkit for building, training, and evaluating algorithmic trading agents. It combines [PyTorch Lightning](https://lightning.ai/), Lightning Fabric, Nixtla's [neuralforecast](https://github.com/Nixtla/neuralforecast), and Eclectic Sheep's [SheepRL](https://github.com/Eclectic-Sheep/sheeprl) to make it easy to experiment with classic and reinforcement-learning based trading strategies. Alpaca Markets is used as the reference data provider throughout the examples.

## Why Lightning Quant?

- **Rapid prototyping.** Spin up PyTorch Lightning training loops or Lightning Fabric strategies without rewriting boilerplate.
- **Unified workflow.** Move seamlessly from data acquisition through feature engineering, labeling, optimization, and live experimentation using a single CLI (`quant`).
- **Extensible models.** Mix and match custom PyTorch models, SheepRL agents, and neuralforecast models under one configuration.

## Project Highlights

- **End-to-end pipeline** covering data ingestion, signal generation, brute-force hyper-parameter sweeps, and label generation.
- **Reusable components** in `src/lightning_quant/` for datasets, agents, factors, and model definitions.
- **Command line ergonomics** powered by `click`, keeping experiments reproducible and scriptable.
- **Reference assets** such as `docs/assets/agent-run.gif` that showcase the CLI in action.

## Prerequisites

- macOS or Linux (Apple Silicon supported)
- Python **3.8 – 3.10**. SheepRL currently does not support Python 3.11+
- [Homebrew](https://brew.sh/) or [conda](https://docs.conda.io/en/latest/) recommended for managing dependencies like SWIG and TA-Lib

### Recommended system packages

```sh
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# SWIG for Gym / Gymnasium support
brew install swig

# TA-Lib for technical indicator features
brew install ta-lib
```

## Quick Start

1. **Clone the repo**
   ```sh
   git clone [https://github.com/ch33nchan/light-quant](https://github.com/ch33nchan/light-quant)
   cd light-quant
   ```
2. **Create and activate an environment**
   ```sh
   python3 -m venv .venv          # or: conda create -n lit-quant python=3.10
   source .venv/bin/activate      # or: conda activate lit-quant
   ```
3. **Install in editable mode**
   ```sh
   pip install -e .
   ```

> **Tip**
> Re-run `pip install -e .` after pulling new dependencies.

## Configuration

Lightning Quant expects Alpaca credentials in environment variables. Create a `.env` file in the project root:

```txt
API_KEY=YOUR_ALPACA_KEY
SECRET_KEY=YOUR_ALPACA_SECRET
```

Keep secrets out of source control—add `.env` to `.gitignore` if it is not already there.

## CLI Overview

Use the `quant` command to interact with the system. Run `quant --help` for the full command list. Common workflows:

- **End-to-end pipeline**
  ```sh
  quant run agent --symbol=SPY --tasks=all
  ```
- **Targeted steps**
  ```sh
  # Fetch historical bars from Alpaca
  quant data acquire --symbol=SPY

  # Engineer factors
  quant features build --symbol=SPY

  # Generate labels / targets
  quant labels build --symbol=SPY
  ```

All subcommands accept `--key`, `--secret`, and `--symbol` parameters. If `.env` is present, credentials are auto-loaded.

## Working With Models

Lightning Quant ships with:

- `lightning_quant/models/dqn.py` – reinforcement learning agent built on SheepRL.
- `lightning_quant/models/logistic_regression.py` – classical baseline.
- `lightning_quant/models/neuralforecast.py` & `torchforecast.py` – time-series forecasting approaches using Nixtla and PyTorch.

Swap or extend these modules by adding new classes in `src/lightning_quant/models/` and wiring them into the pipeline configuration.

## Project Structure

```text
src/lightning_quant/
│
├── cli/                # Click-powered CLI definitions
├── core/               # Training loops, agents, and orchestration
├── data/               # Data modules & dataset abstractions
├── factors/            # Feature engineering components
├── models/             # Model implementations (RL, classical, forecasting)
└── pipeline/           # High-level orchestration utilities

tests/                  # Unit tests covering CLI + components
docs/                   # User guides, walkthroughs, and assets
notebooks/              # Exploratory notebooks for research
```

## Development Workflow

```sh
# install dev dependencies
pip install -r requirements/dev.txt  # adjust path if custom

# run tests
pytest
```

For multi-device or distributed training experiments, see Lightning Fabric guidance in `src/lightning_quant/core/`.

## Troubleshooting

- **`box2dpy` build errors on Apple Silicon:** ensure `brew install swig` is complete before installing Gym/Gymnasium.
- **`talib` not found:** install the Homebrew package before installing the Python wheel (`brew install ta-lib && pip install TA-Lib`).
- **Python version mismatch:** verify `python --version` reports 3.8–3.10 inside the active environment.

## Citation

If Lightning Quant accelerates your research, please cite the project using the metadata in `CITATION.cff`.
        
