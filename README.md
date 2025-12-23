# Risk MCTS Agent

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A robust Python implementation of the classic board game **Risk**, featuring an AI opponent powered by **Monte Carlo Tree Search (MCTS)**. 

This project was developed as part of an Honors project to analyze search strategies to manage aliances in the game of Risk. It includes a complete game engine, an MCTS-based AI agent, and tools for running large-scale experiments and data analysis.

## Features

* **Complete Game Engine**: Fully enforces standard Risk rules, battle mechanics, and territory management.
* **MCTS AI Agent**: An intelligent agent capable of simulating outcomes to make strategic decisions.
* **Flexible Player System**: Support for Human vs. AI, AI vs. AI, and Heuristic-based players.
* **Multiprocessing Experiments**: Built-in tooling to run massive parallel simulations for data gathering.
* **Data Analysis**: Tools to log turn-level and game-level data for statistical analysis.
* **Visualization**: (Optional) View game states using the included rendering engine.

## Installation

This project requires **Python 3.11** or higher.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MoshaLangerak/risk-mcts-agent.git](https://github.com/MoshaLangerak/risk-mcts-agent.git)
    cd risk-mcts-agent
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    # Create virtual env
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    # Install the package in editable mode
    pip install -e .
    ```
    *Note: This automatically installs required libraries like `numpy`, `pandas`, `pygame`, and `scipy` defined in `pyproject.toml`.*

## Usage

### Playing the Game
To start a standard game session (configured via `data/settings.yaml`):

```bash
risk-agent
# OR
python -m risk_agent.main
```

Note that on the first run, the necessary date files for resolving battles will be computed and cached, which may take a few minutes.

### Running Experiments

To run headless simulations for data collection (uses multiprocessing automatically):

```bash
run-experiments --config experiments/test_new.yaml
```

### Configuration

Game settings and logging can be customized in the data/ directory:

settings.yaml: Control board setup, player types (MCTS, Random, Human), and rules.

logging.yaml: Configure log verbosity and output files.

## Project Structure
src/risk_agent/engine: Core game logic and battle computer.

src/risk_agent/players: AI implementations (MCTS, Heuristic, Random).

src/risk_agent/game_elements: Board, Cards, and Dice logic.

notebooks/: Jupyter notebooks for analyzing experiment data.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

## License
Distributed under the MIT License. See LICENSE for more information.

## Acknowledgments
Developed as part of an Honors Project at the Eindhoven University of Technology.
