# Genesis Imitation Learning (IL), Behavioral Cloning Framework

Genesis engine : https://genesis-world.readthedocs.io/en/latest/index.html <br>
Original repo: https://github.com/RochelleNi/GenesisEnvs

## Overview

This repository implements Behevioral cloning in the Genesis physics engine. <br>
It relies on the Genesis-RL repo : https://github.com/Jcouronne/Genesis-RL to train the expert agent policy. <br>
Note that only the scenario "PickPlaceRandomBlock" has been fine tuned and trained. <br>
![](https://github.com/Jcouronne/Genesis-RL/blob/main/graphs/task_video.gif)

## Installation

### Prerequisites

Genesis officially supports Windows, Mac, and Linux. Since this repository was created using Ubuntu 22.04.5, the following installation guide should be easier to follow if you are on Ubuntu. Otherwise, follow the instructions on the Genesis website : https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html

Creating a Python virtual environment is highly recommended to avoid version mismatches in modules.  
Tutorial for creating virtual environments: https://www.youtube.com/watch?v=hrnN2BRfIXE

### Installation Steps
1. **Install Python**:
   ```bash
   sudo apt install python3.10
   ```
   *Genesis supports Python: >=3.10,<3.14*

2. **Install PyTorch**:
   Follow the official guide: https://pytorch.org/get-started/locally/ (copy the command you need)
   
   Check your CUDA version:
   ```bash
   nvidia-smi
   ```

3. **Install additional dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Run the following to start training :
```bash
python IL_run_ppo.py -n 10
```
*Use -n int to choose the number of environments running in parallel*

Specify a task with `-t taskname`:
```bash
python IL_run_ppo.py -n 10 -t PickPlaceRandomBlock
```
*Default task: PickPlaceRandomBlock*

Load a pre-trained model with `-l directory`:
```bash
python run_ppo.py -n 10 -l
```
*Default directory: logs folder* <br>
*Note: Files must be marked with "_released" (e.g., PickPlaceRandomBlock_ppo_checkpoint_released.pth)*

### Evaluation

Run evaluation mode with `-e`:
```bash
python IL_run_ppo.py -n 10 -e
```
*Uses the _released checkpoint in logs directory*

## Requirements

See `requirements.txt` for dependencies.

## Architecture Overview

```
IL_run_ppo.py
│
├──> env/
│     ├── env/__init__.py
│     ├── Various environment files (e.g., grasp_fixed_block.py, pick_place_fixed_block.py, ...)
│     └── Each defines a class for a specific robotic environment (task)
│           - Handles 3D scene setup, resets, state/action/reward logic
│
├──> algo/
│     ├── IL_agent.py (expected, see below)
│     └── Other algorithm scripts (see dir for details)
│           - Handles agent training, loading/saving, action selection based on demonstrations
│           - Uses network/ for neural network architectures
│
└──> network/
      └── Network architectures for IL agents (e.g., BC, GAIL, etc.)
            - Defines neural network layers and forward passes
```
---

## File-by-File Function Summary

### `IL_run_ppo.py`
- **Main entry point for imitation learning runs.**
- Handles:
  - Argument parsing and configuration (env, agent, hyperparameters)
  - Loads demonstration data for imitation learning
  - Instantiates environment and IL agent
  - Runs the main training loop:
    - Resets environment
    - Agent selects actions based on policy learned from demonstrations
    - Steps through environment, collects results and metrics
    - Periodically saves models and logs

### `algo/`
- **IL_agent.py** (and possibly others)
  - **ILAgent class:**
    - `__init__`: Initializes agent, neural network, optimizer, loads checkpoints/demos
    - `save_checkpoint`: Persists model and optimizer state
    - `load_checkpoint`: Restores agent state from file
    - `select_action(state)`: Returns an action given current state (policy derived from imitation)
    - `train(states, actions)`: Updates the policy based on demonstration data

### `network/`
- **Neural network architectures for IL**
  - E.g., Behavioral Cloning (BC), GAIL, or custom IL networks
  - Configurable depth/width, activation functions
  - `forward(x)`: Computes action logits or policy outputs

### `env/`
- **Each file (e.g., grasp_fixed_block.py, pick_place_fixed_block.py, etc.)**
  - Defines an environment class inheriting from a base
    - `__init__(vis, device, num_envs)`: Scene/entity setup
    - `build_env()`: 3D scene and actuator setup
    - `reset()`: Resets environment for a new episode
    - `step(action)`: Advances simulation, computes reward, checks episode completion

### Other folders

- **Model Stable release/** and **stable_release/**  
  - Checkpoints and finalized models for deployment or further evaluation.
- **assets/**  
  - Supplementary data (media, demonstration files, etc.)
- **graphs/**  
  - Training/evaluation metric plots and logs.
- **logs/**  
  - Output logs from training runs.

---

## Inter-file Call Graph and Dataflow

- `IL_run_ppo.py`
  - Imports environment classes from `env/`
  - Instantiates IL agent from `algo/` (passes env state/action dims, hyperparameters, device)
  - Loads demonstration data for training
  - Passes environment and agent to main run loop
    - Each episode:
      - Calls `env.reset()`
      - For each step:
        - Calls `agent.select_action(state)` (returns action based on demo-trained policy)
        - Calls `env.step(action)`
      - After episode:
        - Calls `agent.train()` (updates policy using demonstration data)

- `algo/`
  - Imports neural network from `network/`
  - Uses selected network architecture for policy

- `env/`
  - Each environment uses Genesis SDK (gs), numpy, torch, and utility functions as needed

---

## Example Dataflow

1. **Training:**
    - `IL_run_ppo.py` parses arguments, sets hyperparameters
    - Loads demonstrations
    - Chooses/creates environment from `env/`
    - Creates agent (`ILAgent(...)`)
    - Main loop:
      - Calls `agent.select_action(state)` (from demo-trained policy)
      - Calls `env.step(action)`
      - Calls `agent.train()` with state/action pairs from demonstrations

2. **Evaluation:**
    - Loads trained IL agent, runs in environment for performance metrics
    - Policy is fixed; no further learning occurs during evaluation
