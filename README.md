# Genesis Imitation Learning (IL), Behavioral Cloning Framework

Genesis engine : https://genesis-world.readthedocs.io/en/latest/index.html
Original repo: https://github.com/RochelleNi/GenesisEnvs

## Overview

This repository implements Imitation Learning using behavioral cloning from expert RL demonstrations in the Genesis physics engine.
Only the scenario "PickPlaceRandomBlock" has been fine tuned and trained.

## Installation

### Prerequisites

Genesis officially supports Windows, Mac, and Linux. Since this repository was created using Ubuntu, the following installation guide should be easier to follow if you are on Ubuntu. Otherwise, follow the instructions on the Genesis website linked above.

Creating a Python virtual environment is highly recommended to avoid version mismatches in modules.  
Tutorial for creating virtual environments: https://www.youtube.com/watch?v=hrnN2BRfIXE

### Installation Steps

1. **Install PyTorch**:
   Follow the official guide: https://pytorch.org/get-started/locally/ (copy the command you need)
   
   Check your CUDA version:
   ```bash
   nvidia-smi
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Run the following to start training :
```bash
python IL_run_ppo.py -n 10
```

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
