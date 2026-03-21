# CARLA Self-Driving RL Agent

A reinforcement learning agent that learns to drive autonomously in the [CARLA simulator](https://carla.org/) using TD3 (Twin Delayed Deep Deterministic Policy Gradient) via Stable Baselines 3.

## How it works

The agent learns to navigate from a random spawn point to a random destination by trial and error. It receives camera and sensor data as input and outputs throttle, brake, and steer commands.

**Inputs (what the car sees):**
- Depth camera — how far away things are
- Semantic segmentation camera — what things are (road, pedestrian, building, etc.)
- Vehicle state — speed, velocity, acceleration, steering, throttle, brake

**Outputs (what the car does):**
- Throttle `[0, 1]`
- Brake `[0, 1]`
- Steer `[-1, 1]`

**Reward design:**
- Reward for getting closer to destination
- Large reward for reaching destination (episode ends)
- Penalty for collisions
- Large penalty + episode end if stuck for too long
- Small time penalty every step to encourage speed

## Architecture

```
rl_agent/
  main.py                   — training entry point
  carla_rl_env.py           — CARLA wrapped as a Gymnasium environment
  carla_self_driving_nn.py  — actor-critic neural network (CNN + DNN fusion)
  carla_rl_agent_wrapper.py — plugs the custom network into SB3's TD3
  open_carla.py             — utility to launch the CARLA server

1. getting_started.py       — basic CARLA connection test
2. manual_control.py        — drive the car manually with keyboard
```

## Setup

### 1. Install CARLA simulator

Download CARLA 0.9.15 from the [releases page](https://github.com/carla-simulator/carla/releases) and extract it somewhere on your machine.

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> Note: the `carla` Python package may need to be installed manually from your CARLA installation folder rather than via pip, depending on your platform.

### 3. Run training

Set your CARLA install path in `rl_agent/main.py`:

```python
start_carla_simulator(carla_path="/path/to/your/CARLA")
```

Then run:

```bash
cd rl_agent
python main.py
```

The trained model is saved as `td3_carla_model.zip` when complete.

## Requirements

- Python 3.8+
- CARLA 0.9.15
- A GPU is strongly recommended for training
