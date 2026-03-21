"""
the main training program, this is incomplete and probably not possible to run, maybe some day in the future when I can make more sense of it talking with other people
"""

import stable_baselines3 as sb3
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import torch
import torch.nn as nn

# my stuff
from open_carla import start_carla_simulator
from carla_rl_env import CarlaEnv, CarlaRenderCallback
from carla_self_driving_nn import NpToTorchObsWrapper, TorchToNpyActionWrapper
from carla_rl_agent_wrapper import CarlaTD3Policy


start_carla_simulator(offscreen=True)
env = CarlaEnv()
env = NpToTorchObsWrapper(env)
env = TorchToNpyActionWrapper(env)

model = TD3(policy=CarlaTD3Policy, env=env, verbose=0)
render_cb = CarlaRenderCallback()
model.learn(total_timesteps=100000, callback=render_cb)
model.save("td3_carla_model")
