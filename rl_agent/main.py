'''
TD3 training entry point for the CARLA self-driving agent.
'''

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from open_carla import start_carla_simulator
from carla_rl_env import CarlaEnv, CarlaRenderCallback
from carla_rl_agent_wrapper import default_policy_kwargs


# Set CARLA_PATH below to auto-launch the simulator. Leave None if you start
# CARLA yourself (e.g. `./CarlaUnreal.sh -RenderOffScreen` on the server side).
CARLA_PATH = None

start_carla_simulator(carla_path=CARLA_PATH, offscreen=True)
env = CarlaEnv()

n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions),
)

model = TD3(
    policy='MlpPolicy',
    env=env,
    policy_kwargs=default_policy_kwargs(),
    action_noise=action_noise,
    verbose=1,
)

render_cb = CarlaRenderCallback()
model.learn(total_timesteps=100_000, callback=render_cb)
model.save('td3_carla_model')

env.close()
