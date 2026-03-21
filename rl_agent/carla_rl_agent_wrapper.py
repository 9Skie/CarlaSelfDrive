'''
this script will turn the neural network we made into a stable baselines agent

How it fits together:
- CarlaTD3Policy is what gets passed to TD3(...) — SB3 expects a Policy class here
- CarlaTD3Policy subclasses TD3Policy and overrides make_actor() to plug in our custom network
- CarlaActor is the actual actor: it wraps CarlaPolicyNetwork and slots into SB3's TD3 internals
- SB3 handles the critic, target networks, exploration noise, and gradient updates automatically
'''

import torch.nn as nn
from stable_baselines3.td3.policies import TD3Policy, Actor

from carla_self_driving_nn import CarlaPolicyNetwork


class CarlaActor(Actor):
    def __init__(self, observation_space, action_space, net_arch, features_extractor, features_dim, activation_fn=nn.ReLU, normalize_images=True):
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim, activation_fn, normalize_images)
        self.custom_net = CarlaPolicyNetwork()

    def forward(self, obs):
        return self.custom_net(obs)

    def _predict(self, observation, deterministic=False):
        return self(observation)


class CarlaTD3Policy(TD3Policy):
    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CarlaActor(**actor_kwargs).to(self.device)
