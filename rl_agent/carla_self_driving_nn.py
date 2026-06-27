'''
Neural network building blocks + SB3 features extractor for the CARLA agent.

The features extractor fuses two camera streams (depth, segmentation) with the
scalar sensor vector into a flat feature vector that SB3's standard MlpPolicy
can consume. This is the idiomatic way to mix image + vector observations in
SB3 — no custom Actor/Policy subclass needed.
'''

import torch
from torch import nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# 600x800 input through CarlaCNN.cnn flattens to 16 * 16 * 21 = 5376
_CNN_FLATTEN_DIM = 5376


class CarlaCNN(nn.Module):

    def __init__(self, output_features=1024, dropout=0.2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=3),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, stride=3),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        self.fc = nn.Linear(_CNN_FLATTEN_DIM, output_features)

    def forward(self, obs_image):
        return self.fc(self.cnn(obs_image))


class CarlaDNN(nn.Module):

    def __init__(self, input_features=14, hidden_features=512, output_features=1024, dropout=0.2):
        super().__init__()

        self.dnn = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, output_features),
        )

    def forward(self, obs):
        return self.dnn(obs)


class CarlaFeaturesExtractor(BaseFeaturesExtractor):
    '''
    Fuses depth + segmentation cameras with the scalar sensor vector into a
    flat feature vector of size `features_dim` (default 3072 = 1024 * 3).

    SB3 normalizes uint8 images to [0,1] float and adds a batch dim before
    passing obs to forward(), so we receive [B, H, W, C] tensors.
    '''

    DEFAULT_FEATURES_DIM = 3072

    def __init__(self, observation_space: gym.Space, features_dim: int = DEFAULT_FEATURES_DIM):
        assert isinstance(observation_space, gym.spaces.Dict), 'CarlaFeaturesExtractor needs a Dict observation_space'

        super().__init__(observation_space, features_dim)

        spaces = observation_space.spaces
        per_stream = features_dim // 3

        self.depth_net = CarlaCNN(output_features=per_stream)
        self.seg_net = CarlaCNN(output_features=per_stream)

        sensor_dim = (
            spaces['collision_sensor'].shape[0]
            + spaces['velocity'].shape[0]
            + spaces['acceleration'].shape[0]
            + spaces['angular_velocity'].shape[0]
            + spaces['throttle'].shape[0]
            + spaces['brake'].shape[0]
            + spaces['steer'].shape[0]
            + spaces['speed'].shape[0]
        )
        self.feature_net = CarlaDNN(input_features=sensor_dim, output_features=per_stream)

    def forward(self, obs: dict) -> torch.Tensor:
        # SB3 hands us [B, H, W, C]; Conv2d wants [B, C, H, W]
        depth = obs['depth_camera'].permute(0, 3, 1, 2)
        seg = obs['seg_camera'].permute(0, 3, 1, 2)

        sensor = torch.cat(
            [
                obs['collision_sensor'].float(),
                obs['velocity'],
                obs['acceleration'],
                obs['angular_velocity'],
                obs['throttle'],
                obs['brake'],
                obs['steer'],
                obs['speed'],
            ],
            dim=-1,
        )

        depth_f = self.depth_net(depth)
        seg_f = self.seg_net(seg)
        sensor_f = self.feature_net(sensor)

        return torch.cat([depth_f, seg_f, sensor_f], dim=1)
