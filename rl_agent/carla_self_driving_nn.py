'''
this script contains the neural network (and stable baselines model adaptation) that is going to serve as the agent for the carla 

'''

import torch
from torch import nn
import numpy as np
import gymnasium as gym


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
            nn.Flatten()
        )

        self.fc = nn.Linear(5040, output_features) # you can calculate this with that cnn formula...

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



class CarlaPolicyNetwork(nn.Module):

    def __init__(self, image_output_dim=1024, state_output_dim=1024, hidden_dim=512, action_dim=3, dropout=0.2):
        super().__init__()

        self.depth_net = CarlaCNN(output_features=image_output_dim, dropout=dropout)
        self.seg_net = CarlaCNN(output_features=image_output_dim, dropout=dropout)
        self.feature_net = CarlaDNN(input_features=14, hidden_features=512, output_features=state_output_dim, dropout=dropout)

        self.policy_head = nn.Sequential(
            nn.Linear(2*image_output_dim + state_output_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)  # assume [throttle, brake, steer] as 3 outputs
        )

    def forward(self, obs):

        depth_features = self.depth_net(obs["depth_camera"])             # [B, 1024]
        seg_features = self.seg_net(obs["seg_camera"])                   # [B, 1024]
        other_features = self.feature_net(obs["sensor_data"])                           # [B, 1024]
        combined = torch.cat([depth_features, seg_features, other_features], dim=1)   # [B, 3092]

        raw_actions = self.policy_head(combined)     # [B, 3]
        throttle = torch.sigmoid(raw_actions[:, 0])  # Normalize throttle to range [0, 1]
        brake = torch.sigmoid(raw_actions[:, 1])     # Normalize brake to range [0, 1]
        steer = torch.tanh(raw_actions[:, 2])        # Normalize steer to range [-1, 1]
        
        return torch.stack([throttle, brake, steer], dim=1)              # [B, 3]



# Turn Np obs into Torch input for Network
class NpToTorchObsWrapper(gym.ObservationWrapper):

    def __init__(self, env, device="cpu"):
        super().__init__(env)
        self.device = device

    def observation(self, obs):

        depth_image = self.preprocess_image(obs["depth_camera"])
        seg_image = self.preprocess_image(obs["seg_camera"])
        flattened_obs = self.flatten_observation(obs)
        
        return {
            "depth_camera": depth_image.to(self.device),
            "seg_camera": seg_image.to(self.device),
            "sensor_data": flattened_obs.to(self.device),
        }

    @staticmethod
    def flatten_observation(obs):

        components = [
            obs["collision_sensor"],
            obs["velocity"],
            obs["acceleration"],
            obs["angular_velocity"],
            obs["throttle"],
            obs["brake"],
            obs["steer"],
            obs["speed"]
        ]
        return torch.tensor(np.concatenate(components, axis=0).astype(np.float32))  # [14]

    @staticmethod
    def preprocess_image(np_img_array):
        img_tensor = torch.tensor(np_img_array).permute(2, 0, 1).float()  # Convert from HWC -> CHW: [C, H, W]
        return img_tensor



# Turn Torch output into Np action for Environment
class TorchToNpyActionWrapper(gym.ActionWrapper):

    def __init__(self, env, device="cpu"):
        super().__init__(env)
        self.device = device

    def action(self, action):

        action = {
            "throttle": action[:, 0].detach().cpu().numpy()[0], # no more dimensions, just scalers
            "brake": action[:, 1].detach().cpu().numpy()[0],
            "steer": action[:, 2].detach().cpu().numpy()[0],
        }

        return action