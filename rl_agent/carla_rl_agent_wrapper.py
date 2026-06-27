'''
SB3 policy configuration for the CARLA TD3 agent.

The previous version tried to subclass TD3Policy/Actor to plug a custom
CarlaPolicyNetwork straight into SB3 — but SB3's Actor receives features from
SB3's own features extractor, not the raw dict the custom network expected, so
the wiring never lined up.

The idiomatic approach is to keep SB3's default MlpPolicy and plug in a custom
features extractor (CarlaFeaturesExtractor) that fuses the depth + seg cameras
with the scalar sensor vector into a flat feature vector. SB3 then builds the
actor/critic heads, target networks, exploration noise, and gradient updates
itself.
'''

from stable_baselines3.common.type_aliases import PolicyKwargsDict

from carla_self_driving_nn import CarlaFeaturesExtractor


def default_policy_kwargs(features_dim: int = CarlaFeaturesExtractor.DEFAULT_FEATURES_DIM) -> PolicyKwargsDict:
    return {
        'features_extractor_class': CarlaFeaturesExtractor,
        'features_extractor_kwargs': {'features_dim': features_dim},
    }
