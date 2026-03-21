'''
this script will want to try to turn the carla environment into a reinforcement learning environment using gymnasium's gym class
'''

import carla
import gymnasium as gym
import random
import pygame
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CarlaEnv(gym.Env):

    def __init__(self, max_steps=5000):
        super().__init__()

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        self.client = client
        self.world = client.get_world()

        self.depth_image = None
        self.seg_image = None
        self.collision = False

        self.terminated = False
        self.truncated = False
        self.info = {}

        time_step = self.world.get_settings().fixed_delta_seconds
        self.steps_for_30_sec = int(15 / time_step)
        self.step_count = 0
        self.max_steps = max_steps

        self.vehicle = None
        self.depth_cam = None
        self.seg_cam = None
        self.collision_sensor = None

        self.observation_space = gym.spaces.Dict({

            # comes from carla cameras, and sensor
            "depth_camera": gym.spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8),
            "seg_camera": gym.spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8),
            "collision_sensor": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),  # 0 = no collision, 1 = collision

            # comes from the vehicle with .get_velocity(), .get_acceleration() , .get_angular_velocity(), all 3D vectors
            "velocity": gym.spaces.Box(low=-100.0, high=100.0, shape=(3,), dtype=np.float32),  
            "acceleration": gym.spaces.Box(low=-50.0, high=50.0, shape=(3,), dtype=np.float32),
            "angular_velocity": gym.spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32),

            # comes from the vehicle with .get_control()
            "throttle": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "brake": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "steer": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),

            # computed from the velocity vector magnitude
            "speed": gym.spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32),
        })

        self.action_space = gym.spaces.Dict({
           "throttle": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
           "brake": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
           "steer": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        pygame.init()
        self.display_width = 1600
        self.display_height = 800
        self.display = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption("CARLA RL Viewer")
        self.font = pygame.font.SysFont('Arial', 20)


    def reset(self):

        # Destroy previous actors
        if self.vehicle:
            self.vehicle.destroy()
        if self.depth_cam:
            self.depth_cam.stop()
            self.depth_cam.destroy()
        if self.seg_cam:
            self.seg_cam.stop()
            self.seg_cam.destroy()
        if self.collision_sensor:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()

        blueprint_library = self.world.get_blueprint_library()

        car_bp = blueprint_library.find("vehicle.dodgecop.charger")
        car_bp.set_attribute('role_name', 'hero') # ego vehicle
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(car_bp, spawn_point)
        self.vehicle.set_autopilot(False)

        self.goal_location = random.choice(self.world.get_map().get_spawn_points()).location
        current_location = self.vehicle.get_location()
        self.prev_distance = np.linalg.norm(np.array([current_location.x - self.goal_location.x, current_location.y - self.goal_location.y, current_location.z - self.goal_location.z]))
        self.speed_buffer = []

        self.depth_image = None
        self.seg_image = None
        self.collision = False

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.5))  # Roof-ish cam position

        depth_bp = blueprint_library.find('sensor.camera.depth')
        self.depth_cam = self.world.spawn_actor(depth_bp, camera_transform, attach_to=self.vehicle)
        self.depth_cam.listen(lambda image: self._on_depth_image(image))

        seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        self.seg_cam = self.world.spawn_actor(seg_bp, camera_transform, attach_to=self.vehicle)
        self.seg_cam.listen(lambda image: self._on_seg_image(image))

        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        self.terminated = False
        self.truncated = False
        self.info = {}
        self.step_count = 0 

        # Ensure sensors have data before returning
        while self.depth_image is None or self.seg_image is None:
            self.world.tick()

        self.display.fill((0, 0, 0))  # Clear the display
        pygame.display.flip()

        return self.get_observation(), self.info


    def step(self, action):

        control = carla.VehicleControl()
        control.throttle = action["throttle"] # throttle
        control.brake = action["brake"] # brake
        control.steer = action["steer"] # steer
        self.vehicle.apply_control(control)
        self.world.tick()

        # hard ceiling to prevent any episode from going on too long
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.truncated = True

        observation = self.get_observation()
        current_location = self.vehicle.get_location()
        reward = self.reward_fn(current_location, observation["speed"])

        return observation, reward, self.terminated, self.truncated, self.info
    

    def get_observation(self):

        velocity = self.vehicle.get_velocity()
        acceleration = self.vehicle.get_acceleration()
        angular_velocity = self.vehicle.get_angular_velocity()
        control = self.vehicle.get_control()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

        observation = {
            "depth_camera": self.depth_image,
            "seg_camera": self.seg_image,
            "collision_sensor": np.array([int(self.collision)]),
            "velocity": np.array([velocity.x, velocity.y, velocity.z]),
            "acceleration": np.array([acceleration.x, acceleration.y, acceleration.z]),
            "angular_velocity": np.array([angular_velocity.x, angular_velocity.y, angular_velocity.z]),
            "throttle": np.array([control.throttle]),
            "brake": np.array([control.brake]),
            "steer": np.array([control.steer]),
            "speed": np.array([speed])
        }

        return observation

    def reward_fn(self, current_location, speed):

        reward = 0.0

        # car starts at a position, goal is to go to target position
        # so the reward goes something like this...

        # 1. reward the car for getting closer to target position
        # 2. reward+ the car for getting to the target position, terminate episode
        # 3. penalize the car for colliding with anything
        # 4. penalize+ the car if it get stuck, terminate episode (we don't give a shit about green red lights)
        # 5. automatically penalize the car for TIME, such that it loses reward constantly (so I hope it tries to get the destination faster)

        # 1
        current_distance = np.linalg.norm(np.array([current_location.x - self.goal_location.x, current_location.y - self.goal_location.y, current_location.z - self.goal_location.z]))
        distance_delta = self.prev_distance - current_distance
        self.prev_distance = current_distance
        reward += distance_delta

        # 2
        if current_distance < 2.0:
            reward += 1000.0
            self.terminated = True

        # 3
        if self.collision:
            reward -= 100.0
    
        # 4
        self.speed_buffer.append(speed)
        if len(self.speed_buffer) > self.steps_for_30_sec: # basically keeps speed records for 15 seconds
            if np.mean(self.speed_buffer) < 0.1:
                reward -= 1000.0
                self.terminated = True
        
        # 5
        reward -= 0.01  # tune this value to control urgency

        return reward


    # don't know pygame much let AI wrote this code for rendering
    def render(self, mode='human'):
        if self.depth_image is None or self.seg_image is None:
            return
        
        def process_image(img_array):
            img_array = np.rot90(img_array, 1)
            surface = pygame.surfarray.make_surface(img_array)
            surface = pygame.transform.flip(surface, True, False)
            return surface

        # Process both images
        depth_surface = process_image(self.depth_image)
        seg_surface = process_image(self.seg_image)

        # put depth and seg side-by-side
        self.display.blit(depth_surface, (0, 0))
        self.display.blit(seg_surface, (800, 0))

        overlay = pygame.Surface((1600, 200))
        overlay.fill((0, 0, 0))
        self.display.blit(overlay, (0, 600)) 

        observation = self.get_observation()
        sidebar = [
            f"Speed: {observation['speed'][0]:.2f} m/s",
            f"Throttle: {observation['throttle'][0]:.2f}",
            f"Brake: {observation['brake'][0]:.2f}",
            f"Steer: {observation['steer'][0]:.2f}",
            f"Collision: {'Yes' if observation['collision_sensor'] else 'No'}",
            f"Position: ({self.vehicle.get_location().x:.1f}, {self.vehicle.get_location().y:.1f})",
            f"Target: ({self.goal_location.x:.1f}, {self.goal_location.y:.1f})"
        ]

        for i, text in enumerate(sidebar):
            rendered = self.font.render(text, True, (255, 255, 255))
            self.display.blit(rendered, (10 + (i % 4) * 400, 610 + (i // 4) * 30))

        pygame.display.flip()


    def close(self):
        
        self.depth_cam.stop()
        self.depth_cam.destroy()

        self.seg_cam.stop()
        self.seg_cam.destroy()

        self.collision_sensor.stop()
        self.collision_sensor.destroy()

        self.vehicle.destroy()
        pygame.quit()


    def _on_depth_image(self, image):

        image.convert(carla.ColorConverter.LogarithmicDepth)
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img_array = img_array.reshape((image.height, image.width, 4))[:, :, :3]
        self.depth_image = img_array


    def _on_seg_image(self, image):

        image.convert(carla.ColorConverter.CityScapesPalette)
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img_array = img_array.reshape((image.height, image.width, 4))[:, :, :3]
        self.seg_image = img_array


    def _on_collision(self, event):

        if event.normal_impulse.length() > 0.1:  # Threshold for collision force
            if not self.collision:
                self.collision = True
        else:
            if self.collision:
                self.collision = False



class CarlaRenderCallback(BaseCallback):
    def __init__(self, render_freq=10, verbose=0): # render_freq is the number of steps between each render call
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.training_env.render()
        return True