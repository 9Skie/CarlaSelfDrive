'''
CARLA reinforcement learning environment (gymnasium) + render callback.

Runs the simulator in synchronous mode with a fixed timestep so that
world.tick() advances the simulation deterministically.
'''

import random
from collections import deque

import carla
import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3.common.callbacks import BaseCallback


FIXED_DELTA_SECONDS = 0.05  # 20 Hz simulation tick


class CarlaEnv(gym.Env):

    def __init__(self, max_steps=5000, host='localhost', port=2000):
        super().__init__()

        client = carla.Client(host, port)
        client.set_timeout(10.0)
        self.client = client
        self.world = client.get_world()

        # Switch to synchronous mode with a fixed timestep so that
        # world.tick() advances the sim and our step counts are meaningful.
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)

        self.depth_image = None
        self.seg_image = None
        self.collision = False

        self.terminated = False
        self.truncated = False
        self.info = {}

        self.steps_for_15_sec = int(15 / FIXED_DELTA_SECONDS)
        self.step_count = 0
        self.max_steps = max_steps

        self.vehicle = None
        self.depth_cam = None
        self.seg_cam = None
        self.collision_sensor = None

        self.observation_space = gym.spaces.Dict({

            # comes from carla cameras, and sensor
            'depth_camera': gym.spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8),
            'seg_camera': gym.spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8),
            'collision_sensor': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),

            # comes from the vehicle with .get_velocity(), .get_acceleration() , .get_angular_velocity(), all 3D vectors
            'velocity': gym.spaces.Box(low=-100.0, high=100.0, shape=(3,), dtype=np.float32),
            'acceleration': gym.spaces.Box(low=-50.0, high=50.0, shape=(3,), dtype=np.float32),
            'angular_velocity': gym.spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32),

            # comes from the vehicle with .get_control()
            'throttle': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'brake': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'steer': gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),

            # computed from the velocity vector magnitude
            'speed': gym.spaces.Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32),
        })

        # Box(3,) action space — order is [throttle, brake, steer].
        # SB3's TD3 requires a Box action space.
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Pygame display is created lazily in render() so the env can be used
        # headless (e.g. inside SubprocVecEnv) without opening a window.
        self.display_width = 1600
        self.display_height = 800
        self.display = None
        self.font = None


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._destroy_actors()

        blueprint_library = self.world.get_blueprint_library()

        car_bp = blueprint_library.find('vehicle.dodgecop.charger')
        car_bp.set_attribute('role_name', 'hero')
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(car_bp, spawn_point)
        self.vehicle.set_autopilot(False)

        # Pick a goal different from the spawn point so the episode isn't
        # trivially solved at reset.
        current_location = self.vehicle.get_location()
        other_points = [p for p in spawn_points if p.location.distance(current_location) > 5.0]
        if other_points:
            self.goal_location = random.choice(other_points).location
        else:
            self.goal_location = random.choice(spawn_points).location

        self.prev_distance = self._distance(current_location, self.goal_location)
        self.speed_buffer = deque(maxlen=self.steps_for_15_sec)

        self.depth_image = None
        self.seg_image = None
        self.collision = False

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.5))

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

        # Wait for sensors to produce their first frame, with a hard timeout so
        # we don't hang forever if CARLA is misbehaving.
        for _ in range(200):
            if self.depth_image is not None and self.seg_image is not None:
                break
            self.world.tick()

        return self.get_observation(), self.info


    def step(self, action):
        throttle, brake, steer = float(action[0]), float(action[1]), float(action[2])
        control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
        self.vehicle.apply_control(control)
        self.world.tick()

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.truncated = True

        observation = self.get_observation()
        current_location = self.vehicle.get_location()
        reward = self.reward_fn(current_location, observation['speed'])

        return observation, reward, self.terminated, self.truncated, self.info


    def get_observation(self):
        velocity = self.vehicle.get_velocity()
        acceleration = self.vehicle.get_acceleration()
        angular_velocity = self.vehicle.get_angular_velocity()
        control = self.vehicle.get_control()
        speed = float(np.linalg.norm([velocity.x, velocity.y, velocity.z]))

        return {
            'depth_camera': self.depth_image,
            'seg_camera': self.seg_image,
            'collision_sensor': np.array([int(self.collision)], dtype=np.int32),
            'velocity': np.array([velocity.x, velocity.y, velocity.z], dtype=np.float32),
            'acceleration': np.array([acceleration.x, acceleration.y, acceleration.z], dtype=np.float32),
            'angular_velocity': np.array([angular_velocity.x, angular_velocity.y, angular_velocity.z], dtype=np.float32),
            'throttle': np.array([control.throttle], dtype=np.float32),
            'brake': np.array([control.brake], dtype=np.float32),
            'steer': np.array([control.steer], dtype=np.float32),
            'speed': np.array([speed], dtype=np.float32),
        }


    def reward_fn(self, current_location, speed):
        reward = 0.0

        # 1. reward the car for getting closer to target position
        current_distance = self._distance(current_location, self.goal_location)
        reward += self.prev_distance - current_distance
        self.prev_distance = current_distance

        # 2. reward+ the car for getting to the target position, terminate episode
        if current_distance < 2.0:
            reward += 1000.0
            self.terminated = True

        # 3. penalize the car for colliding with anything
        if self.collision:
            reward -= 100.0
            self.terminated = True

        # 4. penalize+ the car if it gets stuck (low avg speed over the last
        #    15 sim-seconds), terminate episode
        self.speed_buffer.append(float(speed))
        if len(self.speed_buffer) >= self.steps_for_15_sec and np.mean(self.speed_buffer) < 0.1:
            reward -= 1000.0
            self.terminated = True

        # 5. time penalty so the agent prefers shorter routes
        reward -= 0.01

        return reward


    def render(self):
        if self.depth_image is None or self.seg_image is None:
            return

        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.display_width, self.display_height))
            pygame.display.set_caption('CARLA RL Viewer')
            self.font = pygame.font.SysFont('Arial', 20)

        def process_image(img_array):
            img_array = np.rot90(img_array, 1)
            surface = pygame.surfarray.make_surface(img_array)
            surface = pygame.transform.flip(surface, True, False)
            return surface

        depth_surface = process_image(self.depth_image)
        seg_surface = process_image(self.seg_image)

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
            f"Collision: {'Yes' if observation['collision_sensor'][0] else 'No'}",
            f"Position: ({self.vehicle.get_location().x:.1f}, {self.vehicle.get_location().y:.1f})",
            f"Target: ({self.goal_location.x:.1f}, {self.goal_location.y:.1f})",
        ]

        for i, text in enumerate(sidebar):
            rendered = self.font.render(text, True, (255, 255, 255))
            self.display.blit(rendered, (10 + (i % 4) * 400, 610 + (i // 4) * 30))

        pygame.display.flip()


    def close(self):
        self._destroy_actors()

        # Restore async mode so an external CARLA server keeps ticking on its own.
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except RuntimeError:
            pass

        if self.display is not None:
            pygame.quit()
            self.display = None


    def _destroy_actors(self):
        for sensor_attr in ('depth_cam', 'seg_cam', 'collision_sensor'):
            sensor = getattr(self, sensor_attr, None)
            if sensor is not None:
                try:
                    sensor.stop()
                except RuntimeError:
                    pass
                try:
                    sensor.destroy()
                except RuntimeError:
                    pass
                setattr(self, sensor_attr, None)

        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except RuntimeError:
                pass
            self.vehicle = None


    @staticmethod
    def _distance(a, b):
        return float(np.linalg.norm([a.x - b.x, a.y - b.y, a.z - b.z]))


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
        # The callback only fires when a collision happens, so any impulse
        # above threshold marks the episode as collided.
        if event.normal_impulse.length() > 0.1:
            self.collision = True



class CarlaRenderCallback(BaseCallback):
    def __init__(self, render_freq=10, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.training_env.render()
        return True
