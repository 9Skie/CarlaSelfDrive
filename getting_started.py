'''
Getting started with CARLA: spawn an ego vehicle on autopilot and display its
RGB camera in a Pygame window.
'''

import random

import carla
import numpy as np
import pygame

from open_carla import start_carla_simulator


# Set CARLA_PATH to auto-launch the simulator, or None if it's already running.
CARLA_PATH = None

start_carla_simulator(carla_path=CARLA_PATH)

pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))

client = carla.Client('localhost', 2000)
world = client.get_world()

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.dodgecop.charger')
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# this is not true autopilot, it's rule based connected to the engine itself
vehicle.set_autopilot(True)

camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_spawn_point = carla.Transform(carla.Location(x=2, z=1))
camera = world.spawn_actor(camera_bp, camera_spawn_point, attach_to=vehicle)


def process_image(image):
    # CARLA hands us raw BGRA bytes; np.array(bytes) gives a 0-d object array,
    # so we have to use np.frombuffer to get a uint8 pixel buffer.
    img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    img_array = img_array.reshape((image.height, image.width, 4))

    # Drop alpha, swap BGR -> RGB so colours look right in Pygame.
    img_array = img_array[:, :, :3][:, :, ::-1]

    # Rotate 90° counter-clockwise to match the surfarray orientation.
    img_array = np.rot90(img_array, 1)

    surface = pygame.surfarray.make_surface(img_array)
    screen.blit(surface, (0, 0))
    pygame.display.flip()


camera.listen(process_image)

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
except KeyboardInterrupt:
    print('Exiting...')
finally:
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    pygame.quit()
