'''
this script is about controlling a carla car manually, adding more traffic

'''

import carla
import random
import pygame
import numpy as np
from open_carla import start_carla_simulator



# start carla
# start_carla_simulator()

# Pygame setup
pygame.init()
display = pygame.display.set_mode((800, 600))
pygame.display.set_caption("CARLA Manual Control")

# Connect to CARLA
client = carla.Client('localhost', 2000)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Spawn a vehicle
car_bp = blueprint_library.find("vehicle.dodgecop.charger")
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(car_bp, spawn_point)

# Enable physics control
vehicle.set_autopilot(False)


# Spawn and attach camera
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_spawn = carla.Transform(carla.Location(x=1.5, z=2.5))  # Roof-ish cam position
camera = world.spawn_actor(camera_bp, camera_spawn, attach_to=vehicle)


# Spawn other vehicles for simulated traffic
traffic_manager = client.get_trafficmanager(8000)
traffic_vehicles = []
spawn_points = world.get_map().get_spawn_points()
# Cap NPC count at the number of available spawn points so we don't IndexError.
num_vehicles = min(5, len(spawn_points))
for _ in range(num_vehicles):
    bp = random.choice(blueprint_library.filter('vehicle.*'))
    spawn = random.choice(spawn_points)
    npc = world.try_spawn_actor(bp, spawn)
    if npc:
        npc.set_autopilot(True, traffic_manager.get_port())
        traffic_vehicles.append(npc)

# Image rendering
def process_image(image):
    img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    img_array = img_array.reshape((image.height, image.width, 4))[:, :, :3]
    # CARLA gives BGRA; swap to RGB so colours display correctly.
    img_array = img_array[:, :, ::-1]
    img_array = np.rot90(img_array, 1)  # Rotate 90° counter-clockwise
    surface = pygame.surfarray.make_surface(img_array)
    surface = pygame.transform.flip(surface, True, False)
    display.blit(surface, (0, 0))
    pygame.display.flip()

camera.listen(process_image)

# Manual control
clock = pygame.time.Clock()
control = carla.VehicleControl()
reverse_mode = False 

try:
    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        keys = pygame.key.get_pressed()
        control = carla.VehicleControl()
        control.reverse = reverse_mode
        
        if keys[pygame.K_w]:
            control.throttle = 1.0
            control.reverse = False
        elif keys[pygame.K_s]:
            control.throttle = 1.0
            control.reverse = True

        if keys[pygame.K_a]:
            control.steer = -0.5
        elif keys[pygame.K_d]:
            control.steer = 0.5

        vehicle.apply_control(control)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    for npc in traffic_vehicles:
        npc.destroy()
    pygame.quit()