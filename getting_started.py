
'''
this script is about getting started in how to using python's carla library to control the carla simulator

'''

import carla
import random
import pygame
import numpy as np
from open_carla import start_carla_simulator


# Open Carla
start_carla_simulator()

# Initialize Pygame for displaying the car's camera
pygame.init()

# Define the screen size (match CARLA image resolution)
width = 800  # Adjust this to match your CARLA camera resolution
height = 600  # Adjust this to match your CARLA camera resolution
screen = pygame.display.set_mode((width, height))

# Connect to carla simulator
client = carla.Client('localhost', 2000)

# Load the world
world = client.get_world()

# Get blueprint of the vehicle
blueprint_library = world.get_blueprint_library()
car_name = 'vehicle.dodgecop.charger'
vehicle_bp = blueprint_library.find(car_name)

# Choose a spawn point
spawn_point = random.choice(world.get_map().get_spawn_points())

# Spawn the vehicle
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Let it drive itself (autopilot on)
vehicle.set_autopilot(True) # this is not true autopilot, it's rule based connected to the engine itself

# Spawn a camera sensor and attach it to the vehicle
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_spawn_point = carla.Transform(carla.Location(x=2, z=1))  # Position the camera at a relative offset

# Spawn the camera and attach it to the vehicle
camera = world.spawn_actor(camera_bp, camera_spawn_point, attach_to=vehicle)

# Function to process images and display them using Pygame
def process_image(image):
    # Convert the image to a numpy array
    img_array = np.array(image.raw_data)
    
    # Reshape the array to match the image format (Carla uses RGBA by default)
    img_array = img_array.reshape((image.height, image.width, 4))
    
    # Convert to RGB for Pygame (discard the alpha channel)
    img_array = img_array[:, :, :3]  # Discard the alpha channel

    # Rotate the image 90 degrees counter-clockwise
    img_array = np.rot90(img_array, 1)  # Rotate by 90 degrees (counter-clockwise)
    
    # Convert to a Pygame surface
    surface = pygame.surfarray.make_surface(img_array)
    
    # Draw the image on the screen
    screen.blit(surface, (0, 0))
    
    # Update the display
    pygame.display.flip()

# Listen for camera data
camera.listen(process_image)

# Main loop to keep the Pygame window open and running
try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt  # Exit gracefully

except KeyboardInterrupt:
    print("Exiting...")

# Cleanup
camera.stop()  # Stop the camera listener
camera.destroy()  # Destroy the camera actor
vehicle.destroy()  # Destroy the vehicle actor

# Quit Pygame
pygame.quit()
