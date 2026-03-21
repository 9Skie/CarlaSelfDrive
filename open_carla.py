import subprocess
import os
import time

def start_carla_simulator(carla_path="E:\\CarlaTesting", offscreen=False):
    """
    Start the CARLA simulator by launching the CarlaUnreal.exe with optional offscreen rendering.
    """

    carla_exe = os.path.join(carla_path, "CarlaUnreal.exe")

    if os.path.exists(carla_exe):
        print(f"Starting CARLA simulator from {carla_exe}")
        
        # Build the command list
        cmd = [carla_exe]
        if offscreen:
            cmd.append("-RenderOffScreen")

        # Start the simulator process
        subprocess.Popen(cmd)
        time.sleep(30)  # wait for CARLA to fully initialize
    else:
        print(f"Error: CARLA executable not found at {carla_exe}. Please verify the path.")


