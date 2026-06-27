import subprocess
import os
import sys
import time


def start_carla_simulator(carla_path=None, offscreen=False):
    """
    Start the CARLA simulator. Set carla_path to wherever you installed CARLA.
    On macOS/Linux the executable is CarlaUnreal.sh, on Windows it's CarlaUnreal.exe.
    """

    if carla_path is None:
        print('Warning: carla_path not set. Assuming CARLA server is already running.')
        return

    if sys.platform == 'win32':
        carla_exe = os.path.join(carla_path, 'CarlaUnreal.exe')
    else:
        carla_exe = os.path.join(carla_path, 'CarlaUnreal.sh')

    if os.path.exists(carla_exe):
        print(f'Starting CARLA simulator from {carla_exe}')

        cmd = [carla_exe]
        if offscreen:
            cmd.append('-RenderOffScreen')

        subprocess.Popen(cmd)
        time.sleep(30)  # wait for CARLA to fully initialize
    else:
        print(f'Error: CARLA executable not found at {carla_exe}. Please verify the path.')
