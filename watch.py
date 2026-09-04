import os
import glob
import time
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from robot_dog_env import RobotDogEnv


def get_latest_checkpoint(folder="checkpoints"):
    # zoek ook in per-run submappen (checkpoints/run_YYYYmmdd_HHMMSS/*.zip)
    files = glob.glob(os.path.join(folder, "**", "*.zip"), recursive=True)
    if not files:
        raise FileNotFoundError(f"Geen checkpoints gevonden in '{folder}/'. Is train.py al gestart?")
    latest = max(files, key=os.path.getmtime)
    return latest


MODEL_PATH = get_latest_checkpoint()
print(f"Loading: {MODEL_PATH}")

model = PPO.load(MODEL_PATH)
env = RobotDogEnv(xml_path="robot_dog.xml", max_steps=100_000)
obs, info = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        viewer.sync()

        if terminated or truncated:
            obs, info = env.reset()

        # match wall-clock time to the control period (frame_skip physics steps) for real-time playback
        elapsed = time.time() - step_start
        sleep_time = env.control_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
