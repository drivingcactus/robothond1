import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from robot_dog_env import RobotDogEnv


def make_env():
    def _init():
        return RobotDogEnv(xml_path="robot_dog.xml")
    return _init


if __name__ == "__main__":
    N_ENVS = 10  # 12-core CPU: laat 1-2 cores vrij voor OS/overhead

    # elke run zijn eigen map, zodat checkpoints van verschillende runs elkaar niet overschrijven
    # en de tensorboard-run dezelfde naam heeft als de checkpoint-map
    RUN_NAME = time.strftime("run_%Y%m%d_%H%M%S")
    CHECKPOINT_DIR = os.path.join("checkpoints", RUN_NAME)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Run: {RUN_NAME}  ->  checkpoints in {CHECKPOINT_DIR}/")

    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tb_logs/",
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        device="auto",  # uses GPU if available, otherwise CPU
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // N_ENVS,
        save_path=CHECKPOINT_DIR,
        name_prefix="robot_dog"
    )

    model.learn(total_timesteps=5_000_000, callback=checkpoint_callback, tb_log_name=RUN_NAME)
    final_path = os.path.join(CHECKPOINT_DIR, "robot_dog_final")
    model.save(final_path)

    print(f"Training done. Model saved as {final_path}.zip")
