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
        save_path="./checkpoints/",
        name_prefix="robot_dog"
    )

    model.learn(total_timesteps=5_000_000, callback=checkpoint_callback)
    model.save("robot_dog_final")

    print("Training done. Model saved as robot_dog_final.zip")
