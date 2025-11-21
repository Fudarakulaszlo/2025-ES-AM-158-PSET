import gymnasium as gym
import upkie.envs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

# Register Upkie environments
upkie.envs.register()

def train():
    # 1. Configure the environment
    # We use a single environment for simplicity, but vectorized environments (n_envs > 1) 
    # speed up training significantly.
    env_id = "Upkie-Spine-Pendulum"
    env_kwargs = dict(frequency=200.0)
    
    # Create the vectorized environment
    env = make_vec_env(env_id, n_envs=1, env_kwargs=env_kwargs)

    # 2. Define the PPO Model
    # MlpPolicy uses standard dense neural networks.
    # We use default hyperparameters which are generally robust for robotics tasks.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    # 3. Train
    # 100,000 timesteps is usually enough for the pendulum task to stabilize,
    # equating to about 8 minutes of simulated time at 200Hz.
    print("Starting training...")
    model.learn(total_timesteps=100_000)

    # 4. Save the model
    model_path = "models/pendulum_best/ppo_upkie_pendulum"
    model.save(model_path)
    print(f"Training finished. Model saved to {model_path}.zip")

if __name__ == "__main__":
    train()