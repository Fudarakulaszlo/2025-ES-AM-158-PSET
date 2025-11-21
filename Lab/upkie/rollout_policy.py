# rollout_eval_no_norm.py
import gymnasium as gym
import upkie.envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

# Register Upkie environments
upkie.envs.register()

MODEL_PATH = "models/pendulum_best/ppo_upkie_pendulum"  # Path to the saved model
ENV_ID = "Upkie-Spine-Pendulum"
ENV_KWARGS = dict(frequency=200.0)
SEED = 42

def main():
    # Create environment
    # We assume the environment is already running or will be started by the wrapper
    env = make_vec_env(ENV_ID, n_envs=1, env_kwargs=ENV_KWARGS, seed=SEED)

    # Load the trained model
    try:
        model = PPO.load(MODEL_PATH, env=env)
        print(f"Successfully loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}.zip' not found. Run train_policy.py first.")
        return

    # Evaluation Loop
    obs = env.reset()
    ep_return = 0.0
    ep_len = 0
    
    print("Starting rollout...")
    while True:
        # Predict action (deterministic=True for evaluation)
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, dones, infos = env.step(action)
        
        ep_return += float(reward[0])
        ep_len += 1
        
        # Check termination
        if bool(dones[0]):
            break
            
    print(f"[EVAL] Return={ep_return:.3f}, Length={ep_len} steps")
    
    # Optional: Print final observation to see why it ended
    # obs is vectorized, so we take the first element
    final_pitch = obs[0][0] 
    final_pos = obs[0][1]
    print(f"Final State - Pitch: {final_pitch:.3f} rad, Position: {final_pos:.3f} m")

if __name__ == "__main__":
    main()