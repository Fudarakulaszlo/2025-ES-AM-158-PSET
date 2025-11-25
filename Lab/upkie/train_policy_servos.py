#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym
import numpy as np
import upkie.envs
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Dict, Optional
import logging

# Register environments
upkie.envs.register()

# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------
# Suppress specific loggers that spam warnings during CPU-intensive training
logging.getLogger("upkie").setLevel(logging.ERROR)
logging.getLogger("loop_rate_limiters").setLevel(logging.ERROR)

# ---------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------
class ServoVelActionWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        fixed_order: Optional[Dict[str, list]] = None,
        gains: Optional[Dict[str, float]] = None,
    ):
        super().__init__(env)
        if not isinstance(self.env.action_space, gym.spaces.Dict):
            raise TypeError("UpkieServos expected Dict action_space")

        all_names = list(self.env.action_space.spaces.keys())
        wheels = [n for n in all_names if "wheel" in n]
        legs   = [n for n in all_names if ("hip" in n) or ("knee" in n)]

        if fixed_order is not None:
            self.wheel_names = list(fixed_order["wheel_names"])
            self.leg_names   = list(fixed_order["leg_names"])
        else:
            self.wheel_names = wheels
            self.leg_names   = legs

        self.wheel_vel_lim = np.array(
            [float(self.env.action_space[j]["velocity"].high[0]) for j in self.wheel_names],
            dtype=np.float32,
        )
        self.wheel_tau_lim = np.array(
            [float(self.env.action_space[j]["maximum_torque"].high[0]) for j in self.wheel_names],
            dtype=np.float32,
        )

        self.leg_pos_low  = np.array(
            [float(self.env.action_space[j]["position"].low[0]) for j in self.leg_names],
            dtype=np.float32,
        )
        self.leg_pos_high = np.array(
            [float(self.env.action_space[j]["position"].high[0]) for j in self.leg_names],
            dtype=np.float32,
        )
        self.leg_tau_lim = np.array(
            [float(self.env.action_space[j]["maximum_torque"].high[0]) for j in self.leg_names],
            dtype=np.float32,
        )

        gains = gains or {}
        self.kp_wheel = float(gains.get("kp_wheel", 0.0))
        self.kd_wheel = float(gains.get("kd_wheel", 1.7))
        self.kp_leg   = float(gains.get("kp_leg",   2.0))
        self.kd_leg   = float(gains.get("kd_leg",   1.7))

        n = len(self.wheel_names) + len(self.leg_names)
        self._n_wheels = len(self.wheel_names)
        self._n_legs   = len(self.leg_names)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)

    def action(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a = np.clip(a, -1.0, 1.0)

        a_wheel = a[: self._n_wheels] if self._n_wheels > 0 else np.zeros(0, np.float32)
        a_leg   = a[self._n_wheels :]  if self._n_legs   > 0 else np.zeros(0, np.float32)

        wheel_v_cmd = a_wheel * self.wheel_vel_lim
        leg_pos_cmd = (
            self.leg_pos_low + (a_leg + 1.0) * 0.5 * (self.leg_pos_high - self.leg_pos_low)
            if self._n_legs > 0 else np.zeros(0, np.float32)
        )

        env_action = {}
        EPS = 1e-6

        for i, name in enumerate(self.wheel_names):
            env_action[name] = dict(
                position=np.nan,
                velocity=float(wheel_v_cmd[i]),
                feedforward_torque=0.0,
                kp_scale=self.kp_wheel,
                kd_scale=self.kd_wheel,
                maximum_torque=float(self.wheel_tau_lim[i] - EPS),
            )

        for i, name in enumerate(self.leg_names):
            env_action[name] = dict(
                position=float(leg_pos_cmd[i]),
                velocity=0.0,
                feedforward_torque=0.0,
                kp_scale=self.kp_leg,
                kd_scale=self.kd_leg,
                maximum_torque=float(self.leg_tau_lim[i] - EPS),
            )

        return env_action

class ServoObsFlattenWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.joint_names = list(self.env.observation_space.spaces.keys())

        lows, highs = [], []
        for j in self.joint_names:
            pos_box = self.env.observation_space[j]["position"]
            vel_box = self.env.observation_space[j]["velocity"]
            lows  += [float(pos_box.low[0]),  float(vel_box.low[0])]
            highs += [float(pos_box.high[0]), float(vel_box.high[0])]
        self.observation_space = gym.spaces.Box(
            low=np.asarray(lows, dtype=np.float32),
            high=np.asarray(highs, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation):
        vec = []
        for j in self.joint_names:
            vec += [float(observation[j]["position"][0]),
                    float(observation[j]["velocity"][0])]
        return np.asarray(vec, dtype=np.float32)

def make_wrapped_env():
    env = gym.make("Upkie-Spine-Servos", frequency=200.0)
    env = ServoVelActionWrapper(env)
    env = ServoObsFlattenWrapper(env)
    env = TimeLimit(env, max_episode_steps=1000)
    return env

def train():
    # Create vectorized env for training
    env = make_vec_env(make_wrapped_env, n_envs=4)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log="./tensorboard/",
    )

    print("Starting training (1,000,000 timesteps)...")
    model.learn(total_timesteps=3_000_000)
    
    model_path = "ppo_upkie_servos"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

if __name__ == "__main__":
    train()