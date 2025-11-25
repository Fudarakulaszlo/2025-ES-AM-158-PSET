#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

## \namespace upkie.envs.upkie_servos
## \brief Upkie environment where actions command servomotors directly.

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

# We try to import Backend/RobotState if available, otherwise we assume
# the user might be running in a simplified context.
try:
    from upkie.envs.backends import Backend
    from upkie.utils.robot_state import RobotState
except ImportError:
    # Fallback placeholders if run independently without full package
    Backend = None
    RobotState = None

from upkie.exceptions import UpkieRuntimeError
from upkie.utils.clamp import clamp_and_warn
from .upkie_env import UpkieEnv


class UpkieServos(UpkieEnv):
    r"""!
    Upkie environment where actions command servomotors directly.

    \anchor upkie_servos_description

    Actions and observations correspond to the moteus servo API.
    
    **Task**: The agent is rewarded for keeping the robot upright (pitch ~ 0).
    """

    ACTION_KEYS: Tuple[str, str, str, str, str, str] = (
        "position",
        "velocity",
        "feedforward_torque",
        "kp_scale",
        "kd_scale",
        "maximum_torque",
    )

    ## \var action_space
    ## Action space of the environment.
    action_space: gym.Space

    ## \var observation_space
    ## Observation space of the environment.
    observation_space: gym.Space

    def __init__(
        self,
        backend: Optional[Backend] = None,
        frequency: Optional[float] = 200.0,
        frequency_checks: bool = True,
        init_state: Optional[RobotState] = None,
        regulate_frequency: bool = True,
        max_gain_scale: float = 5.0,
        fall_pitch: float = 1.0,
        shm_name: str = "/vulp",  # Added for compatibility with simple init
        **kwargs,
    ) -> None:
        r"""!
        Initialize servos environment.

        \param backend Backend for interfacing with a simulator or a spine.
        \param frequency Regulated frequency of the control loop, in Hz.
        \param frequency_checks Warning if loop is slower than frequency.
        \param init_state Initial state of the robot.
        \param regulate_frequency If set, env regulates loop frequency.
        \param max_gain_scale Maximum value for kp or kd gain scales.
        \param fall_pitch Pitch angle (rad) at which episode terminates.
        \param shm_name Shared memory name (if backend is not provided).
        """
        if not (0.0 < max_gain_scale < 10.0):
            raise UpkieRuntimeError(f"Invalid value {max_gain_scale=}")

        self.fall_pitch = fall_pitch

        # Handle backend/shm_name compatibility for simple gym.make usage
        # If the parent UpkieEnv supports shm_name directly, we pass it.
        # Otherwise, standard UpkieEnv usage often relies on backend.
        # We pass kwargs to be safe if the parent signature varies.
        init_kwargs = dict(
            frequency=frequency,
            frequency_checks=frequency_checks,
            init_state=init_state,
            regulate_frequency=regulate_frequency,
            **kwargs
        )
        
        # If Backend class exists and backend arg is provided, pass it
        if Backend is not None and backend is not None:
            init_kwargs["backend"] = backend
        elif "shm_name" in UpkieEnv.__init__.__code__.co_varnames:
             # Fallback for the simplified UpkieEnv defined earlier
            init_kwargs["shm_name"] = shm_name

        super().__init__(**init_kwargs)

        # Initialize action and observation spaces
        (
            action_space,
            observation_space,
            neutral_action,
            max_action,
            min_action,
        ) = self.__create_servo_spaces(max_gain_scale)

        self.action_space = action_space
        self.observation_space = observation_space
        self._max_action = max_action
        self._min_action = min_action
        self._neutral_action = neutral_action

    def __create_servo_spaces(self, max_gain_scale: float):
        action_space = {}
        neutral_action = {}
        max_action = {}
        min_action = {}
        servo_space = {}

        for joint in self.model.joints:
            action_space[joint.name] = gym.spaces.Dict(
                {
                    "position": gym.spaces.Box(
                        low=joint.limit.lower,
                        high=joint.limit.upper,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                    "velocity": gym.spaces.Box(
                        low=-joint.limit.velocity,
                        high=+joint.limit.velocity,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                    "feedforward_torque": gym.spaces.Box(
                        low=-joint.limit.effort,
                        high=+joint.limit.effort,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                    "kp_scale": gym.spaces.Box(
                        low=0.0,
                        high=max_gain_scale,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                    "kd_scale": gym.spaces.Box(
                        low=0.0,
                        high=max_gain_scale,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                    "maximum_torque": gym.spaces.Box(
                        low=0.0,
                        high=joint.limit.effort,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                }
            )
            servo_space[joint.name] = gym.spaces.Dict(
                {
                    "position": gym.spaces.Box(
                        low=joint.limit.lower,
                        high=joint.limit.upper,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                    "velocity": gym.spaces.Box(
                        low=-joint.limit.velocity,
                        high=+joint.limit.velocity,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                    "torque": gym.spaces.Box(
                        low=-joint.limit.effort,
                        high=+joint.limit.effort,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                    "temperature": gym.spaces.Box(
                        low=0.0,
                        high=100.0,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                    "voltage": gym.spaces.Box(
                        low=10.0,
                        high=44.0,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                }
            )
            neutral_action[joint.name] = {
                "position": np.nan,
                "velocity": 0.0,
                "feedforward_torque": 0.0,
                "kp_scale": 1.0,
                "kd_scale": 1.0,
                "maximum_torque": joint.limit.effort,
            }
            max_action[joint.name] = {
                "position": joint.limit.upper,
                "velocity": joint.limit.velocity,
                "feedforward_torque": joint.limit.effort,
                "kp_scale": max_gain_scale,
                "kd_scale": max_gain_scale,
                "maximum_torque": joint.limit.effort,
            }
            min_action[joint.name] = {
                "position": joint.limit.lower,
                "velocity": -joint.limit.velocity,
                "feedforward_torque": -joint.limit.effort,
                "kp_scale": 0.0,
                "kd_scale": 0.0,
                "maximum_torque": 0.0,
            }

        return (
            gym.spaces.Dict(action_space),
            gym.spaces.Dict(servo_space),
            neutral_action,
            max_action,
            min_action,
        )

    def get_env_observation(self, spine_observation: dict) -> dict:
        return {
            joint.name: {
                key: np.array(
                    [spine_observation["servo"][joint.name][key]],
                    dtype=np.float32,
                )
                for key in self.observation_space[joint.name]
            }
            for joint in self.model.joints
        }

    def get_neutral_action(self) -> dict:
        return self._neutral_action.copy()

    def get_spine_action(self, env_action: dict) -> dict:
        spine_action = {"servo": {}}
        for joint in self.model.joints:
            servo_action = {}
            for key in self.ACTION_KEYS:
                action = (
                    env_action[joint.name][key]
                    if key in env_action[joint.name]
                    else self._neutral_action[joint.name][key]
                )
                action_value = (
                    action.item()
                    if isinstance(action, np.ndarray)
                    else float(action)
                )
                servo_action[key] = clamp_and_warn(
                    action_value,
                    self._min_action[joint.name][key],
                    self._max_action[joint.name][key],
                    label=f"{joint.name}: {key}",
                )
            spine_action["servo"][joint.name] = servo_action
        return spine_action

    def step(self, action):
        # 1. Step the base environment
        _, _, terminated, truncated, info = super().step(action)
        
        # 2. Extract observations
        spine_observation = info["spine_observation"]
        env_obs = self.get_env_observation(spine_observation)
        
        # 3. Calculate State Variables
        # Pitch (Angle)
        pitch = spine_observation["base_orientation"]["pitch"]
        
        # Angular Velocity (Speed of rotation)
        # Note: Check your keys. Usually it's under 'imu' -> 'angular_velocity' 
        # If not available, you might need to calculate (pitch - last_pitch) / dt
        imu = spine_observation.get("imu", {})
        angular_velocity = imu.get("angular_velocity", [0, 0, 0])
        pitch_velocity = angular_velocity[1] # Y-axis is usually pitch

        # 4. Calculate Reward Components
        
        # A: Upright reward (Gaussian is good, maybe slightly tighter)
        reward_upright = np.exp(-10.0 * pitch**2)
        
        # B: Damping reward (Penalize spinning fast)
        # This prevents the "whiplash" effect.
        reward_damping = np.exp(-1.0 * pitch_velocity**2)
        
        # C: Survival Bonus (Small constant reward just for staying alive)
        # This encourages the agent to avoid the termination state.
        reward_survival = 0.1

        # D: Small Penalty for Action Magnitude (Energy efficiency)
        # Penalize high torques to prevent jittering
        # We iterate over joints to sum the squared commanded torque or velocity
        action_penalty = 0.0
        for joint in action:
            # Assuming 'feedforward_torque' or 'velocity' is used
            if "velocity" in action[joint]:
                action_penalty += float(action[joint]["velocity"])**2
        
        # Combine them
        # We weight uprightness highest.
        reward = (1.0 * reward_upright) + (0.5 * reward_damping) + reward_survival - (0.01 * action_penalty)

        # 5. Termination (Fall Detection)
        if abs(pitch) > self.fall_pitch:
            terminated = True
            reward = -1.0  # Hard penalty for failure

        return env_obs, reward, terminated, truncated, info