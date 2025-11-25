Upkie Policy Training Report

This report details the design and results of training reinforcement learning policies for two distinct control tasks on the Upkie robot: the Wheeled Inverted Pendulum task and the Full Servos task.

Part 1: Wheeled Inverted Pendulum

Environment Design (upkie/envs/upkie_pendulum.py)

In this initial task, the robot's legs were rigidly locked in a straight configuration. The agent had control only over the wheels' ground velocity.

Observation Space: 4D vector $[\theta, p, \dot{\theta}, \dot{p}]$ (Pitch, Position, Pitch Velocity, Ground Velocity).

Action Space: 1D vector $[\dot{p}^*]$ (Commanded Ground Velocity).

Reward Function:
The reward was designed to stabilize the robot at the origin. We used Gaussian kernels to provide dense, bounded rewards:

$$R_{pendulum} = 1.0 \cdot e^{-15 \theta^2} + 0.1 \cdot e^{-1.0 p^2} + 0.1 \cdot e^{-0.1 \dot{\theta}^2}$$

This prioritized keeping the pitch $\theta$ near zero, while secondarily encouraging the robot to stay near $p=0$ and move smoothly.

Results

The PPO agent successfully learned to stabilize the pendulum. Training was efficient due to the low dimensionality of the action space. The agent learned to modulate wheel velocity to counteract gravity, effectively balancing indefinitely (up to the 1000 step limit).

Part 2: Upkie Servos (6-DOF)

Environment Design (upkie/envs/upkie_servos.py)

This task is significantly more challenging. The agent controls all 6 joints (hips, knees, wheels) directly.

Action Space: 6 joints $\times$ {position, velocity, torque, gains}. Wrapped to expose Velocity for wheels and Position for legs.

Observation Space: Full servo state (Position, Velocity) for all 6 joints.

Revised Reward Function

For the Servos task, the simple Gaussian used in the Pendulum task was insufficient. The robot has more degrees of freedom and can exhibit high-frequency vibrations or "thrashing" while staying upright. We refined the reward function as follows:

$$R_{servos} = w_1 R_{upright} + w_2 R_{damping} + w_3 R_{survival} - w_4 C_{action}$$

Upright Term ($w_1=1.0$): $e^{-10 \theta^2}$.

Goal: The primary objective remains keeping the base pitch zero.

Damping Term ($w_2=0.5$): $e^{-1 \dot{\theta}_{pitch}^2}$.

Reasoning: In the Servos task, the robot can oscillate violently while technically remaining "upright" on average. This term penalizes high pitch velocity, encouraging smooth, static balancing.

Survival Bonus ($w_3=0.1$): Constant $+0.1$ per step.

Reasoning: This provides a small, positive gradient just for avoiding the termination condition (falling). It helps in the early stages of training when the agent is struggling to find any stable state.

Action Penalty ($w_4=0.01$): $\sum (\text{velocity}_{cmd})^2$.

Reasoning: We penalize high commanded velocities to improve energy efficiency and prevent the "jittery" behavior often seen in RL policies on hardware.

Training Strategy

Algorithm: PPO (Stable Baselines3).

Duration: Increased to 1,000,000 timesteps to accommodate the higher dimensionality and difficulty.

Rate Limiting: Logging for rate-limiters was suppressed to prevent I/O bottlenecks during the CPU-intensive training process.

Expected Results

The addition of the damping and action penalty terms is expected to yield a policy that is not only stable but smooth. Instead of reacting frantically to every small deviation, the agent should learn to lock its knees (or maintain a steady crouch) and make micro-adjustments with the wheels, mimicking the efficiency of the Pendulum policy but learned from scratch with full authority.