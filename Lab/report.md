Upkie Pendulum Environment

Overview

The UpkiePendulum environment wraps the Upkie wheeled biped robot to behave like a Wheeled Inverted Pendulum. The legs are kept rigid, and the agent controls the ground velocity of the wheels to maintain balance.

Observation Space

The observation is a 4-dimensional vector:

Pitch ($\theta$): The angle of the base relative to the vertical world Z-axis. (0 = upright).

Position ($p$): The odometry position of the wheels (0 = starting point).

Pitch Velocity ($\dot{\theta}$): The angular velocity of the base.

Ground Velocity ($\dot{p}$): The linear velocity of the wheels.

Action Space

The action is a 1-dimensional vector:

Commanded Velocity ($\dot{p}^*$): The target ground velocity for the wheel controllers.

Reward Function

The goal of the agent is to keep the robot upright and, secondarily, to keep it near the starting position. The reward function uses Gaussian kernels to provide a smooth, dense signal within the range $[0, \approx 1.2]$.

$$R = w_\theta \cdot \exp(-k_\theta \cdot \theta^2) + w_p \cdot \exp(-k_p \cdot p^2) + w_{\dot{\theta}} \cdot \exp(-k_{\dot{\theta}} \cdot \dot{\theta}^2)$$

Parameters:

Pitch Term ($w_\theta=1.0, k_\theta=15.0$): Strongly rewards keeping the pitch near 0. The kernel width corresponds to a standard deviation of roughly $\pm 0.25$ rad ($\approx 15^\circ$).

Position Term ($w_p=0.1, k_p=1.0$): Lightly rewards staying near the origin ($p=0$). This prevents the robot from stabilizing by simply driving off to infinity.

Velocity Term ($w_{\dot{\theta}}=0.1, k_{\dot{\theta}}=0.1$): Lightly rewards low angular velocity, encouraging smooth, static balancing rather than oscillating violently.

If a termination condition is met, the reward for that step is $0.0$.

Termination Conditions

The episode terminates (terminated = True) if:

Fall Detection: The pitch angle $|\theta|$ exceeds fall_pitch (default 1.0 rad).

Position Limit: The robot drives more than 3.0 meters away from the origin ($|p| > 3.0$).

The episode truncates (truncated = True) if:

Time Limit: The episode exceeds 1000 steps (approx. 5 seconds at 200Hz). This was increased from the original 300 steps to allow sufficient time for the agent to demonstrate stability.