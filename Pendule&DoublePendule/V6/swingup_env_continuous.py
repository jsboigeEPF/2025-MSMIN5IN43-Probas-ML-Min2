import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class SwingUpCartPoleEnvV6(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self):
        super(SwingUpCartPoleEnvV6, self).__init__()
        
        # Physics constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 20.0 # We can keep high power because the agent can choose to use less
        self.tau = 0.02 
        
        # Wide track
        self.x_threshold = 12.0 

        # --- MAJOR CHANGE: CONTINUOUS ACTION SPACE ---
        # Value between -1.0 (Full Left) and 1.0 (Full Right)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space
        high = np.array([
            self.x_threshold * 2, 
            np.finfo(np.float32).max, 
            1.0, 
            1.0, 
            np.finfo(np.float32).max
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        
        # Clip action to ensure valid range [-1, 1]
        force_input = np.clip(action[0], -1.0, 1.0)
        
        # Apply physics
        force = self.force_mag * force_input
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # --- REWARD FUNCTION (Anti-Helicopter + Vertical Magnet) ---
        
        # 1. Angle Reward: 1.0 if upright, -1.0 if down
        r_angle = math.cos(theta)
        
        # 2. Vertical Magnet (NEW): Sucks the agent towards perfect verticality
        # Raising cos(theta) to a high power creates a reward that is negligible 
        # unless we are VERY close to 1.0.
        # Example: 0.9^20 = 0.12, but 1.0^20 = 1.0.
        r_magnet = 0.0
        if r_angle > 0.0:
            r_magnet = 5.0 * (r_angle ** 20) 

        # 3. Angular Velocity Penalty (Adaptive)
        r_omega = 0.0
        if r_angle > 0.0: # Upper hemisphere
            r_omega = -0.1 * (theta_dot**2)
            if r_angle > 0.8:
                r_omega = -0.5 * (theta_dot**2) 
        
        # 4. Position Penalty (Keep Center)
        r_pos = -0.05 * (x**2)
        
        # 5. Action/Energy Penalty
        r_action = -0.01 * (force_input**2)
        
        # 6. Stability Bonus (Relaxed slightly to encourage latching)
        r_stability = 0.0
        if r_angle > 0.95 and abs(theta_dot) < 1.5: # Relaxed v from 1.0 to 1.5
            r_stability = 5.0

        reward = r_angle + r_magnet + r_omega + r_pos + r_action + r_stability
        
        # Add survival bonus
        reward += 1.0

        # Termination
        terminated = bool(x < -self.x_threshold or x > self.x_threshold)
        
        # Fail penalty
        if terminated:
            reward = -10.0

        obs = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)

        return obs, float(reward), terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize initialization to help exploration
        # Instead of strictly sticking to the bottom (PI), we initialize
        # anywhere in the lower hemisphere [PI/2, 3*PI/2] roughly.
        # This forces the agent to react to gravity immediately, not just "break static".
        
        low = np.array([-0.1, -0.1, np.pi - 1.0, -0.1])
        high = np.array([0.1, 0.1, np.pi + 1.0, 0.1])
        
        self.state = np.random.uniform(low=low, high=high, size=(4,))
        self.state = self.state.astype(np.float32)
        
        x, x_dot, theta, theta_dot = self.state
        obs = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)
        return obs, {}
