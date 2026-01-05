import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class SwingUpCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self):
        super(SwingUpCartPoleEnv, self).__init__()
        
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 20.0 
        self.tau = 0.02 
        self.x_threshold = 15.0 

        self.action_space = spaces.Discrete(2)

        # Observation Space: [x, x_dot, cos(theta), sin(theta), theta_dot]
        high = np.array([
            self.x_threshold * 2, 
            np.finfo(np.float32).max, 
            1.2, 
            1.2, 
            np.finfo(np.float32).max
        ], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None

    def step(self, action):
        # Ensure state is unpacked as native floats for internal math
        x, x_dot, theta, theta_dot = self.state.tolist()
        
        force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # --- REWARD (Explicitly native Python floats) ---
        r_theta = math.cos(theta)
        
        # 2. Radical Position Penalty
        # Base penalty increased
        r_x = -0.2 * (x**2) 
        
        # "Zone of Death": If you go beyond 4 meters, massive penalty
        if abs(x) > 4.0:
            r_x -= 5.0
            
        # "Wrong Way" Penalty: If far (> 3m) AND moving away from center
        if abs(x) > 3.0 and (x * x_dot > 0):
            r_x -= 5.0

        # 3. Cart Velocity Penalty: Prevent "charging"
        r_cart_vel = -0.01 * (x_dot**2)
        
        # 4. Modified Velocity Penalty: Stricter when upright
        r_vel = 0.0
        if r_theta > 0.0:
            r_vel = -0.5 * (theta_dot**2) * r_theta

        # 5. Stability Bonus: ONLY IF NEAR CENTER
        r_stability = 0.0
        if r_theta > 0.95 and abs(theta_dot) < 2.0:
            if abs(x) < 2.0: # Condition sine qua non: stay in the middle
                r_stability = 10.0
            else:
                # No bonus at all if far away now. Radical.
                r_stability = 0.0

        reward = float(r_theta + r_x + r_cart_vel + r_vel + r_stability)

        terminated = bool(x < -self.x_threshold or x > self.x_threshold)
        
        # Observation must be float32 array
        obs = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)

        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        low, high = -0.1, 0.1
        self.state = np.random.uniform(low=low, high=high, size=(4,))
        self.state[2] += math.pi 
        self.state = self.state.astype(np.float32)

        x, x_dot, theta, theta_dot = self.state.tolist()
        obs = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)
        return obs, {}
