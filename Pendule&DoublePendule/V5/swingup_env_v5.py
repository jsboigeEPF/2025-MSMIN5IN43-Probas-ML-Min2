import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class SwingUpCartPoleEnvV5(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self):
        super(SwingUpCartPoleEnvV5, self).__init__()
        
        # Physics constants (JohnBuffer/NEAT inspired - roughly standard)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # half-length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0 # Reduced force for finer control (was 20.0 in V4)
        self.tau = 0.02 
        
        # Wide track but we want to stay in center
        self.x_threshold = 10.0 

        # Discrete Action Space: Left, Right
        # Note: NEAT often uses continuous, but PPO Discrete is fine if we shape reward well.
        self.action_space = spaces.Discrete(2)

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
        
        # Physics
        force = self.force_mag if action == 1 else -self.force_mag
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

        # --- REWARD FUNCTION V5 (Physics & Energy Based) ---
        # Goal: Upright (cos=1) and Center (x=0) and Stable (vel=0)
        
        # 1. Verticality (0.0 to 1.0)
        # cos(theta): 1.0 (up), -1.0 (down). 
        # (cos + 1) / 2 ranges from 0 to 1.
        upright_score = (math.cos(theta) + 1.0) / 2.0
        
        # 2. Position Score (0.0 to 1.0)
        # Smooth Gaussian bell curve centered at 0.
        # sigma = 2.0 means at x=2.0 score is ~0.6, at x=4.0 score is ~0.13
        pos_score = math.exp(-(x / 2.0)**2)

        # 3. Stability Penalty (Energy minimization when at top)
        # We want small velocities.
        # Penalize kinetic energy: v^2
        penalty_spin = 0.05 * abs(theta_dot) # Linear penalty often trains better than quadratic for small values
        penalty_move = 0.05 * abs(x_dot)

        # TOTAL REWARD
        # We combine them multiplicatively for the core goal (must be upright AND centered)
        # Then subtract penalties.
        # This prevents the "fly away upright" bug because if pos_score is low, the whole reward is low.
        
        # Base reward: [0, 1]
        reward = (upright_score * pos_score)
        
        # Subtract penalties (only if we have some reasonable score, to avoid negative infinity loops)
        reward -= (penalty_spin + penalty_move) * 0.1
        
        # Bonus for "Solving" it (Perfectly still and upright in center)
        if upright_score > 0.98 and pos_score > 0.9 and abs(theta_dot) < 0.1:
            reward += 1.0

        # Survival constraint (soft)
        if abs(x) > self.x_threshold:
            terminated = True
            reward = -10.0 # Crash penalty
        else:
            terminated = False

        obs = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)

        return obs, float(reward), terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start hanging down (PI) with random perturbation
        low, high = -0.1, 0.1
        self.state = np.random.uniform(low=low, high=high, size=(4,))
        self.state[2] += math.pi 
        
        self.state = self.state.astype(np.float32)
        x, x_dot, theta, theta_dot = self.state
        obs = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)
        return obs, {}
