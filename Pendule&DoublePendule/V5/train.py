import os
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from swingup_env_v5 import SwingUpCartPoleEnvV5

def train():
    # 1. Directories
    run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models", "PPO_SwingUp_V5", run_id)
    log_dir = os.path.join(current_dir, "logs_swingup_v5", run_id)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Env
    env = SwingUpCartPoleEnvV5()
    print(f"Checking environment V5...")
    check_env(env)
    print("Environment check passed.")

    # 3. Model
    # V5 Hyperparameters: adjusted for finer control
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir, 
        learning_rate=0.0003,
        n_steps=2048,   
        batch_size=64,
        n_epochs=10,
        gamma=0.99,     
        gae_lambda=0.95,
        ent_coef=0.01, # Exploration
        clip_range=0.2
    )

    # 4. Loop
    TIMESTEPS = 25000
    TOTAL_LOOPS = 40 # 1M steps

    print(f"Starting V5 Training (Multiplicative Reward) - ID: {run_id}")
    
    for i in range(1, TOTAL_LOOPS + 1):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        save_path = f"{models_dir}/{TIMESTEPS*i}"
        model.save(save_path)
        print(f"Saved {i}/{TOTAL_LOOPS} -> {save_path}")

    env.close()

if __name__ == "__main__":
    train()
