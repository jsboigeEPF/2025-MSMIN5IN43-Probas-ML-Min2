import os
import glob
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from swingup_env_continuous import SwingUpCartPoleEnvV6

def get_latest_checkpoint(base_dir):
    checkpoint_files = glob.glob(os.path.join(base_dir, "**", "*.zip"), recursive=True)
    if not checkpoint_files:
        return None, None, None
    
    latest_file = max(checkpoint_files, key=os.path.getmtime)
    run_dir = os.path.dirname(latest_file)
    run_id = os.path.basename(run_dir)
    return latest_file, run_dir, run_id

def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_models_dir = os.path.join(current_dir, "models", "PPO_SwingUp_V6")
    base_log_dir = os.path.join(current_dir, "logs_swingup_v6")

    latest_checkpoint, run_dir, run_id = get_latest_checkpoint(base_models_dir)

    def make_env():
        return SwingUpCartPoleEnvV6()

    if latest_checkpoint:
        print(f"Reprise de l'entraînement : {run_id}")
        
        # REMOVED NORMALIZATION FOR DEBUGGING
        env = DummyVecEnv([make_env])
        # env = VecNormalize.load(stats_path, env) 
        
        model = PPO.load(latest_checkpoint, env=env)
        
        try:
            start_step = int(os.path.basename(latest_checkpoint).replace(".zip", ""))
        except:
            start_step = 0
    else:
        run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        print(f"Nouveau run : {run_id}")
        run_dir = os.path.join(base_models_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        log_dir = os.path.join(base_log_dir, run_id)
        os.makedirs(log_dir, exist_ok=True)
        
        # REMOVED NORMALIZATION FOR DEBUGGING
        env = DummyVecEnv([make_env])
        # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        model = PPO(
            "MlpPolicy", env, verbose=1, tensorboard_log=log_dir,
            learning_rate=0.0003, n_steps=2048, batch_size=64,
            use_sde=True
        )
        start_step = 0

    TIMESTEPS = 25000
    TOTAL_STEPS = 1000000
    current_loop = (start_step // TIMESTEPS) + 1
    total_loops = TOTAL_STEPS // TIMESTEPS

    try:
        for i in range(current_loop, total_loops + 1):
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
            steps = TIMESTEPS * i
            model.save(os.path.join(run_dir, f"{steps}"))
            # env.save(os.path.join(run_dir, "vec_normalize.pkl")) # No norm stats to save
            print(f"Sauvegarde étape {steps}")
    except KeyboardInterrupt:
        print("\nInterrompu. Sauvegarde...")
        model.save(os.path.join(run_dir, "interrupted"))
        # env.save(os.path.join(run_dir, "vec_normalize.pkl"))

    env.close()

if __name__ == "__main__":
    train()