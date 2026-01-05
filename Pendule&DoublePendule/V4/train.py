import os
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from swingup_env import SwingUpCartPoleEnv

def get_latest_checkpoint(base_models_dir):
    """Trouve le fichier .zip le plus récent dans tous les sous-dossiers de base_models_dir."""
    checkpoint_files = glob.glob(os.path.join(base_models_dir, "**", "*.zip"), recursive=True)
    if not checkpoint_files:
        return None, None
    
    latest_file = max(checkpoint_files, key=os.path.getmtime)
    # Extraire le run_id (nom du dossier parent)
    run_id = os.path.basename(os.path.dirname(latest_file))
    return latest_file, run_id

def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_models_dir = os.path.join(current_dir, "models", "PPO_SwingUp_V4")
    base_log_dir = os.path.join(current_dir, "logs_swingup_v4")

    # 1. Tenter de reprendre un entraînement
    latest_checkpoint, run_id = get_latest_checkpoint(base_models_dir)
    
    env = SwingUpCartPoleEnv()
    
    if latest_checkpoint:
        print(f"Reprise de l'entraînement : {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env)
        # On extrait le nombre de pas déjà effectués à partir du nom du fichier (ex: 50000.zip)
        try:
            start_step = int(os.path.basename(latest_checkpoint).replace(".zip", ""))
        except ValueError:
            start_step = 0
    else:
        import datetime
        run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        print(f"Nouvel entraînement : {run_id}")
        
        log_dir = os.path.join(base_log_dir, run_id)
        os.makedirs(log_dir, exist_ok=True)
        
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
            ent_coef=0.01
        )
        start_step = 0

    # Dossiers de sauvegarde pour ce run_id
    models_dir = os.path.join(base_models_dir, run_id)
    log_dir = os.path.join(base_log_dir, run_id)
    os.makedirs(models_dir, exist_ok=True)

    # 2. Boucle d'entraînement
    TIMESTEPS = 25000
    TOTAL_STEPS = 1000000
    
    # Calculer combien de boucles il reste à faire
    current_loop = (start_step // TIMESTEPS) + 1
    total_loops = TOTAL_STEPS // TIMESTEPS

    print(f"Début/Reprise à l'étape {start_step} (Boucle {current_loop}/{total_loops})")
    
    try:
        for i in range(current_loop, total_loops + 1):
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
            save_path = os.path.join(models_dir, f"{TIMESTEPS * i}")
            model.save(save_path)
            print(f"Sauvegarde effectuée : {save_path}.zip")
    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur. Sauvegarde en cours...")
        model.save(os.path.join(models_dir, "interrupted_model"))
        print("Sauvegardé sous 'interrupted_model.zip'.")

    env.close()

if __name__ == "__main__":
    train()