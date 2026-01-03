from stable_baselines3 import PPO
import os
from custom_env import SwingUpCartPoleEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
import datetime
from custom_env import SwingUpCartPoleEnv

# Génération d'un nom unique pour cet entraînement
run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

# Chemin absolu du dossier V3
current_dir = os.path.dirname(os.path.abspath(__file__))

models_dir = os.path.join(current_dir, "models", "PPO_SwingUp_V3", run_id)
log_dir = os.path.join(current_dir, "logs_swingup_v3", run_id)

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

env = SwingUpCartPoleEnv()

# Vérification de l'environnement (Bonne pratique pour éviter les bugs silencieux)
print(f"Vérification de l'environnement pour {run_id}...")
check_env(env)
print("Environnement OK.")

# Hyperparamètres ajustés pour le Swing-Up
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

TIMESTEPS = 25000
TOTAL_LOOPS = 40 # J'augmente à 1 Million de steps (le swing up est dur)

print(f"Début de l'entraînement V3 - ID: {run_id}")
for i in range(1, TOTAL_LOOPS + 1):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    print(f"Sauvegarde {i}/{TOTAL_LOOPS}")

print("Terminé !")
env.close()