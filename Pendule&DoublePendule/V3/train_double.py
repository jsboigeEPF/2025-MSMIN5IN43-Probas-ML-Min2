from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
import os
import datetime
import gymnasium as gym
from double_pendulum_env import DoubleSwingUpCartPoleEnv

# Génération d'un nom unique
run_id = datetime.datetime.now().strftime("double_run_%Y%m%d_%H%M%S")

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models", "PPO_DoublePendulum", run_id)
log_dir = os.path.join(current_dir, "logs_double", run_id)

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 1. Création de l'environnement de base
env = DoubleSwingUpCartPoleEnv()

# 2. Vérification rapide
print("Vérification de l'environnement Double Pendule...")
check_env(env)
print("Environnement OK.")

# 3. Wrapping pour la normalisation automatique (CRITIQUE pour le double pendule)
# VecNormalize va centrer et réduire les obs et les rewards, ce qui aide énormément le réseau.
env = DummyVecEnv([lambda: DoubleSwingUpCartPoleEnv()])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Hyperparamètres
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
    ent_coef=0.0 # Moins d'exploration forcée nécessaire si on démarre en haut
)

TIMESTEPS = 25000
TOTAL_LOOPS = 40 

print(f"Début de l'entraînement Double Pendule - ID: {run_id}")

for i in range(1, TOTAL_LOOPS + 1):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    
    # On sauvegarde le modèle ET les stats de normalisation
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    env.save(f"{models_dir}/{TIMESTEPS*i}_env.pkl") 
    
    print(f"Sauvegarde {i}/{TOTAL_LOOPS}")

print("Terminé !")
env.close()