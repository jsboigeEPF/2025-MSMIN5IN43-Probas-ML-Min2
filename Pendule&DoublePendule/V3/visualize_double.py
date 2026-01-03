from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pygame
import numpy as np
import math
import glob
import os
from double_pendulum_env import DoubleSwingUpCartPoleEnv

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1200, 600
SCALE = 100 

def get_latest_model_dir():
    # Cherche le dernier dossier créé dans models/PPO_DoublePendulum
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "models", "PPO_DoublePendulum")
        
    if not os.path.exists(base_dir): return None

    # Trouve tous les sous-dossiers (double_run_...)
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs: return None
    
    # Prend le plus récent
    latest_dir = max(subdirs, key=os.path.getmtime)
    return latest_dir

def get_latest_files(model_dir):
    # Trouve le dernier zip et le dernier env.pkl
    zips = glob.glob(os.path.join(model_dir, "*.zip"))
    if not zips: return None, None
    
    latest_zip = max(zips, key=lambda x: int(os.path.basename(x).split('_')[0]) if os.path.basename(x).split('_')[0].isdigit() else (int(os.path.basename(x).split('.')[0]) if os.path.basename(x).split('.')[0].isdigit() else 0))
    
    # Le fichier env correspondant
    base_name = os.path.splitext(latest_zip)[0]
    env_path = base_name + "_env.pkl"
    
    return latest_zip, env_path

def run_visualizer():
    model_dir = get_latest_model_dir()
    if not model_dir:
        print("Aucun dossier d'entraînement trouvé.")
        return

    model_path, env_path = get_latest_files(model_dir)
    if not model_path:
        print("Aucun modèle trouvé.")
        return

    print(f"Chargement : {model_path}")
    
    # 1. Recréer l'env de base
    env = DummyVecEnv([lambda: DoubleSwingUpCartPoleEnv()])
    
    # 2. Charger les stats de normalisation (Moyenne/Variance) apprises
    if os.path.exists(env_path):
        print(f"Chargement normalisation : {env_path}")
        env = VecNormalize.load(env_path, env)
        env.training = False # IMPORTANT: Ne pas mettre à jour les stats en test
        env.norm_reward = False # Pas besoin de normaliser la reward pour l'affichage
    else:
        print("ATTENTION: Pas de fichier de normalisation trouvé. Le modèle risque de faire n'importe quoi.")

    model = PPO.load(model_path)

    # Pygame Init
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Double Pendulum CartPole (Balance + Perturbation)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    
    obs = env.reset()
    running = True
    
    # Interaction
    is_dragging = False
    mouse_pos = (0,0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEBUTTONDOWN: is_dragging = True
            elif event.type == pygame.MOUSEBUTTONUP: is_dragging = False
            elif event.type == pygame.MOUSEMOTION: mouse_pos = event.pos

        # IA
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        # Récupération état réel (non normalisé) pour affichage
        # L'environnement est wrappé, on doit descendre pour trouver l'env original
        real_env = env.envs[0]
        state = real_env.state
        
        x = state[0]
        th1 = state[2]
        th2 = state[4]
        
        # Positions
        base_x, base_y = WIDTH // 2, HEIGHT // 2
        cx = base_x + (x * SCALE)
        cy = base_y
        p1_x = cx + math.sin(th1) * SCALE
        p1_y = cy - math.cos(th1) * SCALE 
        p2_x = p1_x + math.sin(th2) * SCALE
        p2_y = p1_y - math.cos(th2) * SCALE

        # --- PERTURBATION SOURIS ---
        if is_dragging:
            mx, my = mouse_pos
            # On applique la force sur le bout du 2ème pendule (le plus fun)
            dx = mx - p2_x
            
            # Injection de "force" directement dans la vélocité angulaire
            # C'est une approximation physique simple d'un "coup de doigt"
            strength = 0.05
            
            # Modifier l'état interne de l'environnement
            # state = [x, x_dot, th1, th1_dot, th2, th2_dot]
            # On booste th2_dot (index 5) proportionnellement à la distance horizontale
            real_env.state[5] += dx * strength * 0.1
            
            # On met aussi un peu sur le premier pendule pour le réalisme
            real_env.state[3] += dx * strength * 0.05

            # Feedback visuel (Ligne verte)
            pygame.draw.line(screen, (0, 255, 0), (int(p2_x), int(p2_y)), (mx, my), 2)
            pygame.draw.circle(screen, (0, 255, 0), (mx, my), 5)

        if dones[0]:
            obs = env.reset()
            is_dragging = False # Reset drag on fail

        # --- DESSIN ---
        screen.fill((255, 255, 255))
        
        # Casting Int pour Pygame (Important !)
        icx, icy = int(cx), int(cy)
        ip1x, ip1y = int(p1_x), int(p1_y)
        ip2x, ip2y = int(p2_x), int(p2_y)

        # Dessin Chariot
        pygame.draw.rect(screen, (150, 150, 150), (icx-30, icy-15, 60, 30))
        
        # Dessin Tiges
        pygame.draw.line(screen, (0, 0, 0), (icx, icy), (ip1x, ip1y), 4)
        pygame.draw.line(screen, (0, 0, 0), (ip1x, ip1y), (ip2x, ip2y), 4)
        
        # Dessin Joints
        pygame.draw.circle(screen, (200, 50, 50), (ip1x, ip1y), 8)
        pygame.draw.circle(screen, (50, 50, 200), (ip2x, ip2y), 8)

        # Force (Action)
        force = action[0][0] # VecEnv retourne [[action]]
        start_arrow = (icx, icy + 30)
        end_arrow = (int(icx + force * 50), int(icy + 30))
        pygame.draw.line(screen, (0, 200, 0), start_arrow, end_arrow, 3)
        
        # Texte Perturbation
        if is_dragging:
            txt = font.render("PERTURBATION ACTIVE", True, (0, 200, 0))
            screen.blit(txt, (20, 20))

        pygame.display.flip()
        clock.tick(50)

    pygame.quit()
    env.close()

if __name__ == "__main__":
    run_visualizer()