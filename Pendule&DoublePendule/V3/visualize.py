from stable_baselines3 import PPO
import pygame
import numpy as np
import math
from custom_env import SwingUpCartPoleEnv # Important !

import os
import glob

# --- CONFIGURATION ---
def get_latest_model():
    # Chemin absolu du dossier V3
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Chemins prioritaires (V3 avec sous-dossiers, puis V2)
    search_dirs = [
        os.path.join(current_dir, "models", "PPO_SwingUp_V3", "**", "*.zip"),
        os.path.join(current_dir, "models", "PPO_SwingUp_V2", "*.zip"),
    ]
    
    all_files = []
    for pattern in search_dirs:
        # Recursive globbing pour trouver dans les sous-dossiers run_...
        found = glob.glob(pattern, recursive=True)
        all_files.extend(found)

    if not all_files:
        return None

    # On trie par date de modification du fichier pour avoir vraiment le dernier touché
    latest = max(all_files, key=os.path.getmtime)
    
    print(f"Modèle chargé automatiquement : {latest}")
    return latest

MODEL_PATH = get_latest_model()

if MODEL_PATH is None:
    print("ERREUR: Aucun modèle (.zip) trouvé dans les dossiers models/PPO_SwingUp_V3 ou V2.")
    print(f"Dossier actuel : {os.getcwd()}")
    # Fallback par défaut pour éviter l'erreur de variable non définie
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "PPO_SwingUp_V3", "500000.zip") 

# Couleurs & Setup
WHITE, BLACK, RED, GREY, BLUE, GREEN = (255, 255, 255), (0, 0, 0), (200, 50, 50), (150, 150, 150), (50, 50, 200), (50, 200, 50)
WIDTH, HEIGHT = 1200, 400
SCALE = 80 

def draw_arrow(screen, color, start, end, width=3):
    start_int = (int(start[0]), int(start[1]))
    end_int = (int(end[0]), int(end[1]))
    pygame.draw.line(screen, color, start_int, end_int, width)
    rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
    p1 = (int(end[0]+15*math.sin(math.radians(rotation))), int(end[1]+15*math.cos(math.radians(rotation))))
    p2 = (int(end[0]+15*math.sin(math.radians(rotation-120))), int(end[1]+15*math.cos(math.radians(rotation-120))))
    p3 = (int(end[0]+15*math.sin(math.radians(rotation+120))), int(end[1]+15*math.cos(math.radians(rotation+120))))
    pygame.draw.polygon(screen, color, (p1, p2, p3))

def run_visualizer():
    # 1. Charger l'env Custom
    env = SwingUpCartPoleEnv()
    
    try:
        model = PPO.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Modèle introuvable : {MODEL_PATH}")
        return

    # 2. Pygame Init
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CartPole Swing-Up (360°)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    
    obs, _ = env.reset()
    running = True
    is_dragging = False
    mouse_pos = (0,0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEBUTTONDOWN: is_dragging = True
            elif event.type == pygame.MOUSEBUTTONUP: is_dragging = False
            elif event.type == pygame.MOUSEMOTION: mouse_pos = event.pos

        # --- IA ---
        action, _ = model.predict(obs, deterministic=True)
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extraction des données réelles de l'état interne (plus précis pour l'affichage)
        real_state = env.state 
        cart_x = real_state[0]
        pole_angle = real_state[2] # Vrai angle theta
        
        # Reset si le chariot sort de l'écran (mais pas si le pendule tourne !)
        if terminated:
            obs, _ = env.reset()
            is_dragging = False

        # --- DESSIN ---
        screen.fill(WHITE)
        base_x, base_y = WIDTH // 2, HEIGHT // 2 + 50
        
        # Positions
        screen_cart_x = base_x + (cart_x * SCALE)
        screen_cart_y = base_y
        pole_len = 120
        # Math: angle 0 est vertical haut. 
        pole_end_x = screen_cart_x + pole_len * math.sin(pole_angle)
        pole_end_y = screen_cart_y - pole_len * math.cos(pole_angle)

        cx, cy = int(screen_cart_x), int(screen_cart_y)
        px, py = int(pole_end_x), int(pole_end_y)

        # Interaction Souris (Force Élastique)
        if is_dragging:
            mx, my = mouse_pos
            dx = mx - px
            # Injection de vélocité
            env.unwrapped.state[3] += dx * 0.005
            # Update obs (nécessaire car l'obs dépend de state)
            obs[4] = env.unwrapped.state[3] 
            
            pygame.draw.line(screen, GREEN, (px, py), mouse_pos, 2)
            pygame.draw.circle(screen, GREEN, mouse_pos, 5)

        # Chariot
        pygame.draw.rect(screen, GREY, (cx-40, cy-20, 80, 40))
        pygame.draw.rect(screen, BLACK, (cx-40, cy-20, 80, 40), 2)
        pygame.draw.circle(screen, BLACK, (cx-25, cy+20), 10)
        pygame.draw.circle(screen, BLACK, (cx+25, cy+20), 10)

        # Sol
        pygame.draw.line(screen, BLACK, (0, base_y+15), (WIDTH, base_y+15), 2)

        # Pendule
        pygame.draw.line(screen, BLACK, (cx, cy), (px, py), 6)
        bob_col = GREEN if is_dragging else RED
        pygame.draw.circle(screen, bob_col, (px, py), 15)

        # Fleche Action
        if action == 1: draw_arrow(screen, BLUE, (cx, cy), (cx+60, cy))
        else: draw_arrow(screen, BLUE, (cx, cy), (cx-60, cy))

        # Texte
        txt = font.render(f"Angle: {math.degrees(pole_angle):.1f}°", True, BLACK)
        screen.blit(txt, (20, 20))
        if is_dragging: screen.blit(font.render("PERTURBATION ACTIVE", True, GREEN), (20, 50))

        pygame.display.flip()
        clock.tick(50)

    pygame.quit()
    env.close()

if __name__ == "__main__":
    run_visualizer()