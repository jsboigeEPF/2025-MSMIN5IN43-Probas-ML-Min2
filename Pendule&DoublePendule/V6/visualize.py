from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pygame
import numpy as np
import math
import sys
import os
import glob
from swingup_env_continuous import SwingUpCartPoleEnvV6

# --- HELPERS ---
def get_latest_run_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "models", "PPO_SwingUp_V6")
    if not os.path.exists(base_dir): return None
    
    # Find all run directories
    runs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not runs: return None
    
    # Sort by time
    latest_run = max(runs, key=os.path.getmtime)
    return latest_run

def get_latest_model_in_run(run_dir):
    zips = glob.glob(os.path.join(run_dir, "*.zip"))
    if not zips: return None
    # Sort by timestep number in filename (e.g. 25000.zip)
    # Filter out non-numeric names if any
    valid_zips = []
    for z in zips:
        try:
            int(os.path.basename(z).replace(".zip", ""))
            valid_zips.append(z)
        except: pass
        
    if not valid_zips: return max(zips, key=os.path.getmtime)
    
    return max(valid_zips, key=lambda x: int(os.path.basename(x).replace(".zip", "")))

# Colors
WHITE, BLACK, RED, GREY, BLUE, GREEN = (255, 255, 255), (0, 0, 0), (200, 50, 50), (150, 150, 150), (50, 50, 200), (50, 200, 50)
WIDTH, HEIGHT = 1200, 600
SCALE = 80 

def draw_arrow_continuous(screen, color, start, end_max, force_ratio, width=3):
    # force_ratio is between -1 and 1
    # end_max is the pixel position for force=1.0
    
    start_x, start_y = start
    end_max_x, end_max_y = end_max
    
    # Calculate actual end point based on force
    # Length of max arrow
    max_len = end_max_x - start_x
    actual_len = max_len * abs(force_ratio)
    
    if force_ratio > 0:
        direction = 1
    else:
        direction = -1
        
    end_x = start_x + (actual_len * direction)
    end_y = start_y # Horizontal force only
    
    start_int = (int(start_x), int(start_y))
    end_int = (int(end_x), int(end_y))
    
    if abs(force_ratio) > 0.05:
        pygame.draw.line(screen, color, start_int, end_int, width)
        # Arrowhead
        rotation = math.degrees(math.atan2(start_y-end_y, end_x-start_x))+90
        p1 = (int(end_x+10*math.sin(math.radians(rotation))), int(end_y+10*math.cos(math.radians(rotation))))
        p2 = (int(end_x+10*math.sin(math.radians(rotation-120))), int(end_y+10*math.cos(math.radians(rotation-120))))
        p3 = (int(end_x+10*math.sin(math.radians(rotation+120))), int(end_y+10*math.cos(math.radians(rotation+120))))
        pygame.draw.polygon(screen, color, (p1, p2, p3))

def run_visualizer():
    # 1. Locate Model and Stats
    if len(sys.argv) > 1:
        # User provided specific zip. Assume vec_normalize.pkl is in same folder.
        model_path = sys.argv[1]
        run_dir = os.path.dirname(model_path)
    else:
        run_dir = get_latest_run_dir()
        if not run_dir:
            print("No run directory found in V6.")
            return
        model_path = get_latest_model_in_run(run_dir)
    
    stats_path = os.path.join(run_dir, "vec_normalize.pkl")
    
    print(f"Loading Model: {model_path}")
    print(f"Loading Stats: {stats_path}")

    # 2. Load Env and Stats
    # IMPORTANT: We must wrap the env exactly like in training
    env = SwingUpCartPoleEnvV6()
    env = DummyVecEnv([lambda: env])
    
    # NORMALIZATION REMOVED FOR DEBUGGING
    # try:
    #     # Load the normalization stats
    #     env = VecNormalize.load(stats_path, env)
    #     # Disable training mode for viz (don't update stats)
    #     env.training = False 
    #     env.norm_reward = False # We don't care about normalized reward for viz
    # except Exception as e:
    #     print(f"Warning: Could not load normalization stats ({e}). Visuals might be broken.")

    model = PPO.load(model_path)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CartPole V6 (Continuous Control)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    
    # VecEnv reset returns just obs
    obs = env.reset()
    
    running = True
    is_dragging = False
    mouse_pos = (0,0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEBUTTONDOWN: is_dragging = True
            elif event.type == pygame.MOUSEBUTTONUP: is_dragging = False
            elif event.type == pygame.MOUSEMOTION: mouse_pos = event.pos

        # Predict (Deterministic)
        action, _ = model.predict(obs, deterministic=True)
        
        # Step
        obs, rewards, dones, infos = env.step(action)
        
        # Access internal state from the UNWRAPPED env inside VecEnv
        real_env = env.envs[0]
        real_state = real_env.state 
        cart_x = real_state[0]
        pole_angle = real_state[2] 
        
        # Fix DeprecationWarning: Extract scalar properly
        if isinstance(action, np.ndarray):
            force_applied = float(action.item())
        else:
            force_applied = float(action)

        if dones[0]:
            obs = env.reset()
            is_dragging = False

        # --- DRAWING ---
        screen.fill(WHITE)
        
        # Camera
        camera_x = cart_x
        def to_screen(x, y):
            sx = int((x - camera_x) * SCALE + WIDTH // 2)
            sy = int(HEIGHT // 2 + 100 - y) 
            return sx, sy

        base_y_world = 0
        cx, cy = to_screen(cart_x, base_y_world)

        # Grid
        for i in range(int(cart_x - 10), int(cart_x + 10)):
            gx, gy = to_screen(i, base_y_world)
            pygame.draw.line(screen, GREY, (gx, gy-10), (gx, gy+10), 1)
        
        pygame.draw.line(screen, BLACK, (0, cy+15), (WIDTH, cy+15), 2)

        pole_len = 120
        px = cx + pole_len * math.sin(pole_angle)
        py = cy - pole_len * math.cos(pole_angle)

        if is_dragging:
            mx, my = mouse_pos
            dx = mx - px
            real_env.state[3] += dx * 0.005 
            # We need to re-normalize the obs if we hack the state, but PPO is robust enough for viz
            pygame.draw.line(screen, GREEN, (px, py), mouse_pos, 2)

        pygame.draw.rect(screen, GREY, (cx-40, cy-20, 80, 40))
        pygame.draw.rect(screen, BLACK, (cx-40, cy-20, 80, 40), 2)
        pygame.draw.circle(screen, BLACK, (cx-25, cy+20), 10)
        pygame.draw.circle(screen, BLACK, (cx+25, cy+20), 10)

        pygame.draw.line(screen, BLACK, (cx, cy), (px, py), 6)
        pygame.draw.circle(screen, RED, (int(px), int(py)), 15)

        # Force Arrow (Variable Length)
        # Max length 80px
        draw_arrow_continuous(screen, BLUE, (cx, cy), (cx+80, cy), force_applied)

        # Normalize angle for display (0-360)
        deg = math.degrees(pole_angle) % 360
        
        txt = font.render(f"Angle: {deg:.1f}° | Force: {force_applied:.2f}", True, BLACK)
        screen.blit(txt, (20, 20))
        
        pygame.display.flip()
        clock.tick(50)

    pygame.quit()
    env.close()

if __name__ == "__main__":
    run_visualizer()
