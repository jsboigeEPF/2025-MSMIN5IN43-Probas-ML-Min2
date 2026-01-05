from stable_baselines3 import PPO
import pygame
import numpy as np
import math
from swingup_env import SwingUpCartPoleEnv 

import os
import glob
import sys

# --- CONFIGURATION ---
def get_latest_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [os.path.join(current_dir, "models", "PPO_SwingUp_V4", "**", "*.zip")]
    
    all_files = []
    for pattern in search_dirs:
        found = glob.glob(pattern, recursive=True)
        all_files.extend(found)

    if not all_files:
        return None

    # Sort by modification time
    latest = max(all_files, key=os.path.getmtime)
    return latest

# Logic to select model: CLI argument or latest
if len(sys.argv) > 1:
    MODEL_PATH = sys.argv[1]
    print(f"Using specified model: {MODEL_PATH}")
else:
    MODEL_PATH = get_latest_model()
    print(f"Using latest model: {MODEL_PATH}")

# Colors
WHITE, BLACK, RED, GREY, BLUE, GREEN = (255, 255, 255), (0, 0, 0), (200, 50, 50), (150, 150, 150), (50, 50, 200), (50, 200, 50)
WIDTH, HEIGHT = 1200, 600
SCALE = 80 # We keep high scale but use camera tracking

def draw_arrow(screen, color, start, end, width=3):
    start_int = (int(start[0]), int(start[1]))
    end_int = (int(end[0]), int(end[1]))
    pygame.draw.line(screen, color, start_int, end_int, width)
    # Simple arrowhead
    rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
    p1 = (int(end[0]+15*math.sin(math.radians(rotation))), int(end[1]+15*math.cos(math.radians(rotation))))
    p2 = (int(end[0]+15*math.sin(math.radians(rotation-120))), int(end[1]+15*math.cos(math.radians(rotation-120))))
    p3 = (int(end[0]+15*math.sin(math.radians(rotation+120))), int(end[1]+15*math.cos(math.radians(rotation+120))))
    pygame.draw.polygon(screen, color, (p1, p2, p3))

def run_visualizer():
    env = SwingUpCartPoleEnv()
    
    if MODEL_PATH is None:
        print("Error: No model found in V4/models. Run train.py first.")
        return

    try:
        model = PPO.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CartPole Swing-Up V4 (Ambitious - Camera Follow)")
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

        # Predict
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        real_state = env.state 
        cart_x = real_state[0]
        pole_angle = real_state[2] 
        
        if terminated:
            obs, _ = env.reset()
            is_dragging = False

        # --- DRAWING WITH CAMERA TRACKING ---
        screen.fill(WHITE)
        
        # Camera Logic: Keep cart in center
        # World coordinate to Screen coordinate: 
        # ScreenX = (WorldX - CameraX) * SCALE + CenterX
        camera_x = cart_x
        
        def to_screen(x, y):
            sx = int((x - camera_x) * SCALE + WIDTH // 2)
            sy = int(HEIGHT // 2 + 100 - y) # Y is inverted (up is -y in pygame), +100 offset
            return sx, sy

        base_y_world = 0
        cx, cy = to_screen(cart_x, base_y_world)

        # Ground (Infinite-ish lines)
        # We draw vertical grid lines to show movement
        for i in range(int(cart_x - 10), int(cart_x + 10)):
            gx, gy = to_screen(i, base_y_world)
            pygame.draw.line(screen, GREY, (gx, gy-10), (gx, gy+10), 1) # Ticks
        
        pygame.draw.line(screen, BLACK, (0, cy+15), (WIDTH, cy+15), 2) # Main line

        # Pole End
        pole_len = 120 # Display length
        # Angle 0 is vertical UP in math, but in screen Y goes down.
        # Physics: x + L*sin(theta), y + L*cos(theta) (if y is up)
        # Pygame: y is down.
        px = cx + pole_len * math.sin(pole_angle)
        py = cy - pole_len * math.cos(pole_angle)

        # Interaction (Mouse Drag)
        if is_dragging:
            mx, my = mouse_pos
            dx = mx - px
            # Apply force to angular velocity for perturbation
            env.unwrapped.state[3] += dx * 0.005 
            obs[4] = env.unwrapped.state[3]
            
            pygame.draw.line(screen, GREEN, (px, py), mouse_pos, 2)

        # Draw Cart
        pygame.draw.rect(screen, GREY, (cx-40, cy-20, 80, 40))
        pygame.draw.rect(screen, BLACK, (cx-40, cy-20, 80, 40), 2)
        # Wheels
        pygame.draw.circle(screen, BLACK, (cx-25, cy+20), 10)
        pygame.draw.circle(screen, BLACK, (cx+25, cy+20), 10)

        # Draw Pole
        pygame.draw.line(screen, BLACK, (cx, cy), (px, py), 6)
        bob_col = GREEN if is_dragging else RED
        pygame.draw.circle(screen, bob_col, (int(px), int(py)), 15)

        # Force Arrow
        if action == 1: # Right
            draw_arrow(screen, BLUE, (cx, cy), (cx+60, cy))
        else: # Left
            draw_arrow(screen, BLUE, (cx, cy), (cx-60, cy))

        # HUD
        infos = [
            f"Angle: {math.degrees(pole_angle):.1f}°",
            f"Position: {cart_x:.2f} m",
            f"Reward: {reward:.3f}",
            "Camera Locked on Cart"
        ]
        
        for i, info_txt in enumerate(infos):
            txt = font.render(info_txt, True, BLACK)
            screen.blit(txt, (20, 20 + i*25))

        pygame.display.flip()
        clock.tick(50)

    pygame.quit()
    env.close()

if __name__ == "__main__":
    run_visualizer()
