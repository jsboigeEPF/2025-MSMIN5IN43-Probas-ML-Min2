"""
Visualisations avancées pour le projet CFR Kuhn Poker
Génère des graphiques professionnels pour la présentation
Lance l'entraînement puis génère les visualisations basées sur les résultats
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from cfr_algorithm import CFRTrainer
from cfr_academic import compute_game_value
import os
import time

# Configuration du style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Dossier de sortie
OUTPUT_DIR = "d:\\Documents\\Ecole\\EPF\\5A EPF\\IA 2\\Poker\\figures"

# Variable globale pour stocker les résultats de l'entraînement
TRAINING_RESULTS = None


def ensure_output_dir():
    """Crée le dossier de sortie si nécessaire"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def run_main_training(iterations: int = 10000):
    """
    Lance l'entraînement principal et retourne les résultats
    Similaire à main.py mais retourne les données pour les visualisations
    """
    global TRAINING_RESULTS
    
    print("\n" + "="*70)
    print("   POKER AI - COUNTERFACTUAL REGRET MINIMIZATION (CFR)")
    print("="*70)
    print(f"\n   Jeu: Kuhn Poker")
    print(f"   Algorithme: CFR (Counterfactual Regret Minimization)")
    print(f"   Iterations: {iterations:,}")
    print("\n   Debut de l'entrainement...")
    
    start_time = time.time()
    
    # === Phase 1: Entraînement avec tracking de convergence ===
    trainer = CFRTrainer()
    
    # Collecter les données de convergence pendant l'entraînement
    checkpoints = 50
    checkpoint_interval = iterations // checkpoints
    
    convergence_data = {
        'iterations': [],
        'jack_bluffs': [],
        'queen_calls': [],
        'king_bets': [],
        'precisions': [],
        'game_values': []
    }
    
    print(f"\n   Entrainement avec {checkpoints} checkpoints...")
    
    for i in range(1, checkpoints + 1):
        # Entraînement
        for _ in range(checkpoint_interval):
            cards = [0, 1, 2]
            np.random.shuffle(cards)
            player_cards = cards[:2]
            trainer.cfr(player_cards, "", 1.0, 1.0)
            trainer.iterations += 1
        
        # Extraire les stratégies clés
        strategy = trainer.get_strategy_profile()
        jack = strategy.get('0', np.array([0.5, 0.5]))[1] * 100
        queen = strategy.get('1b', np.array([0.5, 0.5]))[1] * 100
        king = strategy.get('2', np.array([0.5, 0.5]))[1] * 100
        
        # Calcul précision (erreur relative par rapport à Nash)
        jack_error = abs(jack - 33.3) / 33.3 * 100  # Erreur en %
        queen_error = abs(queen - 33.3) / 33.3 * 100
        king_error = abs(king - 100.0) / 100.0 * 100
        avg_error = (jack_error + queen_error + king_error) / 3
        precision = max(0, 100 - avg_error)  # Précision = 100% - erreur moyenne
        
        # Calculer la game value
        game_value = compute_game_value(trainer.game, strategy)
        
        convergence_data['iterations'].append(i * checkpoint_interval)
        convergence_data['jack_bluffs'].append(jack)
        convergence_data['queen_calls'].append(queen)
        convergence_data['king_bets'].append(king)
        convergence_data['precisions'].append(precision)
        convergence_data['game_values'].append(game_value)
        
        # Afficher progression
        if i % 10 == 0:
            print(f"      [{i}/{checkpoints}] Precision: {precision:.1f}%, Game Value: {game_value:.6f}")
    
    training_time = time.time() - start_time
    
    # === Phase 2: Analyse finale ===
    strategy_profile = trainer.get_strategy_profile()
    
    jack_bet = strategy_profile.get('0', np.array([0.5, 0.5]))[1] * 100
    queen_call = strategy_profile.get('1b', np.array([0.5, 0.5]))[1] * 100
    king_bet = strategy_profile.get('2', np.array([0.5, 0.5]))[1] * 100
    
    # Calcul précision (erreur relative)
    jack_error = abs(jack_bet - 33.3) / 33.3 * 100
    queen_error = abs(queen_call - 33.3) / 33.3 * 100
    king_error = abs(king_bet - 100.0) / 100.0 * 100
    avg_error = (jack_error + queen_error + king_error) / 3
    overall_accuracy = max(0, 100 - avg_error)
    
    # Stocker tous les résultats
    TRAINING_RESULTS = {
        'trainer': trainer,
        'strategy_profile': strategy_profile,
        'convergence_data': convergence_data,
        'final_metrics': {
            'jack_bet': jack_bet,
            'queen_call': queen_call,
            'king_bet': king_bet,
            'jack_error': jack_error,
            'queen_error': queen_error,
            'king_error': king_error,
            'overall_accuracy': overall_accuracy,
        },
        'training_time': training_time,
        'iterations': iterations,
        'speed': iterations / training_time
    }
    
    # === Afficher les résultats ===
    print(f"\n" + "="*70)
    print("   RESULTATS DE L'ENTRAINEMENT")
    print("="*70)
    
    print(f"\n   Temps d'entrainement: {training_time:.2f} secondes")
    print(f"   Vitesse: {iterations/training_time:.0f} iterations/seconde")
    
    print(f"\n   Precision des strategies vs Nash theorique:")
    # Calculer la game value finale
    final_game_value = compute_game_value(trainer.game, strategy_profile)
    nash_value = -1/18  # Valeur théorique Nash (convention académique)
    game_value_error = abs(final_game_value - nash_value)
    
    print(f"      Jack bluff:      {jack_bet:5.1f}% (theorie: 33.3%) -> erreur {jack_error:.1f}%")
    print(f"      Queen call:      {queen_call:5.1f}% (theorie: 33.3%) -> erreur {queen_error:.1f}%")  
    print(f"      King value bet:  {king_bet:5.1f}% (theorie: 100%)  -> erreur {king_error:.1f}%")
    print(f"\n      Precision globale: {overall_accuracy:.1f}%")
    print(f"\n   Game Value:")
    print(f"      Valeur apprise:  {final_game_value:.6f}")
    print(f"      Valeur Nash:     {nash_value:.6f} (-1/18)")
    print(f"      Difference:      {game_value_error:.6f}")
    
    if overall_accuracy >= 99.5:
        quality = "EXCELLENT - Convergence quasi-parfaite vers Nash"
    elif overall_accuracy >= 99.0:
        quality = "TRES BON - Convergence solide vers Nash"
    elif overall_accuracy >= 95.0:
        quality = "BON - Convergence satisfaisante"
    else:
        quality = "MOYEN - Necessite plus d'iterations"
    
    print(f"\n   Qualite: {quality}")
    
    return TRAINING_RESULTS


def generate_convergence_plot():
    """
    Génère un graphique de convergence élégant basé sur les résultats de l'entraînement
    """
    print("\n[1/4] Generation du graphique de convergence...")
    ensure_output_dir()
    
    data = TRAINING_RESULTS['convergence_data']
    iterations = data['iterations']
    jack_bluffs = data['jack_bluffs']
    queen_calls = data['queen_calls']
    king_bets = data['king_bets']
    precisions = data['precisions']
    max_iterations = TRAINING_RESULTS['iterations']
    checkpoint_interval = iterations[1] - iterations[0] if len(iterations) > 1 else 200
    
    # Créer la figure avec 2 graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    
    # === Graphique 1: Convergence des stratégies ===
    ax1.plot(iterations, jack_bluffs, 'r-', linewidth=2.5, label='Jack bluff %', marker='o', markersize=4)
    ax1.plot(iterations, queen_calls, 'b-', linewidth=2.5, label='Queen call %', marker='s', markersize=4)
    ax1.plot(iterations, king_bets, 'g-', linewidth=2.5, label='King bet %', marker='^', markersize=4)
    
    # Lignes de référence Nash
    ax1.axhline(y=33.3, color='purple', linestyle='--', alpha=0.7, linewidth=2, label='Nash optimal (33.3%)')
    ax1.axhline(y=100, color='darkgreen', linestyle='--', alpha=0.7, linewidth=2, label='Nash optimal (100%)')
    
    ax1.set_xlabel('Nombre d\'iterations', fontweight='bold')
    ax1.set_ylabel('Probabilite d\'action (%)', fontweight='bold')
    ax1.set_title('CONVERGENCE DES STRATEGIES VERS L\'EQUILIBRE DE NASH', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='center right', fontsize=10)
    ax1.set_ylim(-5, 110)
    ax1.grid(True, alpha=0.3)
    
    # Ajouter zone de convergence
    ax1.fill_between(iterations, 30, 36.6, alpha=0.1, color='purple')
    
    # === Graphique 2: Précision globale ===
    colors = ['#ff6b6b' if p < 95 else '#ffd93d' if p < 99 else '#6bcb77' for p in precisions]
    ax2.bar(iterations, precisions, width=checkpoint_interval*0.8, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=100, color='green', linestyle='-', linewidth=3, alpha=0.8, label='Nash parfait (100%)')
    ax2.axhline(y=99, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Excellent (99%)')
    ax2.axhline(y=95, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Bon (95%)')
    
    ax2.set_xlabel('Nombre d\'iterations', fontweight='bold')
    ax2.set_ylabel('Precision vs Nash (%)', fontweight='bold')
    ax2.set_title('PRECISION GLOBALE DE LA STRATEGIE APPRISE', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim(85, 101)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Ajouter la valeur finale
    final_precision = precisions[-1]
    ax2.annotate(f'{final_precision:.1f}%', 
                xy=(iterations[-1], final_precision), 
                xytext=(iterations[-1] - max_iterations*0.15, final_precision - 3),
                fontsize=14, fontweight='bold', color='darkgreen',
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, '1_convergence.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"      -> {filepath}")


def generate_strategy_comparison():
    """
    Génère un graphique comparant stratégie apprise vs Nash théorique
    """
    print("[2/4] Generation du graphique de comparaison des strategies...")
    ensure_output_dir()
    
    metrics = TRAINING_RESULTS['final_metrics']
    
    # Données
    categories = ['Jack\n(Bluff)', 'Queen\n(Call)', 'King\n(Value Bet)']
    nash_values = [33.3, 33.3, 100.0]
    learned_values = [metrics['jack_bet'], metrics['queen_call'], metrics['king_bet']]
    
    x = np.arange(len(categories))
    width = 0.35  # Plus large pour 2 barres
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 2 séries de barres
    bars1 = ax.bar(x - width/2, nash_values, width, label='Nash Theorique', 
                   color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, learned_values, width, label='CFR Appris', 
                   color='#27ae60', edgecolor='black', linewidth=1.5)
    
    # Ajouter les valeurs sur les barres
    for bar, val in zip(bars1, nash_values):
        ax.annotate(f'{val:.1f}%', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=11, fontweight='bold', color='#2980b9')
    
    for bar, val in zip(bars2, learned_values):
        ax.annotate(f'{val:.1f}%', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=11, fontweight='bold', color='#1e8449')
    
    ax.set_ylabel('Probabilite d\'action (%)', fontweight='bold', fontsize=13)
    ax.set_title('COMPARAISON: NASH THEORIQUE vs CFR APPRIS', 
                fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ajouter une annotation explicative
    ax.text(0.5, -0.12, f'Precision globale: {metrics["overall_accuracy"]:.1f}%', 
           transform=ax.transAxes, fontsize=13, fontweight='bold',
           ha='center', color='darkgreen',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, '2_strategy_comparison.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"      -> {filepath}")


def generate_game_value_plot():
    """
    Génère un graphique montrant la convergence de la game value vers Nash
    """
    print("[3/7] Generation du graphique de la game value...")
    ensure_output_dir()
    
    data = TRAINING_RESULTS['convergence_data']
    iterations = data['iterations']
    game_values = data['game_values']
    nash_value = -1/18  # Valeur théorique Nash pour Kuhn Poker (-0.0556)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Tracer la game value
    ax.plot(iterations, game_values, 'b-', linewidth=2.5, label='Game Value (CFR)', marker='o', markersize=4, alpha=0.9)
    
    # Ligne de référence Nash
    ax.axhline(y=nash_value, color='red', linestyle='--', linewidth=2.5, alpha=0.8, 
              label=f'Nash Equilibrium = {nash_value:.6f} (-1/18)')
    
    # Zone de convergence acceptable (±0.0002 pour être plus réaliste)
    convergence_tolerance = 0.0002
    ax.fill_between(iterations, nash_value - convergence_tolerance, nash_value + convergence_tolerance, 
                    alpha=0.25, color='green', label=f'Zone de convergence (±{convergence_tolerance:.4f})')
    
    # Définir les limites de l'axe Y pour mieux voir la convergence
    min_val = min(min(game_values), nash_value - 0.002)
    max_val = max(max(game_values), nash_value + 0.002)
    ax.set_ylim(min_val, max_val)
    
    ax.set_xlabel('Nombre d\'itérations', fontweight='bold', fontsize=13)
    ax.set_ylabel('Game Value (joueur 0)', fontweight='bold', fontsize=13)
    ax.set_title('CONVERGENCE DE LA GAME VALUE VERS L\'ÉQUILIBRE DE NASH\n(Convention académique: -1/18)', 
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Ajouter la valeur finale avec positionnement amélioré
    final_value = game_values[-1]
    final_error = abs(final_value - nash_value)
    ax.annotate(f'Valeur finale: {final_value:.6f}\nErreur: {final_error:.6f}', 
                xy=(iterations[-1], final_value), 
                xytext=(iterations[-1] * 0.6, max_val - 0.0005),
                fontsize=11, fontweight='bold', color='darkblue',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='blue', lw=2),
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=2, connectionstyle='arc3,rad=0.2'))
    
    # Ajouter une annotation pour Nash - mieux positionnée
    ax.text(iterations[len(iterations)//4], nash_value + 0.0003, 
            f'Nash: {nash_value:.6f}', 
            fontsize=11, fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='red', alpha=0.9))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, '3_game_value.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"      -> {filepath}")


def generate_behavior_analysis():
    """
    Génère un graphique montrant les comportements émergents avec témoin
    """
    print("[4/7] Generation de l'analyse des comportements emergents...")
    ensure_output_dir()
    
    strategy = TRAINING_RESULTS['strategy_profile']
    
    # Données des comportements
    behaviors = ['BLUFF\nJack bet', 'VALUE BET\nKing bet', 'CALL DEFENSIF\nQueen call', 
                'FOLD OPTIMAL\nJack fold pb']
    
    nash_pct = [33.3, 100, 33.3, 100]
    
    learned_pct = [
        strategy.get('0', np.array([0.5, 0.5]))[1] * 100,
        strategy.get('2', np.array([0.5, 0.5]))[1] * 100,
        strategy.get('1b', np.array([0.5, 0.5]))[1] * 100,
        strategy.get('0pb', np.array([0.5, 0.5]))[0] * 100,
    ]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(behaviors))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, nash_pct, width, label='Nash Optimal', 
                  color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, learned_pct, width, label='CFR Appris', 
                  color='#27ae60', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Frequence du comportement (%)', fontweight='bold', fontsize=12)
    ax.set_title('COMPORTEMENTS STRATEGIQUES: NASH OPTIMAL vs CFR APPRIS', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, fontsize=10, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ajouter valeurs
    for bar, val in zip(bars1, nash_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
               f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold', color='#2980b9')
    
    for bar, val in zip(bars2, learned_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
               f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold', color='#1e8449')
    
    # Message clé
    ax.text(0.5, -0.15, 'Le CFR decouvre automatiquement les comportements strategiques optimaux',
           transform=ax.transAxes, fontsize=11, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.8))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, '4_emergent_behaviors.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"      -> {filepath}")


def generate_summary_dashboard():
    """
    Génère un dashboard récapitulatif avec toutes les métriques clés
    """
    print("[5/7] Generation du dashboard recapitulatif...")
    ensure_output_dir()
    
    metrics = TRAINING_RESULTS['final_metrics']
    
    jack_bet = metrics['jack_bet']
    queen_call = metrics['queen_call']
    king_bet = metrics['king_bet']
    jack_error = metrics['jack_error']
    queen_error = metrics['queen_error']
    king_error = metrics['king_error']
    overall_acc = metrics['overall_accuracy']
    
    # Recalculer les précisions individuelles
    jack_acc = max(0, 100 - jack_error)
    queen_acc = max(0, 100 - queen_error)
    king_acc = max(0, 100 - king_error)
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # === Titre principal ===
    fig.suptitle('CFR KUHN POKER - TABLEAU DE BORD DES RESULTATS', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # === 1. Jauge de précision globale ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Dessiner la jauge
    theta = np.linspace(0, np.pi, 100)
    r = 0.4
    x_arc = 0.5 + r * np.cos(theta)
    y_arc = 0.3 + r * np.sin(theta)
    ax1.plot(x_arc, y_arc, 'lightgray', linewidth=20, solid_capstyle='round')
    
    # Partie remplie
    fill_angle = np.pi * overall_acc / 100
    x_fill = 0.5 + r * np.cos(np.linspace(0, fill_angle, 50))
    y_fill = 0.3 + r * np.sin(np.linspace(0, fill_angle, 50))
    color = '#27ae60' if overall_acc >= 99 else '#f39c12' if overall_acc >= 95 else '#e74c3c'
    ax1.plot(x_fill, y_fill, color=color, linewidth=18, solid_capstyle='round')
    
    ax1.text(0.5, 0.3, f'{overall_acc:.1f}%', fontsize=28, fontweight='bold', 
            ha='center', va='center', color=color)
    ax1.text(0.5, 0.85, 'PRECISION GLOBALE', fontsize=12, fontweight='bold', ha='center')
    
    # === 2. Métriques individuelles ===
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.axis('off')
    
    metrics_data = [
        ('Jack Bluff', f'{jack_bet:.1f}%', '33.3%', f'{jack_acc:.1f}%'),
        ('Queen Call', f'{queen_call:.1f}%', '33.3%', f'{queen_acc:.1f}%'),
        ('King Bet', f'{king_bet:.1f}%', '100%', f'{king_acc:.1f}%'),
    ]
    
    ax2.text(0.5, 0.95, 'DETAIL DES STRATEGIES CLES', fontsize=14, fontweight='bold', 
            ha='center', transform=ax2.transAxes)
    
    headers = ['Strategie', 'Appris', 'Nash', 'Precision']
    col_positions = [0.1, 0.35, 0.55, 0.75]
    
    for i, header in enumerate(headers):
        ax2.text(col_positions[i], 0.75, header, fontsize=11, fontweight='bold',
                transform=ax2.transAxes, color='darkblue')
    
    # Ligne de séparation (utiliser plot au lieu de axhline avec transform)
    ax2.plot([0.05, 0.95], [0.72, 0.72], color='gray', linewidth=1, transform=ax2.transAxes)
    
    for row, (name, learned, nash, acc) in enumerate(metrics_data):
        y_pos = 0.55 - row * 0.2
        ax2.text(col_positions[0], y_pos, name, fontsize=11, transform=ax2.transAxes)
        ax2.text(col_positions[1], y_pos, learned, fontsize=11, transform=ax2.transAxes,
                color='#e74c3c', fontweight='bold')
        ax2.text(col_positions[2], y_pos, nash, fontsize=11, transform=ax2.transAxes,
                color='#3498db', fontweight='bold')
        
        acc_val = float(acc.replace('%', ''))
        acc_color = '#27ae60' if acc_val >= 99 else '#f39c12' if acc_val >= 95 else '#e74c3c'
        ax2.text(col_positions[3], y_pos, acc, fontsize=11, transform=ax2.transAxes,
                color=acc_color, fontweight='bold')
    
    # === 3. Barres de comparaison ===
    ax3 = fig.add_subplot(gs[1, :2])
    strategies = ['Jack Bluff', 'Queen Call', 'King Bet']
    nash = [33.3, 33.3, 100]
    learned = [jack_bet, queen_call, king_bet]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    ax3.bar(x - width/2, nash, width, label='Nash Theorique', color='#3498db', edgecolor='black')
    ax3.bar(x + width/2, learned, width, label='CFR Appris', color='#e74c3c', edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies, fontweight='bold')
    ax3.set_ylabel('Probabilite (%)')
    ax3.set_title('COMPARAISON NASH vs CFR', fontweight='bold')
    ax3.legend()
    ax3.set_ylim(0, 115)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # === 4. Points clés ===
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    training_time = TRAINING_RESULTS['training_time']
    speed = TRAINING_RESULTS['speed']
    
    key_points = [
        f'[OK] {TRAINING_RESULTS["iterations"]:,} iterations',
        f'[OK] Temps: {training_time:.1f}s',
        f'[OK] Vitesse: {speed:.0f} it/sec',
        f'[OK] Precision: {overall_acc:.1f}%',
        '[OK] Bluff et Value Bet appris',
    ]
    
    ax4.text(0.5, 0.95, 'POINTS CLES', fontsize=12, fontweight='bold', 
            ha='center', transform=ax4.transAxes, color='darkgreen')
    
    for i, point in enumerate(key_points):
        ax4.text(0.05, 0.75 - i*0.15, point, fontsize=10, transform=ax4.transAxes)
    
    # === 5. Message de conclusion ===
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    conclusion = f"""CONCLUSION: L'algorithme CFR a appris avec succes l'equilibre de Nash du Kuhn Poker.

    * Le bot a decouvert SEUL des strategies de poker professionnel
      (bluff, value bet, call defensif)
    * Precision de {overall_acc:.1f}% par rapport a la solution theorique optimale
    * Preuve que CFR converge vers Nash sans connaissance prealable du jeu"""
    
    ax5.text(0.5, 0.5, conclusion, fontsize=12, ha='center', va='center',
            transform=ax5.transAxes,
            bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', 
                     edgecolor='darkgreen', linewidth=2, alpha=0.8))
    
    filepath = os.path.join(OUTPUT_DIR, '5_dashboard.png')
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"      -> {filepath}")


def generate_decision_tree():
    """
    Génère l'arbre de décision complet du Kuhn Poker
    """
    if TRAINING_RESULTS is None:
        print("   ERREUR: Pas de resultats d'entrainement disponibles")
        return
    
    print("   Generation de l'arbre de decision...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Couleurs
    color_terminal = '#d4edda'
    color_p0 = '#cfe2ff'
    color_p1 = '#fff3cd'
    
    # Fonction pour dessiner un nœud
    def draw_node(x, y, text, color, size=0.4):
        circle = mpatches.Circle((x, y), size, color=color, ec='black', lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold', zorder=4)
    
    # Fonction pour dessiner une arête
    def draw_edge(x1, y1, x2, y2, label, is_pass=True):
        style = 'solid' if is_pass else 'dashed'
        ax.plot([x1, x2], [y1, y2], 'k-', lw=2, linestyle=style, alpha=0.6, zorder=1)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.15, label, fontsize=10, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', lw=1), zorder=2)
    
    # Titre
    ax.text(5, 9.5, 'Arbre de Décision - Kuhn Poker', fontsize=18, weight='bold', ha='center')
    ax.text(5, 9, '3 cartes (J, Q, K) | 2 joueurs | Pass (—) ou Bet (···)', fontsize=11, ha='center', style='italic')
    
    # Racine - P0 décide
    draw_node(5, 7.5, 'P0\n(J/Q/K)', color_p0, 0.5)
    
    # Niveau 1: P0 Pass ou Bet
    # P0 Pass
    draw_node(2.5, 5.5, 'P1\n(J/Q/K)', color_p1, 0.4)
    draw_edge(5, 7.5, 2.5, 5.5, 'Pass', is_pass=True)
    
    # P0 Bet
    draw_node(7.5, 5.5, 'P1\n(J/Q/K)', color_p1, 0.4)
    draw_edge(5, 7.5, 7.5, 5.5, 'Bet', is_pass=False)
    
    # Niveau 2: Après P0 Pass
    # P0 Pass, P1 Pass -> Terminal
    draw_node(1.5, 3.5, 'pp\n±1', color_terminal, 0.35)
    draw_edge(2.5, 5.5, 1.5, 3.5, 'Pass', is_pass=True)
    ax.text(1.5, 2.8, 'Showdown', fontsize=8, ha='center', style='italic')
    
    # P0 Pass, P1 Bet -> P0 décide
    draw_node(3.5, 3.5, 'P0\n(J/Q/K)', color_p0, 0.35)
    draw_edge(2.5, 5.5, 3.5, 3.5, 'Bet', is_pass=False)
    
    # Niveau 2: Après P0 Bet
    # P0 Bet, P1 Pass -> Terminal
    draw_node(6.5, 3.5, 'bp\n+1', color_terminal, 0.35)
    draw_edge(7.5, 5.5, 6.5, 3.5, 'Pass', is_pass=True)
    ax.text(6.5, 2.8, 'P0 gagne', fontsize=8, ha='center', style='italic')
    
    # P0 Bet, P1 Bet -> Terminal
    draw_node(8.5, 3.5, 'bb\n±2', color_terminal, 0.35)
    draw_edge(7.5, 5.5, 8.5, 3.5, 'Bet', is_pass=False)
    ax.text(8.5, 2.8, 'Showdown', fontsize=8, ha='center', style='italic')
    
    # Niveau 3: Après P0 Pass, P1 Bet
    # P0 Pass, P1 Bet, P0 Pass -> Terminal
    draw_node(2.5, 1.5, 'pbp\n-1', color_terminal, 0.35)
    draw_edge(3.5, 3.5, 2.5, 1.5, 'Pass', is_pass=True)
    ax.text(2.5, 0.8, 'P1 gagne', fontsize=8, ha='center', style='italic')
    
    # P0 Pass, P1 Bet, P0 Bet -> Terminal
    draw_node(4.5, 1.5, 'pbb\n±2', color_terminal, 0.35)
    draw_edge(3.5, 3.5, 4.5, 1.5, 'Bet', is_pass=False)
    ax.text(4.5, 0.8, 'Showdown', fontsize=8, ha='center', style='italic')
    
    # Légende
    legend_y = 0.3
    ax.text(0.5, legend_y, '■', fontsize=20, color=color_p0, weight='bold')
    ax.text(1.2, legend_y, 'Décision P0', fontsize=10, va='center')
    
    ax.text(3, legend_y, '■', fontsize=20, color=color_p1, weight='bold')
    ax.text(3.7, legend_y, 'Décision P1', fontsize=10, va='center')
    
    ax.text(5.5, legend_y, '■', fontsize=20, color=color_terminal, weight='bold')
    ax.text(6.2, legend_y, 'Terminal (payoff)', fontsize=10, va='center')
    
    ax.plot([8, 8.8], [legend_y, legend_y], 'k-', lw=2, alpha=0.6)
    ax.text(9.2, legend_y, 'Pass', fontsize=10, va='center')
    
    ax.plot([8, 8.8], [legend_y-0.4, legend_y-0.4], 'k--', lw=2, alpha=0.6)
    ax.text(9.2, legend_y-0.4, 'Bet', fontsize=10, va='center')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '6_decision_tree.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      → {output_path}")


def generate_probability_matrix():
    """
    Génère une matrice de probabilités comparant stratégie apprise vs Nash théorique
    """
    if TRAINING_RESULTS is None:
        print("   ERREUR: Pas de resultats d'entrainement disponibles")
        return
    
    print("   Generation de la matrice de probabilites...")
    
    trainer = TRAINING_RESULTS['trainer']
    strategy_profile = trainer.get_strategy_profile()
    
    # Stratégies Nash théoriques
    nash_strategies = {
        'Jack initial': [2/3, 1/3],
        'Jack après bet': [1.0, 0.0],
        'Queen initial': [1.0, 0.0],
        'Queen après bet': [2/3, 1/3],
        'King initial': [0.0, 1.0],
        'King après bet': [0.0, 1.0],
    }
    
    # Stratégies apprises correspondantes
    infoset_mapping = {
        'Jack initial': '0',
        'Jack après bet': '0b',
        'Queen initial': '1',
        'Queen après bet': '1b',
        'King initial': '2',
        'King après bet': '2b',
    }
    
    learned_strategies = {}
    for name, infoset in infoset_mapping.items():
        if infoset in strategy_profile:
            learned_strategies[name] = strategy_profile[infoset]
        else:
            learned_strategies[name] = np.array([0.5, 0.5])
    
    # Créer la figure
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 0.15])
    
    # Sous-plot Nash
    ax1 = fig.add_subplot(gs[0, 0])
    # Sous-plot Learned
    ax2 = fig.add_subplot(gs[0, 1])
    # Sous-plot pour la légende
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')
    
    situations = list(nash_strategies.keys())
    actions = ['Pass', 'Bet']
    
    # Préparer les données
    nash_data = np.array([nash_strategies[s] for s in situations])
    learned_data = np.array([learned_strategies[s] for s in situations])
    
    # Fonction pour tracer une matrice
    def plot_matrix(ax, data, title):
        im = ax.imshow(data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(len(situations)))
        ax.set_yticks(range(len(actions)))
        ax.set_xticklabels(situations, rotation=45, ha='right')
        ax.set_yticklabels(actions)
        
        ax.set_title(title, fontsize=14, weight='bold', pad=15)
        ax.set_ylabel('Actions', fontsize=12, weight='bold')
        
        # Ajouter les valeurs dans les cellules
        for i in range(len(situations)):
            for j in range(len(actions)):
                value = data[i, j]
                text_color = 'white' if value > 0.5 else 'black'
                ax.text(i, j, f'{value:.2%}', ha='center', va='center',
                       fontsize=11, weight='bold', color=text_color)
        
        return im
    
    # Tracer les deux matrices
    plot_matrix(ax1, nash_data, 'Stratégie Nash Théorique')
    im = plot_matrix(ax2, learned_data, f'Stratégie Apprise (CFR)')
    
    # Ajouter une barre de couleur
    cbar = fig.colorbar(im, ax=[ax1, ax2], orientation='horizontal', 
                       pad=0.15, aspect=30, shrink=0.6)
    cbar.set_label('Probabilité', fontsize=11, weight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Calculer et afficher les écarts
    differences = np.abs(nash_data - learned_data)
    max_diff = np.max(differences)
    avg_diff = np.mean(differences)
    
    # Texte d'analyse
    analysis_text = f"""Écart moyen: {avg_diff:.2%}  |  Écart maximum: {max_diff:.2%}  |  Précision: {(1-avg_diff)*100:.1f}%"""
    
    ax_legend.text(0.5, 0.5, analysis_text, ha='center', va='center',
                  fontsize=12, weight='bold',
                  bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', 
                           edgecolor='black', lw=2))
    
    plt.suptitle('Matrice de Probabilités - Comparaison Nash vs CFR', 
                fontsize=16, weight='bold', y=0.98)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '7_probability_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      → {output_path}")


def generate_all_visualizations(iterations: int = 10000):
    """
    Lance l'entraînement puis génère toutes les visualisations basées sur les résultats
    """
    # === PHASE 1: Entraînement ===
    run_main_training(iterations=iterations)
    
    # === PHASE 2: Visualisations ===
    print("\n" + "="*70)
    print("   GENERATION DES VISUALISATIONS")
    print("="*70)
    
    generate_convergence_plot()
    generate_strategy_comparison()
    generate_game_value_plot()
    generate_behavior_analysis()
    generate_summary_dashboard()
    generate_decision_tree()
    generate_probability_matrix()
    
    print("\n" + "="*70)
    print("   TOUTES LES VISUALISATIONS GENEREES!")
    print("="*70)
    print(f"\n   Dossier: {OUTPUT_DIR}\n")
    print("   Fichiers generes:")
    print("      1_convergence.png          - Convergence des strategies")
    print("      2_strategy_comparison.png  - Nash vs CFR")
    print("      3_game_value.png           - Convergence de la game value")
    print("      4_emergent_behaviors.png   - Comportements emergents")
    print("      5_dashboard.png            - Dashboard complet")
    print("      6_decision_tree.png        - Arbre de decision du jeu")
    print("      7_probability_matrix.png   - Matrice de probabilites")
    print()


def choose_iterations():
    """
    Permet à l'utilisateur de choisir le nombre d'itérations
    """
    print("\n" + "="*70)
    print("   CHOIX DU NOMBRE D'ITERATIONS")
    print("="*70)
    print("\n   Options disponibles:")
    print("   1. Rapide       - 10,000 iterations   (~1 sec)")
    print("   2. Standard     - 50,000 iterations   (~3 sec)")
    print("   3. Eleve        - 100,000 iterations  (~6 sec)")
    print("   4. Tres eleve   - 500,000 iterations  (~25 sec)")
    print("   5. Maximum      - 1,000,000 iterations (~50 sec)")
    print("   6. Personnalise - Entrer un nombre")
    
    while True:
        choice = input("\n   Votre choix (1-6): ").strip()
        
        if choice == '1':
            return 10000
        elif choice == '2':
            return 50000
        elif choice == '3':
            return 100000
        elif choice == '4':
            return 500000
        elif choice == '5':
            return 1000000
        elif choice == '6':
            while True:
                try:
                    custom = int(input("   Nombre d'iterations (min 1000): "))
                    if custom >= 1000:
                        return custom
                    else:
                        print("   Minimum 1000 iterations requis.")
                except ValueError:
                    print("   Veuillez entrer un nombre valide.")
        else:
            print("   Choix invalide. Veuillez choisir entre 1 et 6.")


if __name__ == "__main__":
    # Choisir le nombre d'itérations puis lancer
    iterations = choose_iterations()
    generate_all_visualizations(iterations=iterations)
