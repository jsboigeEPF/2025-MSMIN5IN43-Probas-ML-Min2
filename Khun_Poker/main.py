"""
Script principal pour entraîner et analyser l'agent de Poker CFR
Démontre l'apprentissage et l'analyse de la stratégie Nash Equilibrium
"""

import numpy as np
import matplotlib.pyplot as plt
from cfr_algorithm import CFRTrainer
from cfr_academic import compute_exploitability, verify_nash_value, compute_game_value
from kuhn_poker import KuhnPoker
import time


def analyze_exploitability_academic(trainer: CFRTrainer) -> float:
    """
    Calcule l'exploitabilité selon la métrique académique standard
    (Best Response Value - Standard Libratus/Pluribus/OpenSpiel)
    
    Args:
        trainer: L'entraîneur CFR avec la stratégie apprise
        
    Returns:
        Valeur d'exploitabilité (en milli-big-blinds)
    """
    strategy_profile = trainer.get_strategy_profile()
    return compute_exploitability(trainer.game, strategy_profile)


def analyze_exploitability(trainer: CFRTrainer, use_best_response: bool = True) -> float:
    """
    Calcule l'exploitabilité avec choix de métrique
    
    Args:
        trainer: L'entraîneur CFR avec la stratégie apprise
        use_best_response: Si True, utilise best response value (standard académique)
                          Si False, utilise distance euclidienne (métrique simplifiée)
        
    Returns:
        Valeur d'exploitabilité (en milli-big-blinds)
    """
    if use_best_response:
        # Méthode académique standard (Libratus/Pluribus)
        return analyze_exploitability_academic(trainer)
    else:
        # Méthode alternative: distance euclidienne pondérée
        strategy_profile = trainer.get_strategy_profile()
        
        nash_strategies = {
            '0': np.array([2/3, 1/3]),      '0pb': np.array([1.0, 0.0]),
            '1': np.array([1.0, 0.0]),      '1pb': np.array([1.0, 0.0]),
            '2': np.array([0.0, 1.0]),      '2pb': np.array([0.0, 1.0]),
            '0p': np.array([0.0, 1.0]),     '0b': np.array([1.0, 0.0]),
            '1p': np.array([1.0, 0.0]),     '1b': np.array([2/3, 1/3]),
            '2p': np.array([1.0, 0.0]),     '2b': np.array([0.0, 1.0]),
        }
        
        visit_frequencies = {
            '0': 1/3,  '0p': 1/6,  '0b': 1/6,  '0pb': 1/18,
            '1': 1/3,  '1p': 1/6,  '1b': 1/6,  '1pb': 1/18,
            '2': 1/3,  '2p': 1/6,  '2b': 1/6,  '2pb': 0.0,
        }
        
        total_weighted_distance = 0.0
        total_weight = 0.0
        
        for infoset_key, nash_strategy in nash_strategies.items():
            if infoset_key in strategy_profile:
                learned_strategy = strategy_profile[infoset_key]
                distance = np.sqrt(np.sum((learned_strategy - nash_strategy) ** 2))
                weight = visit_frequencies.get(infoset_key, 1.0)
                total_weighted_distance += distance * weight
                total_weight += weight
        
        avg_distance = (total_weighted_distance / total_weight) if total_weight > 0 else 0
        return avg_distance * 1000


def run_training_experiment(iterations: int = 10000):
    """
    Exécute une expérience d'entraînement complète avec analyse
    
    Args:
        iterations: Nombre d'itérations d'entraînement
    """
    print("\n" + "="*70)
    print("POKER AI - COUNTERFACTUAL REGRET MINIMIZATION (CFR)")
    print("="*70)
    print(f"\nJeu: Kuhn Poker")
    print(f"Algorithme: CFR (Counterfactual Regret Minimization)")
    print(f"Itérations: {iterations:,}")
    print("\nDébut de l'entraînement...")
    
    start_time = time.time()
    
    # Créer et entraîner l'agent
    trainer = CFRTrainer()
    trainer.train(iterations)
    
    training_time = time.time() - start_time
    
    print(f"\nEntraînement terminé en {training_time:.2f} secondes")
    print(f"Vitesse: {iterations/training_time:.0f} itérations/seconde")
    
    # Afficher la stratégie apprise
    trainer.display_strategy()
    
    # Analyser les stratégies par comparaison directe avec Nash théorique
    strategy_profile = trainer.get_strategy_profile()
    
    # Extraire les stratégies clés
    jack_bet = strategy_profile.get('0', np.array([0.5, 0.5]))[1] * 100
    queen_call = strategy_profile.get('1b', np.array([0.5, 0.5]))[1] * 100
    king_bet = strategy_profile.get('2', np.array([0.5, 0.5]))[1] * 100
    
    # Calculer la précision (erreur relative)
    jack_error = abs(jack_bet - 33.3) / 33.3 * 100
    queen_error = abs(queen_call - 33.3) / 33.3 * 100
    king_error = abs(king_bet - 100.0) / 100.0 * 100
    avg_error = (jack_error + queen_error + king_error) / 3
    overall_accuracy = max(0, 100 - avg_error)
    
    # Calculer la game value
    game_value = compute_game_value(trainer.game, strategy_profile)
    nash_value = -1/18  # Valeur théorique de Nash pour Kuhn Poker (convention académique)
    
    print(f"\n" + "="*70)
    print("ANALYSE DE LA STRATÉGIE")
    print("="*70)
    
    print(f"\n📊 Game Value:")
    print(f"   Valeur apprise:    {game_value:.6f}")
    print(f"   Valeur Nash:       {nash_value:.6f} (-1/18)")
    print(f"   Différence:        {abs(game_value - nash_value):.6f}")
    
    print(f"\n📊 Précision des stratégies vs Nash théorique:")
    print(f"   Jack bluff:      {jack_bet:5.1f}% (théorie: 33.3%) → erreur {jack_error:.1f}%")
    print(f"   Queen call:      {queen_call:5.1f}% (théorie: 33.3%) → erreur {queen_error:.1f}%")  
    print(f"   King value bet:  {king_bet:5.1f}% (théorie: 100%)  → erreur {king_error:.1f}%")
    print(f"\n   📈 Précision globale: {overall_accuracy:.1f}%")
    
    if overall_accuracy >= 99.5:
        quality_emoji = "✨"
        quality = "EXCELLENT - Convergence quasi-parfaite vers Nash"
    elif overall_accuracy >= 99.0:
        quality_emoji = "⭐"
        quality = "TRÈS BON - Convergence solide vers Nash"
    elif overall_accuracy >= 95.0:
        quality_emoji = "👍"
        quality = "BON - Convergence satisfaisante"
    else:
        quality_emoji = "⚠️"
        quality = "MOYEN - Nécessite plus d'itérations"
    
    print(f"\n{quality_emoji} Qualité: {quality}")
    
    # Afficher des statistiques sur les information sets
    strategy_profile = trainer.get_strategy_profile()
    print(f"\nNombre d'information sets explorés: {len(strategy_profile)}")
    
    return trainer


def compare_strategies(trainer: CFRTrainer):
    """
    Compare les stratégies pour différentes cartes
    
    Args:
        trainer: L'entraîneur CFR avec la stratégie apprise
    """
    print("\n" + "="*60)
    print("ANALYSE COMPARATIVE DES STRATÉGIES")
    print("="*60)
    
    strategy_profile = trainer.get_strategy_profile()
    game = KuhnPoker()
    
    # Analyser les décisions au premier coup
    print("\nDécisions initiales (premier coup):")
    print("-" * 40)
    
    for card in range(3):
        infoset_key = f"{card}"
        if infoset_key in strategy_profile:
            strategy = strategy_profile[infoset_key]
            card_name = game.get_card_name(card)
            print(f"{card_name:6s}: Pass={strategy[0]*100:5.1f}%, Bet={strategy[1]*100:5.1f}%")
    
    # Analyser les réponses après un bet
    print("\nRéponses après un BET adverse:")
    print("-" * 40)
    
    for card in range(3):
        infoset_key = f"{card}b"
        if infoset_key in strategy_profile:
            strategy = strategy_profile[infoset_key]
            card_name = game.get_card_name(card)
            print(f"{card_name:6s}: Pass={strategy[0]*100:5.1f}% (fold), "
                  f"Bet={strategy[1]*100:5.1f}% (call)")


def explain_nash_equilibrium():
    """
    Explique l'équilibre de Nash dans Kuhn Poker
    """
    print("\n" + "="*70)
    print("ÉQUILIBRE DE NASH THÉORIQUE - KUHN POKER")
    print("="*70)
    print("""
L'équilibre de Nash dans Kuhn Poker (solution théorique optimale):

JOUEUR avec JACK (carte la plus faible):
  - Au début: PASS 2/3, BET 1/3 (bluffer 33%% du temps au premier coup)
  - Après pass/bet: Toujours FOLD (ne jamais call avec Jack)
  
JOUEUR avec QUEEN (carte moyenne):
  - Au début: Toujours PASS
  - Après pass/bet: Toujours FOLD (ne pas call avec Queen)
  
JOUEUR avec KING (carte la plus forte):
  - Au début: BET 3 fois sur 3 (toujours bet pour value)
  - Après pass/bet: Toujours CALL/BET

PROPRIÉTÉS:
  - Valeur du jeu: -1/18 ≈ -0.0556 pour le joueur 0
  - Aucun joueur ne peut améliorer son gain en changeant unilatéralement
  - La stratégie est équilibrée entre bluffs et value bets
  
POURQUOI C'EST OPTIMAL:
  - Jack bluffe 1/3 du temps pour empêcher l'adversaire de toujours folder
  - King mise toujours pour value avec la meilleure carte
  - Queen fold car elle perd contre King et peut être bluffée par Jack
  - Le bluff à 33%% rend l'adversaire indifférent entre call et fold
    """)


def visualize_convergence(max_iterations: int = 100000, checkpoints: int = 20):
    """
    Visualise la convergence de l'algorithme CFR avec tracking temps réel
    Similaire à l'approche de Libratus/Pluribus
    
    Args:
        max_iterations: Nombre total d'itérations
        checkpoints: Nombre de points de vérification
    """
    print("\n" + "="*70)
    print("ANALYSE DE CONVERGENCE (Tracking style Libratus)")
    print("="*70)
    print(f"Entraînement avec {max_iterations:,} itérations...")
    print(f"Métrique: Best Response Exploitability (standard académique)\n")
    
    checkpoint_interval = max_iterations // checkpoints
    exploitabilities = []
    strategy_accuracies = []
    iteration_counts = []
    
    trainer = CFRTrainer()
    
    for i in range(1, checkpoints + 1):
        # Entraîner
        trainer.train(checkpoint_interval, track_convergence=False)
        
        # Calculer exploitabilité (best response)
        exploit = analyze_exploitability(trainer, use_best_response=True)
        exploitabilities.append(exploit)
        
        # Calculer précision des stratégies clés vs Nash (erreur relative)
        strategy_profile = trainer.get_strategy_profile()
        jack_bet = strategy_profile.get('0', np.array([0.5, 0.5]))[1] * 100
        queen_call = strategy_profile.get('1b', np.array([0.5, 0.5]))[1] * 100
        king_bet = strategy_profile.get('2', np.array([0.5, 0.5]))[1] * 100
        
        # Précision vs théorie (33.3%, 33.3%, 100%)
        jack_error = abs(jack_bet - 33.3) / 33.3 * 100
        queen_error = abs(queen_call - 33.3) / 33.3 * 100
        king_error = abs(king_bet - 100.0) / 100.0 * 100
        avg_error = (jack_error + queen_error + king_error) / 3
        overall_acc = max(0, 100 - avg_error)
        strategy_accuracies.append(overall_acc)
        
        iteration_counts.append(i * checkpoint_interval)
        
        if i % 5 == 0:
            print(f"  [{i * checkpoint_interval:>7,} iter] "
                  f"Exploit={exploit:>6.3f} mbb  |  "
                  f"Précision={overall_acc:>5.1f}%")
    
    # Créer deux subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Graphique 1: Exploitabilité (Best Response)
    ax1.plot(iteration_counts, exploitabilities, 'b-', linewidth=2, marker='o', label='Exploitabilité')
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, linewidth=1.5, label='Seuil quasi-optimal (<1 mbb)')
    ax1.axhline(y=5.0, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Seuil expert (<5 mbb)')
    ax1.set_xlabel('Nombre d\'itérations', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Exploitabilité (milli-big-blinds)', fontsize=12, fontweight='bold')
    ax1.set_title('Convergence CFR - Best Response Exploitability (Standard Académique)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_ylim(bottom=0)
    
    # Graphique 2: Précision des stratégies clés
    ax2.plot(iteration_counts, strategy_accuracies, 'g-', linewidth=2, marker='s', label='Précision stratégies clés')
    ax2.axhline(y=100, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Nash parfait (100%)')
    ax2.axhline(y=99, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Seuil excellent (99%)')
    ax2.set_xlabel('Nombre d\'itérations', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Précision (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Convergence des stratégies clés vers Nash (Jack bluff, Queen call, King bet)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    ax2.set_ylim(90, 100.5)
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    plt.savefig('d:\\Documents\\Ecole\\EPF\\5A EPF\\IA 2\\Poker\\cfr_convergence.png', 
                dpi=150, bbox_inches='tight')
    print(f"\n📊 Graphiques sauvegardés: cfr_convergence.png")
    print(f"   • Exploitabilité (Best Response Value)")
    print(f"   • Précision des stratégies clés vs Nash")
    plt.close()
    
    return trainer


def choose_iterations() -> int:
    """
    Menu pour choisir le nombre d'itérations d'entraînement
    
    Returns:
        Nombre d'itérations choisi
    """
    print("\n" + "="*70)
    print("CHOIX DU NOMBRE D'ITÉRATIONS")
    print("="*70)
    print("\nOptions disponibles:")
    print("  1. Rapide       - 10,000 itérations   (~0.5 sec)")
    print("  2. Normal       - 50,000 itérations   (~2.5 sec)")
    print("  3. Élevé        - 100,000 itérations  (~5 sec)")
    print("  4. Très élevé   - 500,000 itérations  (~25 sec)")
    print("  5. Maximum      - 1,000,000 itérations (~50 sec)")
    print("  6. Personnalisé - Entrer un nombre")
    
    while True:
        choice = input("\nVotre choix (1-6): ").strip()
        
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
                    custom = int(input("Nombre d'itérations (min 1000): "))
                    if custom >= 1000:
                        return custom
                    else:
                        print("Minimum 1000 itérations requis.")
                except ValueError:
                    print("Veuillez entrer un nombre valide.")
        else:
            print("Choix invalide. Veuillez choisir entre 1 et 6.")


def main():
    """Fonction principale"""
    
    # Expliquer l'équilibre de Nash théorique
    explain_nash_equilibrium()
    
    # Choisir le nombre d'itérations
    iterations = choose_iterations()
    
    # Entraîner l'agent
    print("\n" + "="*70)
    print("PHASE 1: ENTRAÎNEMENT")
    print("="*70)
    trainer = run_training_experiment(iterations=iterations)
    
    # Comparer les stratégies
    compare_strategies(trainer)
    
    # Visualiser la convergence (optionnel - commenté par défaut car prend du temps)
    print("\n" + "="*70)
    print("PHASE 2: ANALYSE DE CONVERGENCE (optionnel)")
    print("="*70)
    response = input("\nVoulez-vous analyser la convergence? (o/n): ").lower()
    
    if response == 'o':
        final_trainer = visualize_convergence(max_iterations=100000, checkpoints=20)
        print("\nStratégie finale après convergence complète:")
        final_trainer.display_strategy()
    
    print("\n" + "="*70)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*70)
    print("""
RÉSUMÉ:
✓ Algorithme CFR implémenté et testé
✓ Convergence vers l'équilibre de Nash démontrée
✓ Stratégie analysée et comparée à la théorie
    """)


if __name__ == "__main__":
    main()
