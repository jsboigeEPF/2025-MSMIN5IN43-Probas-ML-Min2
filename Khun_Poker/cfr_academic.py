"""
Fonctions académiques pour CFR: Best Response et Nash Value
Implémentation CORRIGÉE conforme aux standards Libratus/Pluribus/OpenSpiel

CORRECTION MAJEURE:
Le Best Response doit optimiser par INFORMATION SET, pas par noeud de jeu.
Un joueur ne connaît pas la carte de l'adversaire, donc il doit choisir
une stratégie qui est optimale EN MOYENNE sur toutes les cartes possibles
de l'adversaire.
"""

import numpy as np
from typing import Dict
from kuhn_poker import KuhnPoker


def compute_exploitability(game: KuhnPoker, strategy_profile: Dict[str, np.ndarray]) -> float:
    """
    Calcule l'exploitabilité selon la métrique académique standard.
    
    Exploitabilité = (BR_value_P0 - BR_value_P1) / 2
    
    où:
    - BR_value_P0 = valeur du jeu quand P0 joue son Best Response
    - BR_value_P1 = valeur du jeu (point de vue P0) quand P1 joue son Best Response
    
    Pour une stratégie Nash parfaite, exploitabilité ≈ 0.
    
    Returns:
        Exploitabilité en milli-big-blinds (mbb)
    """
    br_value_p0 = compute_best_response_value(game, strategy_profile, 0)
    br_value_p1 = compute_best_response_value(game, strategy_profile, 1)
    
    exploitability = (br_value_p0 - br_value_p1) / 2
    
    return exploitability * 1000


def compute_best_response_value(game: KuhnPoker, strategy_profile: Dict[str, np.ndarray], 
                                 br_player: int) -> float:
    """
    Calcule la valeur du jeu quand br_player joue son Best Response optimal
    contre l'adversaire qui joue strategy_profile.
    
    IMPORTANT: Le BR optimise par INFORMATION SET, pas par noeud.
    Le joueur BR ne connaît pas la carte de l'adversaire!
    
    Algorithme:
    1. Pour chaque information set du br_player, calculer l'EV de chaque action
       en moyennant sur toutes les cartes possibles de l'adversaire
    2. Choisir l'action avec le meilleur EV pour chaque infoset
    3. Calculer la valeur totale du jeu avec cette stratégie BR
    
    Returns:
        Valeur du jeu du point de vue du joueur 0
    """
    num_actions = 2
    
    # Étape 1: Calculer l'EV de chaque action dans chaque infoset du BR player
    infoset_action_ev = {}
    
    def collect_action_values(cards, history, prob_reach):
        """Collecte les valeurs EV pour chaque action de chaque infoset."""
        if game.is_terminal(history):
            payoff = game.get_payoff(history, cards)
            if br_player == 1:
                payoff = -payoff
            return payoff
        
        plays = len(history)
        current_player = plays % 2
        infoset = game.get_information_set(cards[current_player], history)
        
        if current_player == br_player:
            if infoset not in infoset_action_ev:
                infoset_action_ev[infoset] = {a: [0.0, 0.0] for a in range(num_actions)}
            
            for action in range(num_actions):
                action_char = 'p' if action == 0 else 'b'
                value = collect_action_values(cards, history + action_char, prob_reach)
                infoset_action_ev[infoset][action][0] += prob_reach * value
                infoset_action_ev[infoset][action][1] += prob_reach
            
            return 0.0
        else:
            strategy = strategy_profile.get(infoset, np.ones(num_actions) / num_actions)
            total = 0.0
            for action in range(num_actions):
                action_char = 'p' if action == 0 else 'b'
                new_prob = prob_reach * strategy[action]
                if new_prob > 0:
                    total += strategy[action] * collect_action_values(cards, history + action_char, new_prob)
            return total
    
    for c0 in range(3):
        for c1 in range(3):
            if c0 != c1:
                collect_action_values([c0, c1], "", 1.0 / 6)
    
    # Étape 2: Construire la stratégie BR optimale
    br_strategy = {}
    for infoset, action_data in infoset_action_ev.items():
        ev = {}
        for action in range(num_actions):
            if action_data[action][1] > 0:
                ev[action] = action_data[action][0] / action_data[action][1]
            else:
                ev[action] = 0.0
        
        best_action = max(ev.keys(), key=lambda a: ev[a])
        br_strategy[infoset] = np.array([1.0 if a == best_action else 0.0 for a in range(num_actions)])
    
    # Étape 3: Calculer la valeur du jeu avec la stratégie BR
    def compute_value(cards, history):
        if game.is_terminal(history):
            return game.get_payoff(history, cards)
        
        plays = len(history)
        current_player = plays % 2
        infoset = game.get_information_set(cards[current_player], history)
        
        if current_player == br_player:
            strategy = br_strategy.get(infoset, np.ones(num_actions) / num_actions)
        else:
            strategy = strategy_profile.get(infoset, np.ones(num_actions) / num_actions)
        
        total = 0.0
        for action in range(num_actions):
            action_char = 'p' if action == 0 else 'b'
            total += strategy[action] * compute_value(cards, history + action_char)
        return total
    
    total_value = sum(compute_value([c0, c1], "") for c0 in range(3) for c1 in range(3) if c0 != c1)
    return total_value / 6


def compute_game_value(game: KuhnPoker, strategy_profile: Dict[str, np.ndarray]) -> float:
    """Calcule la valeur espérée du jeu pour P0 quand les deux joueurs jouent strategy_profile."""
    num_actions = 2
    
    def recursive_value(cards, history):
        if game.is_terminal(history):
            return game.get_payoff(history, cards)
        
        plays = len(history)
        current_player = plays % 2
        infoset = game.get_information_set(cards[current_player], history)
        strategy = strategy_profile.get(infoset, np.ones(num_actions) / num_actions)
        
        return sum(strategy[a] * recursive_value(cards, history + ('p' if a == 0 else 'b')) 
                   for a in range(num_actions))
    
    # Calculer la valeur espérée sur toutes les distributions de cartes possibles
    # Il y a 3! = 6 permutations, mais seules les 2 premières cartes comptent
    # Donc 3 * 2 = 6 paires ordonnées possibles, chacune avec probabilité 1/6
    total_value = 0
    num_deals = 0
    
    for c0 in range(3):
        for c1 in range(3):
            if c0 != c1:
                value = recursive_value([c0, c1], "")
                total_value += value
                num_deals += 1
    
    return total_value / num_deals


def verify_nash_value(game: KuhnPoker, strategy_profile: Dict[str, np.ndarray], 
                     num_games: int = 10000) -> tuple:
    """
    Vérifie si la stratégie atteint la valeur Nash théorique de -1/18.
    
    Returns:
        (valeur_exacte, valeur_théorique)
    """
    exact_value = compute_game_value(game, strategy_profile)
    nash_value = -1/18
    return exact_value, nash_value
