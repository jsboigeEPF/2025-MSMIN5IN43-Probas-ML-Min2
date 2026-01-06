"""
Counterfactual Regret Minimization (CFR) Algorithm
Algorithme d'apprentissage pour les jeux à information imparfaite
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict
import random
from kuhn_poker import KuhnPoker
from cfr_academic import compute_best_response_value, compute_exploitability, verify_nash_value


class InformationSet:
    """
    Représente un information set dans le jeu
    Contient les regrets cumulés et la stratégie pour chaque action possible
    """
    
    def __init__(self, num_actions: int = 2):
        self.num_actions = num_actions
        # Regret cumulé pour chaque action
        self.regret_sum = np.zeros(num_actions, dtype=np.float64)
        # Somme des stratégies sur toutes les itérations (pour calculer la stratégie moyenne)
        self.strategy_sum = np.zeros(num_actions, dtype=np.float64)
        # Stratégie actuelle
        self.strategy = np.zeros(num_actions, dtype=np.float64)
    
    def get_strategy(self, realization_weight: float = 1.0) -> np.ndarray:
        """
        Calcule la stratégie actuelle basée sur les regrets
        Utilise Regret Matching: actions avec regret positif obtiennent plus de probabilité
        
        Args:
            realization_weight: Poids de réalisation pour mettre à jour strategy_sum
            
        Returns:
            Distribution de probabilité sur les actions
        """
        normalizing_sum = 0
        
        for a in range(self.num_actions):
            # Regret Matching: prendre max(0, regret)
            self.strategy[a] = max(0, self.regret_sum[a])
            normalizing_sum += self.strategy[a]
        
        # Normaliser pour obtenir une distribution de probabilité
        if normalizing_sum > 0:
            self.strategy = self.strategy / normalizing_sum
        else:
            # Stratégie uniforme si tous les regrets sont négatifs
            self.strategy = np.ones(self.num_actions) / self.num_actions
        
        # Accumuler la stratégie pour calculer la stratégie moyenne finale
        self.strategy_sum += realization_weight * self.strategy
        
        return self.strategy
    
    def get_average_strategy(self) -> np.ndarray:
        """
        Retourne la stratégie moyenne sur toutes les itérations
        C'est cette stratégie qui converge vers l'équilibre de Nash
        
        Returns:
            Distribution de probabilité moyenne sur les actions
        """
        avg_strategy = np.zeros(self.num_actions, dtype=np.float64)
        normalizing_sum = np.sum(self.strategy_sum)
        
        if normalizing_sum > 0:
            avg_strategy = self.strategy_sum / normalizing_sum
        else:
            # Stratégie uniforme par défaut
            avg_strategy = np.ones(self.num_actions) / self.num_actions
        
        return avg_strategy


class CFRTrainer:
    """
    Entraîneur utilisant l'algorithme CFR (Counterfactual Regret Minimization)
    """
    
    def __init__(self):
        self.game = KuhnPoker()
        # Dictionnaire des information sets
        self.infosets: Dict[str, InformationSet] = defaultdict(
            lambda: InformationSet(num_actions=self.game.NUM_ACTIONS)
        )
        self.iterations = 0
    
    def train(self, iterations: int, track_convergence: bool = False, 
              checkpoint_interval: int = 1000) -> Dict[str, InformationSet]:
        """
        Entraîne l'agent en jouant contre lui-même pendant un nombre d'itérations
        
        Args:
            iterations: Nombre d'itérations d'entraînement
            track_convergence: Si True, track l'exploitabilité pendant l'entraînement
            checkpoint_interval: Intervalle pour calculer l'exploitabilité
            
        Returns:
            Dictionnaire des information sets avec leurs stratégies
        """
        util = 0
        self.exploitability_history = [] if track_convergence else None
        self.iteration_checkpoints = [] if track_convergence else None
        
        for i in range(iterations):
            # Mélanger et distribuer les cartes
            cards = [0, 1, 2]
            random.shuffle(cards)
            
            # Cartes des joueurs (la 3ème carte reste cachée)
            player_cards = cards[:2]
            
            # Exécuter CFR pour les deux joueurs
            util += self.cfr(player_cards, "", 1.0, 1.0)
            
            self.iterations += 1
            
            # Tracking de convergence (comme Libratus/Pluribus)
            if track_convergence and (i + 1) % checkpoint_interval == 0:
                strategy_profile = self.get_strategy_profile()
                exploitability = compute_exploitability(self.game, strategy_profile)
                self.exploitability_history.append(exploitability)
                self.iteration_checkpoints.append(i + 1)
        
        print(f"Utilité moyenne du joueur 0: {util / iterations:.4f}")
        
        return self.infosets
    
    def cfr(self, cards: List[int], history: str, p0: float, p1: float) -> float:
        """
        Algorithme CFR récursif
        
        Args:
            cards: Cartes des joueurs [carte_j0, carte_j1]
            history: Historique des actions ("pb" = pass puis bet)
            p0: Probabilité de reach du joueur 0
            p1: Probabilité de reach du joueur 1
            
        Returns:
            Utilité contrefactuelle pour le joueur actuel
        """
        plays = len(history)
        player = plays % 2
        opponent = 1 - player
        
        # État terminal
        if self.game.is_terminal(history):
            return self.game.get_payoff(history, cards)
        
        # Obtenir l'information set
        infoset_key = self.game.get_information_set(cards[player], history)
        infoset = self.infosets[infoset_key]
        
        # Obtenir la stratégie actuelle
        if player == 0:
            strategy = infoset.get_strategy(p0)
        else:
            strategy = infoset.get_strategy(p1)
        
        # Calculer les utilités pour chaque action
        action_utils = np.zeros(self.game.NUM_ACTIONS)
        
        for action in range(self.game.NUM_ACTIONS):
            action_char = 'p' if action == 0 else 'b'
            next_history = history + action_char
            
            # Récursion
            if player == 0:
                action_utils[action] = -self.cfr(cards, next_history, 
                                                  p0 * strategy[action], p1)
            else:
                action_utils[action] = -self.cfr(cards, next_history, 
                                                  p0, p1 * strategy[action])
        
        # Utilité du nœud
        node_util = np.sum(strategy * action_utils)
        
        # Calculer les regrets et les accumuler
        regrets = action_utils - node_util
        
        if player == 0:
            infoset.regret_sum += p1 * regrets
        else:
            infoset.regret_sum += p0 * regrets
        
        return node_util
    
    def get_strategy_profile(self) -> Dict[str, np.ndarray]:
        """
        Retourne le profil de stratégie moyen pour tous les information sets
        
        Returns:
            Dictionnaire {infoset_key: stratégie_moyenne}
        """
        strategy_profile = {}
        
        for infoset_key, infoset in self.infosets.items():
            strategy_profile[infoset_key] = infoset.get_average_strategy()
        
        return strategy_profile
    
    def evaluate_strategy(self, cards: List[int], history: str, strategy_profile: Dict[str, np.ndarray]) -> float:
        """
        Évalue la stratégie moyenne en calculant l'espérance exacte
        
        Args:
            cards: Cartes des joueurs [carte_j0, carte_j1]
            history: Historique des actions
            strategy_profile: Profil de stratégie à évaluer
            
        Returns:
            Utilité pour le joueur 0
        """
        # État terminal
        if self.game.is_terminal(history):
            return self.game.get_payoff(history, cards)
        
        plays = len(history)
        player = plays % 2
        
        # Obtenir l'information set et la stratégie
        infoset_key = self.game.get_information_set(cards[player], history)
        strategy = strategy_profile.get(infoset_key, np.array([0.5, 0.5]))
        
        # Calculer l'espérance sur toutes les actions
        value = 0
        for action in range(self.game.NUM_ACTIONS):
            action_char = 'p' if action == 0 else 'b'
            next_history = history + action_char
            value += strategy[action] * self.evaluate_strategy(cards, next_history, strategy_profile)
        
        return value
    
    def display_strategy(self):
        """Affiche la stratégie apprise de manière lisible"""
        print("\n" + "="*60)
        print("STRATÉGIE APPRISE (Nash Equilibrium approximé)")
        print("="*60)
        
        strategy_profile = self.get_strategy_profile()
        
        # Organiser par carte
        for card in range(3):
            card_name = self.game.get_card_name(card)
            print(f"\n{card_name}:")
            print("-" * 40)
            
            # Trier les information sets par historique
            relevant_infosets = {k: v for k, v in strategy_profile.items() 
                                if k.startswith(str(card))}
            
            for infoset_key in sorted(relevant_infosets.keys(), key=lambda x: (len(x), x)):
                strategy = relevant_infosets[infoset_key]
                history = infoset_key[1:]  # Enlever la carte
                
                if history == "":
                    history = "début"
                
                pass_prob = strategy[0] * 100
                bet_prob = strategy[1] * 100
                
                print(f"  Après '{history}': "
                      f"Pass={pass_prob:.1f}%, Bet={bet_prob:.1f}%")
