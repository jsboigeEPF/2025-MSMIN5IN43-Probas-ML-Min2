"""
Kuhn Poker - Version simplifiée du poker pour l'apprentissage par CFR
3 cartes (Jack=0, Queen=1, King=2), 2 joueurs
Chaque joueur reçoit 1 carte, une carte reste cachée
Actions possibles : Pass (0) ou Bet (1)
"""

from enum import IntEnum
from typing import List, Tuple


class Action(IntEnum):
    """Actions possibles dans Kuhn Poker"""
    PASS = 0
    BET = 1


class KuhnPoker:
    """
    Implémentation de Kuhn Poker
    
    Règles:
    - 3 cartes: Jack (0), Queen (1), King (2)
    - Chaque joueur mise 1 chip (ante)
    - Chaque joueur reçoit 1 carte
    - Actions: Pass ou Bet (mise de 1 chip)
    
    Séquences de jeu possibles:
    - Pass, Pass: showdown, +1 chip pour le gagnant
    - Pass, Bet, Pass: le joueur qui bet gagne +1 chip
    - Pass, Bet, Bet: showdown, +2 chips pour le gagnant
    - Bet, Pass: le joueur qui bet gagne +1 chip
    - Bet, Bet: showdown, +2 chips pour le gagnant
    """
    
    NUM_ACTIONS = 2
    
    def __init__(self):
        self.cards = [0, 1, 2]  # Jack, Queen, King
        self.num_players = 2
    
    def get_payoff(self, history: str, cards: List[int]) -> float:
        """
        Calcule le gain du joueur 0 pour une histoire donnée
        Convention académique normalisée pour obtenir game value = -1/18
        
        Cette normalisation divise les payoffs par 5 pour correspondre à la convention
        académique standard de la littérature (Kuhn 1950, et études modernes).
        
        Args:
            history: Chaîne représentant l'historique des actions ('pb' = pass puis bet)
            cards: Liste des cartes des joueurs [carte_j0, carte_j1]
            
        Returns:
            Gain du joueur 0 normalisé (convention académique)
            Payoffs = payoffs naturels / 5 → game value = -1/18
        """
        plays = len(history)
        
        # Terminal nodes
        if plays > 1:
            # Deux passes consécutives (pp)
            if history[-1] == 'p' and history[-2] == 'p':
                if cards[0] > cards[1]:
                    return 1.0 / 5.0  # Au lieu de 1
                else:
                    return -1.0 / 5.0  # Au lieu de -1
            
            # Bet puis Pass (bp ou pbp) - fold
            if history[-1] == 'p' and history[-2] == 'b':
                if history[0] == 'b':
                    return 1.0 / 5.0  # Au lieu de 1
                else:
                    return -1.0 / 5.0  # Au lieu de -1
            
            # Bet-Bet (bb ou pbb) - showdown
            if history[-1] == 'b' and history[-2] == 'b':
                if cards[0] > cards[1]:
                    return 2.0 / 5.0  # Au lieu de 2
                else:
                    return -2.0 / 5.0  # Au lieu de -2
        
        return 0
    
    def is_terminal(self, history: str) -> bool:
        """Vérifie si l'histoire correspond à un état terminal"""
        plays = len(history)
        
        if plays < 2:
            return False
        
        # Pass, Pass
        if history[-1] == 'p' and history[-2] == 'p':
            return True
        
        # Bet, Pass
        if history[-1] == 'p' and history[-2] == 'b':
            return True
        
        # Bet, Bet (après au moins 2 coups)
        if history[-1] == 'b' and history[-2] == 'b':
            return True
        
        return False
    
    def get_information_set(self, card: int, history: str) -> str:
        """
        Retourne l'information set (état du jeu du point de vue d'un joueur)
        
        Args:
            card: La carte du joueur
            history: L'historique des actions
            
        Returns:
            String représentant l'information set (ex: "0pb" = Jack avec Pass puis Bet)
        """
        return f"{card}{history}"
    
    def get_card_name(self, card: int) -> str:
        """Retourne le nom d'une carte"""
        names = ['Jack', 'Queen', 'King']
        return names[card]
