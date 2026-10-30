"""
Script interactif pour jouer contre l'IA entraînée
"""

import random
from cfr_algorithm import CFRTrainer
from kuhn_poker import KuhnPoker


class InteractivePlayer:
    """Permet à un humain de jouer contre l'IA"""
    
    def __init__(self, trainer: CFRTrainer):
        self.trainer = trainer
        self.game = KuhnPoker()
        self.strategy_profile = trainer.get_strategy_profile()
    
    def get_ai_action(self, card: int, history: str) -> int:
        """Obtient l'action de l'IA basée sur la stratégie apprise"""
        infoset_key = self.game.get_information_set(card, history)
        
        if infoset_key in self.strategy_profile:
            strategy = self.strategy_profile[infoset_key]
            # Choisir une action selon la distribution de probabilité
            return random.choices([0, 1], weights=strategy)[0]
        else:
            # Stratégie par défaut si l'information set n'a pas été vu
            return random.randint(0, 1)
    
    def get_human_action(self, card: int, history: str) -> int:
        """Demande l'action au joueur humain"""
        print(f"\nVotre carte: {self.game.get_card_name(card)}")
        print(f"Historique: {history if history else 'début de la partie'}")
        
        while True:
            action = input("Votre action (p=Pass, b=Bet): ").lower()
            if action == 'p':
                return 0
            elif action == 'b':
                return 1
            else:
                print("Action invalide. Utilisez 'p' ou 'b'.")
    
    def play_game(self, human_first: bool = True) -> int:
        """
        Joue une partie complète
        
        Args:
            human_first: Si True, l'humain joue en premier
            
        Returns:
            Gain du joueur humain
        """
        # Distribuer les cartes
        cards = [0, 1, 2]
        random.shuffle(cards)
        
        if human_first:
            human_card = cards[0]
            ai_card = cards[1]
            human_pos = 0
        else:
            human_card = cards[1]
            ai_card = cards[0]
            human_pos = 1
        
        print("\n" + "="*50)
        print("NOUVELLE PARTIE")
        print("="*50)
        
        history = ""
        current_player = 0
        
        while not self.game.is_terminal(history):
            if current_player == human_pos:
                action = self.get_human_action(human_card, history)
                action_name = "Pass" if action == 0 else "Bet"
                print(f"Vous jouez: {action_name}")
            else:
                action = self.get_ai_action(ai_card, history)
                action_name = "Pass" if action == 0 else "Bet"
                print(f"IA joue: {action_name}")
            
            history += 'p' if action == 0 else 'b'
            current_player = 1 - current_player
        
        # Calculer le résultat
        game_cards = [human_card, ai_card] if human_first else [ai_card, human_card]
        payoff = self.game.get_payoff(history, game_cards)
        
        # Ajuster le payoff selon la position du joueur
        if not human_first:
            payoff = -payoff
        
        # Afficher le résultat
        print("\n" + "-"*50)
        print(f"Carte de l'IA: {self.game.get_card_name(ai_card)}")
        
        if payoff > 0:
            print(f"✓ Vous gagnez {payoff} chip(s)!")
        elif payoff < 0:
            print(f"✗ Vous perdez {-payoff} chip(s)!")
        else:
            print("⚖ Égalité!")
        
        return payoff
    
    def play_session(self, num_games: int = 10):
        """
        Joue plusieurs parties et affiche les statistiques
        
        Args:
            num_games: Nombre de parties à jouer
        """
        print("\n" + "="*60)
        print("SESSION DE JEU CONTRE L'IA")
        print("="*60)
        print(f"\nVous allez jouer {num_games} parties.")
        print("L'ordre des joueurs alternera entre les parties.\n")
        
        total_payoff = 0
        wins = 0
        losses = 0
        ties = 0
        
        for i in range(num_games):
            print(f"\n{'='*60}")
            print(f"Partie {i+1}/{num_games}")
            print('='*60)
            
            # Alterner qui joue en premier
            human_first = (i % 2 == 0)
            position = "premier" if human_first else "second"
            print(f"Vous jouez en {position}")
            
            payoff = self.play_game(human_first)
            total_payoff += payoff
            
            if payoff > 0:
                wins += 1
            elif payoff < 0:
                losses += 1
            else:
                ties += 1
            
            if i < num_games - 1:
                cont = input("\nAppuyez sur Entrée pour continuer (ou 'q' pour quitter)... ")
                if cont.lower() == 'q':
                    num_games = i + 1
                    break
        
        # Statistiques finales
        print("\n" + "="*60)
        print("STATISTIQUES FINALES")
        print("="*60)
        print(f"Parties jouées: {num_games}")
        print(f"Victoires: {wins}")
        print(f"Défaites: {losses}")
        print(f"Égalités: {ties}")
        print(f"Gain total: {total_payoff:+d} chips")
        print(f"Gain moyen: {total_payoff/num_games:+.2f} chips/partie")
        
        win_rate = (wins / num_games) * 100
        print(f"Taux de victoire: {win_rate:.1f}%")
        
        print("\n" + "="*60)
        print("ANALYSE")
        print("="*60)
        print("""
L'IA joue selon l'équilibre de Nash approximé.
Un gain moyen proche de 0 indique que l'IA joue de manière optimale.
Si vous gagnez régulièrement, l'IA n'a pas assez convergé
ou vous exploitez des patterns dans sa stratégie!
        """)


def main():
    """Fonction principale"""
    print("\n" + "="*60)
    print("JEU INTERACTIF - KUHN POKER vs IA")
    print("="*60)
    
    print("\nChargement de l'IA...")
    print("Entraînement en cours (cela peut prendre quelques secondes)...\n")
    
    # Entraîner l'IA
    trainer = CFRTrainer()
    trainer.train(iterations=50000)
    
    print("✓ IA prête!\n")
    
    # Créer le joueur interactif
    player = InteractivePlayer(trainer)
    
    # Menu
    while True:
        print("\n" + "="*60)
        print("MENU")
        print("="*60)
        print("1. Jouer une partie")
        print("2. Jouer une session (plusieurs parties)")
        print("3. Voir la stratégie de l'IA")
        print("4. Quitter")
        
        choice = input("\nVotre choix: ")
        
        if choice == '1':
            player.play_game(human_first=True)
        elif choice == '2':
            try:
                num = int(input("Nombre de parties (défaut=10): ") or "10")
                player.play_session(num_games=num)
            except ValueError:
                print("Nombre invalide, utilisation de 10 parties.")
                player.play_session(num_games=10)
        elif choice == '3':
            trainer.display_strategy()
        elif choice == '4':
            print("\nMerci d'avoir joué! Au revoir!")
            break
        else:
            print("Choix invalide.")


if __name__ == "__main__":
    main()
