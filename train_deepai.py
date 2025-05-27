import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from collections import deque

from config.config import GRID_ROWS, GRID_COLS, MAX_POINTS
from core.game import SkyjoGame, Scoreboard, Card
from core.player import Player
from ai.deepai import DeepAI
from ai.random import RandomAI
from ai.initial import InitialAI

class TrainingEnvironment:
    """Environnement d'entraînement pour DeepAI"""
    
    def __init__(self, deep_ai, opponents):
        self.deep_ai = deep_ai
        self.opponents = opponents
        self.reset()
        
    def reset(self):
        """Réinitialise l'environnement pour une nouvelle partie"""
        # Créer les joueurs
        self.players = [
            Player(0, "DeepAI", self.deep_ai),
            Player(1, "Opponent1", self.opponents[0]),
            Player(2, "Opponent2", self.opponents[1]),
            Player(3, "Opponent3", self.opponents[2])
        ]
        
        self.scoreboard = Scoreboard(self.players)
        self.game = SkyjoGame(self.players, self.scoreboard)
        
        # État pour le calcul des récompenses
        self.previous_score = 0
        self.previous_grid_size = GRID_ROWS * GRID_COLS
        
    def calculate_reward(self, player_index, action_type):
        """Calcule la récompense pour l'action effectuée"""
        reward = 0
        player = self.players[player_index]
        
        # Score actuel du joueur
        current_score = sum(c.value for row in player.grid for c in row if c.revealed) + 5 * sum(c.value for row in player.grid for c in row if not c.revealed)
        score_diff = self.previous_score - current_score
        
        # Récompense basée sur le changement de score
        if score_diff > 0:
            reward += score_diff  # Bonus pour réduction du score
        else:
            reward -= score_diff * 2 # Malus pour augmentation du score (# BUG on veut que l'IA réduise son score pas l'augmente)
        
        # Bonus pour suppression de colonne
        current_grid_size = len(player.grid) * len(player.grid[0]) if player.grid else 0
        if current_grid_size < self.previous_grid_size:
            columns_removed = (self.previous_grid_size - current_grid_size) // GRID_ROWS
            reward += columns_removed * 20  # Gros bonus pour suppression de colonne
        
        # Malus si on finit premier sans le meilleur score
        if self.game.round_over and self.game.first_finisher == player_index:
            all_scores = [p.round_score() for p in self.players]
            if all_scores[player_index] != min(all_scores):
                reward -= 50  # Gros malus
        
        # Petit bonus pour révéler des cartes
        if action_type == 'reveal':
            reward += 5
            
        # Mettre à jour les états précédents
        self.previous_score = current_score
        self.previous_grid_size = current_grid_size
        
        return reward
    
    def step(self, player_index=0):
        """Effectue un pas dans l'environnement"""
        if player_index != 0:  # Adversaires
            self.game.step()
            return None, 0, False
        
        # Sauvegarder l'état avant l'action
        player = self.players[player_index]
        other_grids = [self.players[i].grid for i in range(len(self.players)) if i != player_index]
        
        # État initial
        state_before = self.deep_ai.get_state(player.grid, self.game.discard, other_grids)
        
        # Variables pour tracker l'action
        action_type = None
        action_value = None
        card_value = None  # Pour stocker la valeur de la carte piochée
        
        # Intercepter les décisions pour l'apprentissage
        original_source = self.deep_ai.choose_source
        original_keep = self.deep_ai.choose_keep
        original_position = self.deep_ai.choose_position
        original_reveal = self.deep_ai.choose_reveal
        
        def wrapped_source(grid, discard, other_p_grids):
            nonlocal action_type, action_value
            action_type = 'source'
            result = original_source(grid, discard, other_p_grids)
            action_value = 0 if result == 'P' else 1
            return result
        
        def wrapped_keep(card, grid, other_p_grids):
            nonlocal action_type, action_value, card_value
            action_type = 'keep'
            card_value = card.value  # Sauvegarder la valeur de la carte
            result = original_keep(card, grid, other_p_grids)
            action_value = 1 if result else 0
            return result
        
        def wrapped_position(card, grid, other_p_grids):
            nonlocal action_type, action_value, card_value
            action_type = 'position'
            card_value = card.value  # Sauvegarder la valeur de la carte
            result = original_position(card, grid, other_p_grids)
            # Utiliser GRID_COLS au lieu de len(grid[0]) pour l'index
            action_value = result[0] * GRID_COLS + result[1]
            return result
        
        def wrapped_reveal(grid):
            nonlocal action_type, action_value
            action_type = 'reveal'
            result = original_reveal(grid)
            # Utiliser GRID_COLS au lieu de len(grid[0]) pour l'index
            action_value = result[0] * GRID_COLS + result[1]
            return result
        
        # Remplacer temporairement les méthodes
        self.deep_ai.choose_source = wrapped_source
        self.deep_ai.choose_keep = wrapped_keep
        self.deep_ai.choose_position = wrapped_position
        self.deep_ai.choose_reveal = wrapped_reveal
        
        # Effectuer l'action
        self.game.step()
        
        # Restaurer les méthodes originales
        self.deep_ai.choose_source = original_source
        self.deep_ai.choose_keep = original_keep
        self.deep_ai.choose_position = original_position
        self.deep_ai.choose_reveal = original_reveal
        
        # Calculer la récompense
        reward = self.calculate_reward(player_index, action_type)
        
        # État après l'action
        state_after = self.deep_ai.get_state(player.grid, self.game.discard, other_grids)
        
        # Vérifier si la partie est terminée
        done = self.game.finished
        
        # Stocker l'expérience avec l'état approprié
        if action_type and action_value is not None:
            # Pour keep et position, ajouter la valeur de la carte à l'état
            if action_type in ['keep', 'position'] and card_value is not None:
                state_before_with_card = np.append(state_before, card_value)
                state_after_with_card = np.append(state_after, card_value)
                self.deep_ai.remember(state_before_with_card, action_value, reward, state_after_with_card, done, action_type)
            else:
                self.deep_ai.remember(state_before, action_value, reward, state_after, done, action_type)
        
        return state_after, reward, done

def train_deepai(episodes=1000):
    """Fonction principale d'entraînement"""
    # Initialiser l'IA
    deep_ai = DeepAI(learning_rate=0.001, epsilon=1.0, epsilon_decay=0.995)
    
    # Adversaires
    opponents = [InitialAI(), InitialAI(), InitialAI()]
    
    # Environnement
    env = TrainingEnvironment(deep_ai, opponents)
    
    # Métriques
    episode_rewards = []
    episode_scores = []
    episode_wins = []
    moving_avg_rewards = deque(maxlen=100)
    
    # Boucle d'entraînement
    for episode in tqdm(range(episodes), desc="Entraînement"):
        env.reset()
        total_reward = 0
        steps = 0
        
        # Jouer une partie complète
        while not env.game.finished:
            # Tour de DeepAI
            if env.game.current_player_index == 0:
                _, reward, done = env.step(0)
                total_reward += reward
                steps += 1
                
                # Entraîner périodiquement
                if steps % 4 == 0:
                    deep_ai.replay()
            else:
                # Tours des adversaires
                env.step(env.game.current_player_index)
        
        # Métriques de fin de partie
        final_scores = env.scoreboard.total_scores
        deep_ai_score = final_scores[0]
        deep_ai_won = deep_ai_score == min(final_scores)
        
        episode_rewards.append(total_reward)
        episode_scores.append(deep_ai_score)
        episode_wins.append(1 if deep_ai_won else 0)
        moving_avg_rewards.append(total_reward)
        
        # Mise à jour d'epsilon
        deep_ai.update_epsilon()
        
        # Affichage périodique
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(moving_avg_rewards)
            win_rate = np.mean(episode_wins[-100:]) * 100
            avg_score = np.mean(episode_scores[-100:])
            
            print(f"\nÉpisode {episode + 1}/{episodes}")
            print(f"Récompense moyenne (100 derniers): {avg_reward:.2f}")
            print(f"Taux de victoire (100 derniers): {win_rate:.1f}%")
            print(f"Score moyen (100 derniers): {avg_score:.1f}")
            print(f"Epsilon: {deep_ai.epsilon:.3f}")
    
    # Sauvegarder le modèle
    torch.save({
        'source_net': deep_ai.source_net.state_dict(),
        'keep_net': deep_ai.keep_net.state_dict(),
        'position_net': deep_ai.position_net.state_dict(),
        'reveal_net': deep_ai.reveal_net.state_dict(),
        'epsilon': deep_ai.epsilon
    }, 'deepai_model.pth')
    
    # Graphiques de performance
    plt.figure(figsize=(15, 5))
    
    # Récompenses
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards, alpha=0.3, label='Récompense par épisode')
    plt.plot(np.convolve(episode_rewards, np.ones(100)/100, mode='valid'), label='Moyenne mobile (100)')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense totale')
    plt.title('Évolution des récompenses')
    plt.legend()
    
    # Scores
    plt.subplot(1, 3, 2)
    plt.plot(episode_scores, alpha=0.3, label='Score par épisode')
    plt.plot(np.convolve(episode_scores, np.ones(100)/100, mode='valid'), label='Moyenne mobile (100)')
    plt.xlabel('Épisode')
    plt.ylabel('Score final')
    plt.title('Évolution des scores')
    plt.legend()
    
    # Taux de victoire
    plt.subplot(1, 3, 3)
    win_rate_moving = [np.mean(episode_wins[max(0, i-100):i+1]) * 100 for i in range(len(episode_wins))]
    plt.plot(win_rate_moving)
    plt.xlabel('Épisode')
    plt.ylabel('Taux de victoire (%)')
    plt.title('Évolution du taux de victoire')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()
    
    print("\nEntraînement terminé!")
    print(f"Modèle sauvegardé dans 'deepai_model.pth'")
    print(f"Graphiques sauvegardés dans 'training_metrics.png'")
    
    return deep_ai

def test_deepai(deep_ai, num_games=100):
    """Teste l'IA entraînée contre différents adversaires"""
    deep_ai.set_training(False)  # Mode évaluation
    
    # Test contre différents types d'adversaires
    test_configs = [
        ("3 RandomAI", [RandomAI(), RandomAI(), RandomAI()]),
        ("3 InitialAI", [InitialAI(), InitialAI(), InitialAI()]),
        ("Mixte", [RandomAI(), InitialAI(), RandomAI()])
    ]
    
    for config_name, opponents in test_configs:
        wins = 0
        total_score = 0
        
        print(f"\nTest contre {config_name}...")
        
        for _ in tqdm(range(num_games)):
            env = TrainingEnvironment(deep_ai, opponents)
            
            while not env.game.finished:
                env.step(env.game.current_player_index)
            
            final_scores = env.scoreboard.total_scores
            if final_scores[0] == min(final_scores):
                wins += 1
            total_score += final_scores[0]
        
        print(f"Taux de victoire: {wins/num_games*100:.1f}%")
        print(f"Score moyen: {total_score/num_games:.1f}")

if __name__ == "__main__":
    # Entraîner l'IA
    print("Démarrage de l'entraînement de DeepAI...")
    trained_ai = train_deepai(episodes=1000)
    
    # Tester l'IA
    print("\nTest de l'IA entraînée...")
    test_deepai(trained_ai, num_games=100)
