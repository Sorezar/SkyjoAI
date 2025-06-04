import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class UnsupervisedDeepAI(BaseAI):
    """
    IA Deep Learning non supervisée utilisant un auto-encodeur pour découvrir
    des représentations latentes de l'état du jeu et développer des stratégies émergentes.
    """
    
    def __init__(self, model_path="deep_models/unsupervised_skyjo.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Hyperparamètres
        self.state_dim = 180  # Dimension de l'état complet
        self.latent_dim = 32  # Dimension de l'espace latent
        self.hidden_dim = 128
        
        # Modèles neuraux
        self.autoencoder = StateAutoencoder(self.state_dim, self.latent_dim, self.hidden_dim).to(self.device)
        self.strategy_network = StrategyNetwork(self.latent_dim, 64).to(self.device)
        
        # Optimiseurs
        self.ae_optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        self.strategy_optimizer = optim.Adam(self.strategy_network.parameters(), lr=0.0005)
        
        # Mémoire d'expérience
        self.experience_buffer = deque(maxlen=10000)
        self.state_history = deque(maxlen=1000)
        
        # Métriques d'apprentissage
        self.reconstruction_losses = []
        self.strategy_rewards = []
        
        self.load_models()
    
    def save_models(self):
        """Sauvegarde les modèles entraînés"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            'autoencoder': self.autoencoder.state_dict(),
            'strategy_network': self.strategy_network.state_dict(),
            'ae_optimizer': self.ae_optimizer.state_dict(),
            'strategy_optimizer': self.strategy_optimizer.state_dict(),
            'reconstruction_losses': self.reconstruction_losses,
            'strategy_rewards': self.strategy_rewards
        }, self.model_path)
    
    def load_models(self):
        """Charge les modèles pré-entraînés s'ils existent"""
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.autoencoder.load_state_dict(checkpoint['autoencoder'])
                self.strategy_network.load_state_dict(checkpoint['strategy_network'])
                self.ae_optimizer.load_state_dict(checkpoint['ae_optimizer'])
                self.strategy_optimizer.load_state_dict(checkpoint['strategy_optimizer'])
                self.reconstruction_losses = checkpoint.get('reconstruction_losses', [])
                self.strategy_rewards = checkpoint.get('strategy_rewards', [])
                print("✅ Modèles Deep Learning chargés avec succès")
            except Exception as e:
                print(f"❌ Erreur lors du chargement: {e}")
    
    def encode_game_state(self, grid, discard, other_p_grids):
        """Encode l'état complet du jeu en vecteur numérique"""
        state = []
        
        # Vérification défensive de la grille
        if not grid or len(grid) == 0 or any(len(row) == 0 for row in grid):
            # Retourner un état par défaut si la grille n'est pas initialisée
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # Encoder notre grille (48 features)
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                if i < len(grid) and j < len(grid[i]) and grid[i][j] is not None:
                    if grid[i][j].revealed:
                        # Normaliser les valeurs de cartes [-2, 12] -> [-1, 1]
                        normalized_value = (grid[i][j].value + 2) / 14.0 * 2 - 1
                        state.extend([1.0, normalized_value, i/2.0, j/3.0])  # revealed, value, pos_x, pos_y
                    else:
                        state.extend([0.0, 0.0, i/2.0, j/3.0])  # non révélée
                else:
                    state.extend([0.0, 0.0, i/2.0, j/3.0])  # Position non initialisée
        
        # Encoder la défausse (20 features)
        if discard and len(discard) > 0:
            recent_discards = discard[-5:] if len(discard) >= 5 else discard
            for card in recent_discards:
                normalized_value = (card.value + 2) / 14.0 * 2 - 1
                state.extend([normalized_value, 1.0])  # valeur, présence
            
            # Padding pour la défausse
            for _ in range(5 - len(recent_discards)):
                state.extend([0.0, 0.0])
        else:
            # Pas de défausse
            state.extend([0.0, 0.0] * 5)
        
        # Encoder les adversaires (112 features : 3 adversaires * ~37 features)
        for opp_idx, opp_grid in enumerate(other_p_grids[:3]):
            if not opp_grid or len(opp_grid) == 0:
                state.extend([0.0] * 37)
                continue
                
            # Grille adversaire simplifiée (24 features)
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    if (i < len(opp_grid) and j < len(opp_grid[i]) and 
                        opp_grid[i][j] is not None and hasattr(opp_grid[i][j], 'revealed')):
                        if opp_grid[i][j].revealed:
                            normalized_value = (opp_grid[i][j].value + 2) / 14.0 * 2 - 1
                            state.extend([1.0, normalized_value])
                        else:
                            state.extend([0.0, 0.0])
                    else:
                        state.extend([0.0, 0.0])
            
            # Statistiques adversaire (13 features)
            opp_revealed_count = 0
            opp_revealed_sum = 0
            
            for row in opp_grid:
                if not row:
                    continue
                for card in row:
                    if (card is not None and hasattr(card, 'revealed') and 
                        hasattr(card, 'value') and card.revealed):
                        opp_revealed_count += 1
                        opp_revealed_sum += card.value
            
            opp_progress = opp_revealed_count / (GRID_ROWS * GRID_COLS) if (GRID_ROWS * GRID_COLS) > 0 else 0
            
            state.extend([
                opp_progress,  # Progression
                opp_revealed_sum / 100.0,  # Score normalisé
                opp_revealed_count / 12.0,  # Nombre de cartes révélées normalisé
            ])
            
            # Analyse par colonne adversaire (10 features)
            for col in range(GRID_COLS):
                col_revealed = 0
                col_complete = 0.0
                
                if col < len(opp_grid[0]) if opp_grid and len(opp_grid) > 0 else False:
                    for row in range(len(opp_grid)):
                        if (row < len(opp_grid) and col < len(opp_grid[row]) and 
                            opp_grid[row][col] is not None and 
                            hasattr(opp_grid[row][col], 'revealed') and 
                            opp_grid[row][col].revealed):
                            col_revealed += 1
                    
                    col_complete = 1.0 if col_revealed == GRID_ROWS else 0.0
                    col_progress = col_revealed / GRID_ROWS
                    state.extend([col_progress, col_complete])
                else:
                    state.extend([0.0, 0.0])
        
        # Padding si moins de 3 adversaires
        for _ in range(3 - min(3, len(other_p_grids))):
            state.extend([0.0] * 37)
        
        # Assurer que l'état a exactement la bonne dimension
        while len(state) < self.state_dim:
            state.append(0.0)
        state = state[:self.state_dim]
        
        return np.array(state, dtype=np.float32)
    
    def get_latent_representation(self, state):
        """Obtient la représentation latente de l'état via l'auto-encodeur"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent, _ = self.autoencoder(state_tensor)
        return latent.cpu().numpy().flatten()
    
    def update_autoencoder(self, state_batch):
        """Met à jour l'auto-encodeur avec un batch d'états"""
        if len(state_batch) < 4:  # Besoin d'un minimum d'échantillons
            return 0.0
        
        states = torch.FloatTensor(np.array(state_batch)).to(self.device)
        
        # Forward pass
        latent, reconstructed = self.autoencoder(states)
        
        # Loss de reconstruction
        reconstruction_loss = F.mse_loss(reconstructed, states)
        
        # Loss de régularisation (pour éviter l'effondrement latent)
        latent_reg = torch.mean(torch.abs(latent))
        
        total_loss = reconstruction_loss + 0.01 * latent_reg
        
        # Backward pass
        self.ae_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1.0)
        self.ae_optimizer.step()
        
        self.reconstruction_losses.append(total_loss.item())
        return total_loss.item()
    
    def train_unsupervised(self, num_episodes=100):
        """Entraîne le modèle de manière non supervisée"""
        print(f"🧠 Début de l'entraînement non supervisé pour {num_episodes} épisodes...")
        
        from ai.initial import InitialAI
        from core.game import SkyjoGame, Scoreboard
        from core.player import Player
        
        # Configuration de l'environnement d'entraînement
        batch_size = 32
        
        for episode in range(num_episodes):
            # Créer une partie d'entraînement
            players = [Player(i, f"TrainAI_{i}", InitialAI()) for i in range(4)]
            scoreboard = Scoreboard(players)
            game = SkyjoGame(players, scoreboard)
            
            episode_states = []
            
            # Jouer la partie et collecter les états
            while not game.finished:
                if not game.round_over:
                    current_player = game.players[game.current_player_index]
                    other_grids = [p.grid for i, p in enumerate(game.players) if i != game.current_player_index]
                    
                    # Encoder l'état actuel
                    state = self.encode_game_state(current_player.grid, game.discard, other_grids)
                    episode_states.append(state)
                    self.state_history.append(state)
                
                game.step()
            
            # Mettre à jour l'auto-encodeur avec les états collectés
            if len(self.state_history) >= batch_size:
                batch_states = list(self.state_history)[-batch_size:]
                loss = self.update_autoencoder(batch_states)
                
                if episode % 10 == 0:
                    print(f"📊 Épisode {episode}, Loss reconstruction: {loss:.4f}")
            
            # Sauvegarder périodiquement
            if episode % 50 == 0 and episode > 0:
                self.save_models()
        
        print("✅ Entraînement non supervisé terminé!")
        self.save_models()
    
    def initial_flip(self):
        """Stratégie initiale basée sur l'analyse latente"""
        # Générer plusieurs configurations initiales et choisir la meilleure selon l'espace latent
        candidates = [
            [[0, 0], [2, 3]],  # Coins opposés
            [[0, 1], [2, 2]],  # Lignes différentes
            [[1, 0], [1, 3]],  # Même ligne, extrémités
            [[0, 0], [1, 1]],  # Proximité
        ]
        
        # Pour l'instant, utiliser une heuristique simple
        return random.choice(candidates)
    
    def choose_source(self, grid, discard, other_p_grids):
        """Choix de source basé sur l'analyse de l'espace latent"""
        if not discard:
            return 'P'
        
        # Encoder l'état actuel
        current_state = self.encode_game_state(grid, discard, other_p_grids)
        latent_repr = self.get_latent_representation(current_state)
        
        # Utiliser le réseau de stratégie pour décider
        latent_tensor = torch.FloatTensor(latent_repr).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            strategy_output = self.strategy_network(latent_tensor)
            source_prob = torch.sigmoid(strategy_output[0, 0]).item()  # Probabilité de prendre la défausse
        
        # Ajouter un biais basé sur la valeur de la carte
        discard_value = discard[-1].value if discard and len(discard) > 0 else 5
        value_bias = max(0, (5 - discard_value) / 7.0)  # Bias positif pour les petites valeurs
        
        final_prob = (source_prob + value_bias) / 2.0
        
        return 'D' if final_prob > 0.5 else 'P'
    
    def choose_keep(self, card, grid, other_p_grids):
        """Décision de garde basée sur l'analyse latente"""
        # Simuler l'état avec la carte pour l'analyse
        temp_discard = [card]  # Simuler comme si c'était dans la défausse
        current_state = self.encode_game_state(grid, temp_discard, other_p_grids)
        latent_repr = self.get_latent_representation(current_state)
        
        latent_tensor = torch.FloatTensor(latent_repr).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            strategy_output = self.strategy_network(latent_tensor)
            keep_prob = torch.sigmoid(strategy_output[0, 1]).item()  # Probabilité de garder
        
        # Biais basé sur la valeur de la carte
        value_bias = max(0, (4 - card.value) / 6.0)
        final_prob = (keep_prob + value_bias) / 2.0
        
        return final_prob > 0.5
    
    def choose_position(self, card, grid, other_p_grids):
        """Choix de position avec optimisation latente"""
        try:
            if not grid or len(grid) == 0:
                return (0, 0)
                
            current_state = self.encode_game_state(grid, [card], other_p_grids)
            latent_repr = self.get_latent_representation(current_state)
            
            available_positions = []
            position_scores = []
            
            # Évaluer toutes les positions possibles avec vérifications défensives
            for i in range(min(len(grid), GRID_ROWS)):
                if not grid[i] or len(grid[i]) == 0:
                    continue
                for j in range(min(len(grid[i]), GRID_COLS)):
                    if grid[i][j] is not None:  # Position valide
                        available_positions.append((i, j))
                        
                        try:
                            # Score basé sur la position et l'analyse latente
                            position_features = np.append(latent_repr, [i/2.0, j/3.0, card.value/14.0])
                            position_tensor = torch.FloatTensor(position_features).unsqueeze(0).to(self.device)
                            
                            with torch.no_grad():
                                if hasattr(self.strategy_network, 'position_evaluator'):
                                    score = self.strategy_network.position_evaluator(position_tensor).item()
                                else:
                                    # Fallback: utiliser les sorties existantes
                                    strategy_out = self.strategy_network(torch.FloatTensor(latent_repr).unsqueeze(0).to(self.device))
                                    score = strategy_out[0, 2].item() + (i + j) * 0.1  # Biais simple
                            
                            # Bonus pour remplacer des cartes de valeur élevée
                            if (hasattr(grid[i][j], 'revealed') and hasattr(grid[i][j], 'value') and
                                grid[i][j].revealed and grid[i][j].value > card.value):
                                score += (grid[i][j].value - card.value) * 0.5
                            
                            position_scores.append(score)
                        except (IndexError, AttributeError, TypeError):
                            # En cas d'erreur, score par défaut
                            position_scores.append(0.0)
            
            if available_positions and position_scores:
                best_idx = np.argmax(position_scores)
                return available_positions[best_idx]
            
            # Fallback sécurisé
            return (0, 0) if grid and len(grid) > 0 and len(grid[0]) > 0 else None
            
        except (IndexError, AttributeError, TypeError):
            # En cas d'erreur majeure, retour sécurisé
            return (0, 0) if grid and len(grid) > 0 and len(grid[0]) > 0 else None
    
    def choose_reveal(self, grid):
        """Choix de révélation avec analyse de l'espace latent"""
        try:
            if not grid or len(grid) == 0:
                return None
                
            current_state = self.encode_game_state(grid, [], [])
            latent_repr = self.get_latent_representation(current_state)
            
            unrevealed_positions = []
            
            # Collecte sécurisée des positions non révélées
            for i in range(min(len(grid), GRID_ROWS)):
                if not grid[i] or len(grid[i]) == 0:
                    continue
                for j in range(min(len(grid[i]), GRID_COLS)):
                    if (grid[i][j] is not None and hasattr(grid[i][j], 'revealed') and 
                        not grid[i][j].revealed):
                        unrevealed_positions.append((i, j))
            
            if not unrevealed_positions:
                return None
            
            best_position = unrevealed_positions[0]
            best_score = float('-inf')
            
            for i, j in unrevealed_positions:
                try:
                    # Analyser le potentiel de chaque position avec l'espace latent
                    position_features = np.append(latent_repr, [i/2.0, j/3.0])
                    score = np.sum(position_features * np.random.normal(0, 0.1, len(position_features)))  # Exploration stochastique
                    
                    # Bonus pour compléter les colonnes - avec vérifications défensives
                    col_revealed = 0
                    for row in range(min(len(grid), GRID_ROWS)):
                        if (j < len(grid[row]) and grid[row][j] is not None and 
                            hasattr(grid[row][j], 'revealed') and grid[row][j].revealed):
                            col_revealed += 1
                    
                    score += col_revealed * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_position = (i, j)
                        
                except (IndexError, AttributeError, TypeError):
                    # En cas d'erreur, continuer avec le score actuel
                    continue
            
            return best_position
            
        except (IndexError, AttributeError, TypeError):
            # En cas d'erreur majeure, retour sécurisé
            try:
                if grid and len(grid) > 0 and len(grid[0]) > 0:
                    return (0, 0)
            except:
                return None


class StateAutoencoder(nn.Module):
    """Auto-encodeur pour apprendre les représentations latentes de l'état du jeu"""
    
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(StateAutoencoder, self).__init__()
        
        # Encodeur
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.Tanh()  # Contraindre l'espace latent
        )
        
        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Normaliser la sortie
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


class StrategyNetwork(nn.Module):
    """Réseau de stratégie pour prendre des décisions basées sur l'espace latent"""
    
    def __init__(self, latent_dim, hidden_dim):
        super(StrategyNetwork, self).__init__()
        
        self.strategy_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 4 sorties pour différentes décisions
        )
    
    def forward(self, latent):
        return self.strategy_head(latent)


def train_unsupervised_model():
    """Fonction utilitaire pour entraîner le modèle non supervisé"""
    print("🚀 Lancement de l'entraînement du modèle Deep Learning non supervisé...")
    
    deep_ai = UnsupervisedDeepAI()
    deep_ai.train_unsupervised(num_episodes=200)
    
    print("✅ Entraînement terminé! Modèle sauvegardé.")
    return deep_ai 