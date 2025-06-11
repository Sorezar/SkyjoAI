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
        self.ae_optimizer.step()
        
        self.reconstruction_losses.append(reconstruction_loss.item())
        return total_loss.item()
    
    def train_unsupervised(self, num_episodes=100):
        """Entraînement non supervisé sur des parties simulées"""
        print("🧠 Début de l'entraînement non supervisé...")
        
        for episode in range(num_episodes):
            # Simuler une partie et collecter les états
            episode_states = []
            
            # État initial aléatoire
            for _ in range(50):  # 50 états par épisode
                # Générer un état de jeu aléatoire mais réaliste
                grid = [[{"value": random.randint(-2, 12), "revealed": random.choice([True, False])} 
                        for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
                
                discard = [{"value": random.randint(-2, 12)} for _ in range(random.randint(1, 10))]
                
                other_grids = [
                    [[{"value": random.randint(-2, 12), "revealed": random.choice([True, False])} 
                      for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
                    for _ in range(random.randint(1, 3))
                ]
                
                state = self.encode_game_state(grid, discard, other_grids)
                episode_states.append(state)
            
            # Entraîner l'auto-encodeur sur ce batch
            if len(episode_states) >= 4:
                loss = self.update_autoencoder(episode_states)
                
                if episode % 10 == 0:
                    print(f"📊 Épisode {episode}/{num_episodes}, Loss: {loss:.4f}")
        
        self.save_models()
        print("✅ Entraînement non supervisé terminé!")
    
    def initial_flip(self):
        """Sélection initiale basée sur l'analyse de l'espace latent"""
        # Positions candidates
        positions = [[0, 0], [0, GRID_COLS-1], [GRID_ROWS-1, 0], [GRID_ROWS-1, GRID_COLS-1]]
        
        # Pour l'instant, utiliser une stratégie simple
        return random.sample(positions, 2)
    
    def choose_source(self, grid, discard, other_p_grids):
        """Choix de source basé sur l'analyse de l'espace latent"""
        try:
            state = self.encode_game_state(grid, discard, other_p_grids)
            self.state_history.append(state)
            
            latent = self.get_latent_representation(state)
            
            # Utiliser le réseau de stratégie pour la décision
            with torch.no_grad():
                latent_tensor = torch.FloatTensor(latent).unsqueeze(0).to(self.device)
                strategy_output = self.strategy_network(latent_tensor)
                source_prob = torch.sigmoid(strategy_output[0]).item()
            
            # Si probabilité > 0.5, choisir défausse, sinon pioche
            return 'D' if source_prob > 0.5 and discard else 'P'
            
        except Exception as e:
            # Fallback vers stratégie simple
            if discard and len(discard) > 0:
                return 'D' if discard[-1].value <= 3 else 'P'
            return 'P'
    
    def choose_keep(self, card, grid, other_p_grids):
        """Décision de garder basée sur l'espace latent"""
        try:
            # Simuler la défausse pour analyser l'impact
            discard_dummy = [card]
            state = self.encode_game_state(grid, discard_dummy, other_p_grids)
            
            latent = self.get_latent_representation(state)
            
            with torch.no_grad():
                latent_tensor = torch.FloatTensor(latent).unsqueeze(0).to(self.device)
                strategy_output = self.strategy_network(latent_tensor)
                keep_prob = torch.sigmoid(strategy_output[1]).item()
            
            return keep_prob > 0.4  # Seuil légèrement conservateur
            
        except Exception as e:
            # Fallback
            return card.value <= 4
    
    def choose_position(self, card, grid, other_p_grids):
        """Choix de position optimisé par deep learning"""
        try:
            best_position = (0, 0)
            best_score = float('-inf')
            
            # Évaluer chaque position possible
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    if (i < len(grid) and j < len(grid[i]) and 
                        grid[i][j] is not None):
                        
                        # Simuler le placement de la carte
                        original_card = grid[i][j]
                        grid[i][j] = card  # Placement temporaire
                        
                        # Encoder l'état résultant
                        state = self.encode_game_state(grid, [], other_p_grids)
                        latent = self.get_latent_representation(state)
                        
                        # Évaluer avec le réseau de stratégie
                        with torch.no_grad():
                            latent_tensor = torch.FloatTensor(latent).unsqueeze(0).to(self.device)
                            strategy_output = self.strategy_network(latent_tensor)
                            position_score = strategy_output[2].item()
                        
                        # Restaurer l'état original
                        grid[i][j] = original_card
                        
                        # Bonus pour l'amélioration directe
                        improvement = original_card.value - card.value
                        total_score = position_score + improvement * 0.1
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_position = (i, j)
            
            return best_position
            
        except Exception as e:
            # Fallback vers position avec amélioration maximale
            best_pos = (0, 0)
            best_improvement = float('-inf')
            
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    if (i < len(grid) and j < len(grid[i]) and 
                        grid[i][j] is not None):
                        improvement = grid[i][j].value - card.value
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_pos = (i, j)
            
            return best_pos
    
    def choose_reveal(self, grid):
        """Choix de révélation basé sur l'apprentissage non supervisé"""
        try:
            best_position = (0, 0)
            best_score = float('-inf')
            
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    if (i < len(grid) and j < len(grid[i]) and 
                        grid[i][j] is not None and not grid[i][j].revealed):
                        
                        # Simuler la révélation
                        grid[i][j].revealed = True
                        
                        # Encoder l'état et obtenir le score latent
                        state = self.encode_game_state(grid, [], [])
                        latent = self.get_latent_representation(state)
                        
                        # Utiliser une heuristique basée sur la variance latente
                        latent_variance = np.var(latent)
                        position_score = latent_variance
                        
                        # Bonus pour les positions stratégiques
                        if (i == 0 or i == GRID_ROWS-1) and (j == 0 or j == GRID_COLS-1):
                            position_score += 0.1  # Coins
                        
                        # Restaurer l'état
                        grid[i][j].revealed = False
                        
                        if position_score > best_score:
                            best_score = position_score
                            best_position = (i, j)
            
            return best_position
            
        except Exception as e:
            # Fallback vers les coins
            corners = [(0, 0), (0, GRID_COLS-1), (GRID_ROWS-1, 0), (GRID_ROWS-1, GRID_COLS-1)]
            
            for pos in corners:
                i, j = pos
                if (i < len(grid) and j < len(grid[i]) and 
                    grid[i][j] is not None and not grid[i][j].revealed):
                    return pos
            
            # Si aucun coin disponible, prendre la première position non révélée
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    if (i < len(grid) and j < len(grid[i]) and 
                        grid[i][j] is not None and not grid[i][j].revealed):
                        return (i, j)
            
            return (0, 0)


class StateAutoencoder(nn.Module):
    """Auto-encodeur pour apprendre des représentations latentes de l'état du jeu"""
    
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(StateAutoencoder, self).__init__()
        
        # Encodeur
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.Tanh()  # Contraindre l'espace latent
        )
        
        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


class StrategyNetwork(nn.Module):
    """Réseau de stratégie qui utilise l'espace latent pour prendre des décisions"""
    
    def __init__(self, latent_dim, hidden_dim):
        super(StrategyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # 3 sorties : source, keep, position_value
        )
    
    def forward(self, latent):
        return self.network(latent)


def train_unsupervised_model():
    """Fonction utilitaire pour entraîner le modèle non supervisé"""
    ai = UnsupervisedDeepAI()
    ai.train_unsupervised(num_episodes=200)
    print("🎯 Modèle Deep Learning entraîné et sauvegardé!")


if __name__ == "__main__":
    train_unsupervised_model() 