import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS
import json
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
import random
from core.game import SkyjoGame, Scoreboard
from core.player import Player
from ai.initial import InitialAI

# Constantes pour les dimensions
GRID_SIZE = GRID_ROWS * GRID_COLS  # 12 cartes
MAX_PLAYERS = 8
TOTAL_GRID_SIZE = GRID_SIZE * MAX_PLAYERS  # 96 entrées totales (12 * 8)

# Constantes pour les valeurs spéciales
CARD_UNKNOWN = -100  # Valeur pour une carte non révélée
CARD_EMPTY = -200    # Valeur pour un emplacement vide (colonne supprimée)
MIN_CARD_VALUE = -2  # Valeur minimale d'une carte
MAX_CARD_VALUE = 12  # Valeur maximale d'une carte

class InputNormalizer(nn.Module):
    def __init__(self):
        super(InputNormalizer, self).__init__()
        # Normalisation pour les valeurs de cartes (-2 à 12)
        self.card_normalizer = nn.LayerNorm(1)
        # Normalisation pour les valeurs spéciales
        self.special_normalizer = nn.LayerNorm(1)
        
    def forward(self, x):
        # Ajouter une dimension de batch si nécessaire
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ajouter une dimension de batch
            
        # Séparer les valeurs normales et spéciales
        normal_mask = (x > CARD_UNKNOWN) & (x != CARD_EMPTY)
        special_mask = ~normal_mask
        
        # Normaliser séparément
        normalized = torch.zeros_like(x)
        if normal_mask.any():
            normalized[normal_mask] = self.card_normalizer(x[normal_mask].unsqueeze(1)).squeeze(1)
        if special_mask.any():
            normalized[special_mask] = self.special_normalizer(x[special_mask].unsqueeze(1)).squeeze(1)
        
        return normalized.squeeze(0)  # Retirer la dimension de batch

class AttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_positions, input_size)
        attention_weights = self.attention(x)  # (batch_size, num_positions, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        return torch.sum(x * attention_weights, dim=1)  # (batch_size, input_size)

class BaseSkyjoNet(nn.Module):
    def __init__(self):
        super(BaseSkyjoNet, self).__init__()
        self.normalizer = InputNormalizer()
        # Traitement des grilles
        self.grid_processor = nn.Sequential(
            nn.Linear(TOTAL_GRID_SIZE, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2)
        )
    
    def process_grids(self, grids_tensor):
        # Normaliser les entrées
        normalized = self.normalizer(grids_tensor)
        # S'assurer que l'entrée est bien 2D (batch, features)
        if normalized.dim() == 1:
            normalized = normalized.unsqueeze(0)
        # Traiter les grilles
        features = self.grid_processor(normalized)
        # Retourner directement les features sans reshape ni attention
        return features.squeeze(0)

class SourceNet(BaseSkyjoNet):
    def __init__(self):
        super(SourceNet, self).__init__()
        self.decision_layers = nn.Sequential(
            nn.Linear(128 + 2, 64),  # 128 (features) + 2 (cartes)
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        grid_features = self.process_grids(x[0])
        card_info = x[1]
        
        # S'assurer que les dimensions sont correctes
        if grid_features.dim() == 1:
            grid_features = grid_features.unsqueeze(0)  # Ajouter une dimension de batch
        if card_info.dim() == 1:
            card_info = card_info.unsqueeze(0)  # Ajouter une dimension de batch
            
        # Vérifier que les dimensions correspondent
        assert grid_features.size(0) == card_info.size(0), f"Dimensions de batch incompatibles: {grid_features.size(0)} vs {card_info.size(0)}"
        
        combined = torch.cat([grid_features, card_info], dim=1)
        return self.decision_layers(combined)

class KeepNet(BaseSkyjoNet):
    def __init__(self):
        super(KeepNet, self).__init__()
        self.decision_layers = nn.Sequential(
            nn.Linear(128 + 1, 64),  # 128 (features) + 1 (carte)
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        grid_features = self.process_grids(x[0])
        card_value = x[1][0].unsqueeze(0)
        combined = torch.cat([grid_features, card_value])
        return self.decision_layers(combined)

class PositionNet(BaseSkyjoNet):
    def __init__(self):
        super(PositionNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decision_layers = nn.Sequential(
            nn.Linear(128 + 1, 64),  # 128 (features) + 1 (carte)
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),
            nn.Linear(64, GRID_SIZE)  # Sortie pour chaque position possible
        )
    
    def get_legal_positions(self, grid):
        """Retourne un masque des positions légales (True = position valide)"""
        legal_positions = torch.zeros(GRID_SIZE, device=self.device, dtype=torch.bool)
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] is not None:  # La position existe
                    legal_positions[i * len(grid[0]) + j] = True
        return legal_positions
    
    def forward(self, x, grid=None):
        grid_features = self.process_grids(x[0])
        card_value = x[1][0].unsqueeze(0)
        combined = torch.cat([grid_features, card_value])
        logits = self.decision_layers(combined)
        if grid is not None:
            legal_positions = self.get_legal_positions(grid)
            logits = logits.masked_fill(~legal_positions, float('-inf'))
        return logits

class RevealNet(BaseSkyjoNet):
    def __init__(self):
        super(RevealNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decision_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),
            nn.Linear(64, GRID_SIZE)  # Sortie pour chaque position possible
        )
    
    def get_legal_positions(self, grid):
        """Retourne un masque des positions légales (True = position valide)"""
        legal_positions = torch.zeros(GRID_SIZE, device=self.device, dtype=torch.bool)
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] is not None and not grid[i][j].revealed:  # La position existe et la carte n'est pas révélée
                    legal_positions[i * len(grid[0]) + j] = True
        return legal_positions
    
    def forward(self, x, grid=None):
        grid_features = self.process_grids(x[0])
        logits = self.decision_layers(grid_features)
        if grid is not None:
            legal_positions = self.get_legal_positions(grid)
            logits = logits.masked_fill(~legal_positions, float('-inf'))
        return logits

class TrainingMetrics:
    def __init__(self):
        self.metrics = {
            'source_accuracy': [],
            'keep_accuracy': [],
            'position_accuracy': [],
            'reveal_accuracy': [],
            'source_loss': [],
            'keep_loss': [],
            'position_loss': [],
            'reveal_loss': [],
            'total_reward': [],
            'episode_length': [],
            'win_rate': [],
            'average_score': [],
            'average_turns': []
        }
        self.current_episode = {
            'source_correct': 0,
            'source_total': 0,
            'keep_correct': 0,
            'keep_total': 0,
            'position_correct': 0,
            'position_total': 0,
            'reveal_correct': 0,
            'reveal_total': 0,
            'rewards': [],
            'steps': 0
        }
    
    def update_decision(self, decision_type, correct):
        """Met à jour les statistiques pour une décision"""
        if decision_type == 'source':
            self.current_episode['source_total'] += 1
            if correct:
                self.current_episode['source_correct'] += 1
        elif decision_type == 'keep':
            self.current_episode['keep_total'] += 1
            if correct:
                self.current_episode['keep_correct'] += 1
        elif decision_type == 'position':
            self.current_episode['position_total'] += 1
            if correct:
                self.current_episode['position_correct'] += 1
        elif decision_type == 'reveal':
            self.current_episode['reveal_total'] += 1
            if correct:
                self.current_episode['reveal_correct'] += 1
    
    def update_reward(self, reward):
        """Ajoute une récompense à l'épisode en cours"""
        self.current_episode['rewards'].append(reward)
    
    def update_step(self):
        """Incrémente le compteur de pas de l'épisode"""
        self.current_episode['steps'] += 1
    
    def end_episode(self):
        """Termine l'épisode en cours et met à jour les métriques"""
        # Calcul des précisions
        if self.current_episode['source_total'] > 0:
            self.metrics['source_accuracy'].append(
                self.current_episode['source_correct'] / self.current_episode['source_total']
            )
        if self.current_episode['keep_total'] > 0:
            self.metrics['keep_accuracy'].append(
                self.current_episode['keep_correct'] / self.current_episode['keep_total']
            )
        if self.current_episode['position_total'] > 0:
            self.metrics['position_accuracy'].append(
                self.current_episode['position_correct'] / self.current_episode['position_total']
            )
        if self.current_episode['reveal_total'] > 0:
            self.metrics['reveal_accuracy'].append(
                self.current_episode['reveal_correct'] / self.current_episode['reveal_total']
            )
        
        # Récompense totale et longueur de l'épisode
        self.metrics['total_reward'].append(sum(self.current_episode['rewards']))
        self.metrics['episode_length'].append(self.current_episode['steps'])
        
        # Réinitialisation de l'épisode
        self.current_episode = {
            'source_correct': 0,
            'source_total': 0,
            'keep_correct': 0,
            'keep_total': 0,
            'position_correct': 0,
            'position_total': 0,
            'reveal_correct': 0,
            'reveal_total': 0,
            'rewards': [],
            'steps': 0
        }
    
    def plot_metrics(self, save_path=None):
        """Génère des graphiques pour les métriques"""
        # Vérifier si nous avons des données
        if not any(len(values) > 0 for values in self.metrics.values()):
            print("Aucune donnée à tracer")
            return

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Métriques d\'apprentissage')
        
        # Précisions
        if self.metrics['source_accuracy']:
            axes[0, 0].plot(self.metrics['source_accuracy'], label='Source')
        if self.metrics['keep_accuracy']:
            axes[0, 0].plot(self.metrics['keep_accuracy'], label='Keep')
        if self.metrics['position_accuracy']:
            axes[0, 0].plot(self.metrics['position_accuracy'], label='Position')
        if self.metrics['reveal_accuracy']:
            axes[0, 0].plot(self.metrics['reveal_accuracy'], label='Reveal')
        axes[0, 0].set_title('Précision des décisions')
        axes[0, 0].set_xlabel('Épisodes')
        axes[0, 0].set_ylabel('Précision')
        axes[0, 0].legend()
        
        # Pertes
        if self.metrics['source_loss']:
            axes[0, 1].plot(self.metrics['source_loss'], label='Source')
        if self.metrics['keep_loss']:
            axes[0, 1].plot(self.metrics['keep_loss'], label='Keep')
        if self.metrics['position_loss']:
            axes[0, 1].plot(self.metrics['position_loss'], label='Position')
        if self.metrics['reveal_loss']:
            axes[0, 1].plot(self.metrics['reveal_loss'], label='Reveal')
        axes[0, 1].set_title('Pertes')
        axes[0, 1].set_xlabel('Épisodes')
        axes[0, 1].set_ylabel('Perte')
        axes[0, 1].legend()
        
        # Récompense totale
        if self.metrics['total_reward']:
            axes[1, 0].plot(self.metrics['total_reward'])
            axes[1, 0].set_title('Récompense totale par épisode')
            axes[1, 0].set_xlabel('Épisodes')
            axes[1, 0].set_ylabel('Récompense')
        
        # Longueur des épisodes
        if self.metrics['episode_length']:
            axes[1, 1].plot(self.metrics['episode_length'])
            axes[1, 1].set_title('Longueur des épisodes')
            axes[1, 1].set_xlabel('Épisodes')
            axes[1, 1].set_ylabel('Pas')
        
        # Moyennes mobiles
        window = 100
        if len(self.metrics['total_reward']) >= window:
            axes[2, 0].plot(np.convolve(self.metrics['total_reward'], 
                                      np.ones(window)/window, mode='valid'),
                           label=f'Moyenne mobile ({window})')
            axes[2, 0].set_title('Récompense moyenne mobile')
            axes[2, 0].set_xlabel('Épisodes')
            axes[2, 0].set_ylabel('Récompense moyenne')
        
        if len(self.metrics['episode_length']) >= window:
            axes[2, 1].plot(np.convolve(self.metrics['episode_length'], 
                                      np.ones(window)/window, mode='valid'),
                           label=f'Moyenne mobile ({window})')
            axes[2, 1].set_title('Longueur moyenne des épisodes')
            axes[2, 1].set_xlabel('Épisodes')
            axes[2, 1].set_ylabel('Longueur moyenne')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def save_metrics(self, path):
        """Sauvegarde les métriques dans un fichier JSON"""
        with open(path, 'w') as f:
            json.dump(self.metrics, f)
    
    def load_metrics(self, path):
        """Charge les métriques depuis un fichier JSON"""
        with open(path, 'r') as f:
            self.metrics = json.load(f)

class Experience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class LearningAI(BaseAI):
    def __init__(self, model_path=None, memory_size=10000, batch_size=64, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialisation des réseaux
        self.source_net = SourceNet().to(self.device)
        self.keep_net = KeepNet().to(self.device)
        self.position_net = PositionNet().to(self.device)
        self.reveal_net = RevealNet().to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        # Optimiseurs
        self.source_optimizer = optim.Adam(self.source_net.parameters(), lr=0.001)
        self.keep_optimizer = optim.Adam(self.keep_net.parameters(), lr=0.001)
        self.position_optimizer = optim.Adam(self.position_net.parameters(), lr=0.001)
        self.reveal_optimizer = optim.Adam(self.reveal_net.parameters(), lr=0.001)
        
        # Paramètres d'apprentissage
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.metrics = TrainingMetrics()
        
        # Critères de perte
        self.source_criterion = nn.BCEWithLogitsLoss()
        self.keep_criterion = nn.BCEWithLogitsLoss()
        self.position_criterion = nn.CrossEntropyLoss()
        self.reveal_criterion = nn.CrossEntropyLoss()
        
        # Paramètres d'exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # État actuel
        self.current_state = None
        self.current_action = None
        self.current_reward = None
    
    def _grid_to_tensor(self, grid):
        """Convertit une grille en tenseur"""
        tensor = torch.full((GRID_SIZE,), CARD_EMPTY, device=self.device)
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] is not None:
                    if grid[i][j].revealed:
                        tensor[i * len(grid[0]) + j] = grid[i][j].value
                    else:
                        tensor[i * len(grid[0]) + j] = CARD_UNKNOWN
        return tensor
    
    def _prepare_input(self, grid, other_p_grids, card_value=None, discard_value=None):
        """Prépare les entrées pour le réseau"""
        all_grids = torch.zeros((TOTAL_GRID_SIZE,), device=self.device)
        all_grids[:GRID_SIZE] = self._grid_to_tensor(grid)
        
        for i, other_grid in enumerate(other_p_grids):
            start_idx = (i + 1) * GRID_SIZE
            all_grids[start_idx:start_idx + GRID_SIZE] = self._grid_to_tensor(other_grid)
        
        card_info = torch.tensor([
            card_value if card_value is not None else CARD_UNKNOWN,
            discard_value if discard_value is not None else CARD_UNKNOWN
        ], device=self.device).float()
        
        return [all_grids, card_info]
    
    def should_explore(self):
        """Décide si on doit explorer (aléatoire) ou exploiter (réseau)"""
        return random.random() < self.epsilon
    
    def choose_source(self, grid, discard, other_p_grids):
        if self.should_explore():
            action = random.choice(['D', 'P'])
            self.update_metrics('source', True, None)
            return action
        input_tensor = self._prepare_input(
            grid=grid,
            other_p_grids=other_p_grids,
            discard_value=discard[-1].value if discard else None
        )
        with torch.no_grad():
            output = self.source_net(input_tensor)
            probability = torch.sigmoid(output).item()
        action = 'D' if probability > 0.5 else 'P'
        self.update_metrics('source', True, None)
        return action
    
    def choose_keep(self, card, grid, other_p_grids):
        if self.should_explore():
            action = random.choice([True, False])
            self.update_metrics('keep', True, None)
            return action
        input_tensor = self._prepare_input(
            grid=grid,
            other_p_grids=other_p_grids,
            card_value=card.value
        )
        with torch.no_grad():
            output = self.keep_net(input_tensor)
            probability = torch.sigmoid(output).item()
        action = probability > 0.5
        self.update_metrics('keep', True, None)
        return action
    
    def choose_position(self, card, grid, other_p_grids):
        if self.should_explore():
            legal_positions = []
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if grid[i][j] is not None:
                        legal_positions.append((i, j))
            action = random.choice(legal_positions)
            self.update_metrics('position', True, None)
            return action
        input_tensor = self._prepare_input(
            grid=grid,
            other_p_grids=other_p_grids,
            card_value=card.value
        )
        with torch.no_grad():
            position_scores = self.position_net(input_tensor, grid=grid)
            probabilities = torch.softmax(position_scores, dim=0)
            best_position_idx = torch.argmax(probabilities).item()
        i = best_position_idx // len(grid[0])
        j = best_position_idx % len(grid[0])
        action = (i, j)
        self.update_metrics('position', True, None)
        return action
    
    def choose_reveal(self, grid):
        if self.should_explore():
            unrevealed_positions = []
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if grid[i][j] is not None and not grid[i][j].revealed:
                        unrevealed_positions.append((i, j))
            action = random.choice(unrevealed_positions)
            self.update_metrics('reveal', True, None)
            return action
        input_tensor = self._prepare_input(grid=grid, other_p_grids=[])
        with torch.no_grad():
            position_scores = self.reveal_net(input_tensor, grid=grid)
            probabilities = torch.softmax(position_scores, dim=0)
            best_position_idx = torch.argmax(probabilities).item()
        i = best_position_idx // len(grid[0])
        j = best_position_idx % len(grid[0])
        action = (i, j)
        self.update_metrics('reveal', True, None)
        return action
    
    def initial_flip(self):
        """Choisit deux positions pour révéler les cartes initiales"""
        # Pour l'instant, on choisit aléatoirement deux positions
        # On pourrait utiliser le réseau reveal_net pour faire un choix plus intelligent
        positions = []
        for _ in range(2):
            i = random.randint(0, 2)  # 3 lignes
            j = random.randint(0, 3)  # 4 colonnes
            positions.append((i, j))
        return positions
    
    def add_experience(self, state, action, reward, next_state, done):
        """Ajoute une expérience à la mémoire"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def train_step(self):
        """Effectue une étape d'apprentissage"""
        if len(self.memory) < self.batch_size:
            return
        
        # Séparer les expériences par type d'action
        source_experiences = [exp for exp in self.memory if exp.action in ('D', 'P')]
        keep_experiences = [exp for exp in self.memory if isinstance(exp.action, bool)]
        position_experiences = [exp for exp in self.memory if isinstance(exp.action, tuple)]
        
        # Entraîner le réseau source
        if len(source_experiences) >= self.batch_size:
            batch = random.sample(source_experiences, self.batch_size)
            states = torch.stack([exp.state[0] for exp in batch]).to(self.device)
            card_infos = torch.stack([exp.state[1] for exp in batch]).to(self.device)
            actions = [exp.action for exp in batch]
            rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32).to(self.device)
            next_states = torch.stack([exp.next_state[0] for exp in batch]).to(self.device)
            next_card_infos = torch.stack([exp.next_state[1] for exp in batch]).to(self.device)
            dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32).to(self.device)
            self._update_source_net(states, card_infos, actions, rewards, next_states, next_card_infos, dones)
        
        # Entraîner le réseau keep
        if len(keep_experiences) >= self.batch_size:
            batch = random.sample(keep_experiences, self.batch_size)
            states = torch.stack([exp.state[0] for exp in batch]).to(self.device)
            card_infos = torch.stack([exp.state[1] for exp in batch]).to(self.device)
            actions = [exp.action for exp in batch]
            rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32).to(self.device)
            next_states = torch.stack([exp.next_state[0] for exp in batch]).to(self.device)
            next_card_infos = torch.stack([exp.next_state[1] for exp in batch]).to(self.device)
            dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32).to(self.device)
            self._update_keep_net(states, card_infos, actions, rewards, next_states, next_card_infos, dones)
        
        # Entraîner les réseaux position et reveal
        if len(position_experiences) >= self.batch_size:
            batch = random.sample(position_experiences, self.batch_size)
            states = torch.stack([exp.state[0] for exp in batch]).to(self.device)
            card_infos = torch.stack([exp.state[1] for exp in batch]).to(self.device)
            actions = [exp.action for exp in batch]
            rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32).to(self.device)
            next_states = torch.stack([exp.next_state[0] for exp in batch]).to(self.device)
            next_card_infos = torch.stack([exp.next_state[1] for exp in batch]).to(self.device)
            dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32).to(self.device)
            self._update_position_net(states, card_infos, actions, rewards, next_states, next_card_infos, dones)
            self._update_reveal_net(states, actions, rewards, next_states, dones)
        
        # Mise à jour d'epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _update_source_net(self, states, card_infos, actions, rewards, next_states, next_card_infos, dones):
        """Met à jour le réseau source pour apprendre à choisir entre pioche et défausse"""
        # Convertir les actions en tenseur binaire (0 = pioche, 1 = défausse)
        actions = torch.tensor([1 if a == 'D' else 0 for a in actions], dtype=torch.float32, device=self.device)
        
        # Calculer les prédictions actuelles
        current_q = self.source_net([states, card_infos])
        
        # Calculer la perte directement entre les prédictions et les actions réelles
        loss = self.source_criterion(current_q, actions.unsqueeze(1))
        
        # Mise à jour du réseau
        self.source_optimizer.zero_grad()
        loss.backward()
        self.source_optimizer.step()
        
        # Mise à jour des métriques
        self.metrics.metrics['source_loss'].append(loss.item())
    
    def _update_keep_net(self, states, card_infos, actions, rewards, next_states, next_card_infos, dones):
        """Met à jour le réseau keep pour apprendre à décider si on garde une carte"""
        # Convertir les actions en tenseur binaire (0 = ne pas garder, 1 = garder)
        actions = torch.tensor([1 if a else 0 for a in actions], dtype=torch.float32, device=self.device)
        
        # Calculer les prédictions actuelles
        current_q = self.keep_net([states, card_infos])
        
        # Calculer la perte directement entre les prédictions et les actions réelles
        loss = self.keep_criterion(current_q, actions.unsqueeze(1))
        
        # Mise à jour du réseau
        self.keep_optimizer.zero_grad()
        loss.backward()
        self.keep_optimizer.step()
        
        # Mise à jour des métriques
        self.metrics.metrics['keep_loss'].append(loss.item())
    
    def _update_position_net(self, states, card_infos, actions, rewards, next_states, next_card_infos, dones):
        """Met à jour le réseau position pour apprendre à choisir la meilleure position"""
        # Convertir les actions (i,j) en indices linéaires
        actions = torch.tensor([i * GRID_COLS + j for i, j in actions], dtype=torch.long, device=self.device)
        
        # Calculer les prédictions actuelles
        current_q = self.position_net([states, card_infos])
        
        # Calculer la perte en utilisant CrossEntropyLoss
        # Cette perte va apprendre à maximiser la probabilité de la position choisie
        loss = self.position_criterion(current_q, actions)
        
        # Mise à jour du réseau
        self.position_optimizer.zero_grad()
        loss.backward()
        self.position_optimizer.step()
        
        # Mise à jour des métriques
        self.metrics.metrics['position_loss'].append(loss.item())
    
    def _update_reveal_net(self, states, actions, rewards, next_states, dones):
        """Met à jour le réseau reveal pour apprendre à choisir la meilleure carte à révéler"""
        # Convertir les actions (i,j) en indices linéaires
        actions = torch.tensor([i * GRID_COLS + j for i, j in actions], dtype=torch.long, device=self.device)
        
        # Calculer les prédictions actuelles
        current_q = self.reveal_net([states])
        
        # Calculer la perte en utilisant CrossEntropyLoss
        # Cette perte va apprendre à maximiser la probabilité de la carte choisie
        loss = self.reveal_criterion(current_q, actions)
        
        # Mise à jour du réseau
        self.reveal_optimizer.zero_grad()
        loss.backward()
        self.reveal_optimizer.step()
        
        # Mise à jour des métriques
        self.metrics.metrics['reveal_loss'].append(loss.item())
    
    def save_model(self, path):
        """Sauvegarde le modèle"""
        torch.save({
            'source_net': self.source_net.state_dict(),
            'keep_net': self.keep_net.state_dict(),
            'position_net': self.position_net.state_dict(),
            'reveal_net': self.reveal_net.state_dict()
        }, path)
    
    def load_model(self, path):
        """Charge le modèle"""
        checkpoint = torch.load(path)
        self.source_net.load_state_dict(checkpoint['source_net'])
        self.keep_net.load_state_dict(checkpoint['keep_net'])
        self.position_net.load_state_dict(checkpoint['position_net'])
        self.reveal_net.load_state_dict(checkpoint['reveal_net'])

    def update_metrics(self, decision_type, correct, reward=None):
        """Met à jour les métriques d'apprentissage"""
        self.metrics.update_decision(decision_type, correct)
        if reward is not None:
            self.metrics.update_reward(reward)
        self.metrics.update_step()

    def end_episode(self):
        """Termine l'épisode en cours et met à jour les métriques"""
        self.metrics.end_episode()

    def save_training_state(self, model_path, metrics_path, plot_path):
        """Sauvegarde l'état complet de l'apprentissage"""
        self.save_model(model_path)
        self.metrics.save_metrics(metrics_path)
        self.metrics.plot_metrics(plot_path)
    
    def load_training_state(self, model_path, metrics_path):
        """Charge l'état complet de l'apprentissage"""
        self.load_model(model_path)
        self.metrics.load_metrics(metrics_path)
