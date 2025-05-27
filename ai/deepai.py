import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class DQN(nn.Module):
    """Réseau de neurones pour approximer la fonction Q"""
    def __init__(self, input_size, hidden_size=256, output_size=1):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

class DeepAI(BaseAI):
    def __init__(self, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        # Hyperparamètres
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Mémoire de replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Réseaux pour chaque type de décision
        # Taille d'entrée : grille joueur (12*2) + défausse (1) + grilles adversaires (12*2*3) + scores (4)
        self.state_size = GRID_ROWS * GRID_COLS * 2 + 1 + GRID_ROWS * GRID_COLS * 2 * 3 + 4
        
        # Réseau pour choisir source (pioche ou défausse)
        self.source_net = DQN(self.state_size, output_size=2)
        self.source_optimizer = optim.Adam(self.source_net.parameters(), lr=self.lr)
        
        # Réseau pour choisir de garder ou non
        self.keep_net = DQN(self.state_size + 1, output_size=2)  # +1 pour la carte piochée
        self.keep_optimizer = optim.Adam(self.keep_net.parameters(), lr=self.lr)
        
        # Réseau pour choisir la position
        self.position_net = DQN(self.state_size + 1, output_size=GRID_ROWS * GRID_COLS)
        self.position_optimizer = optim.Adam(self.position_net.parameters(), lr=self.lr)
        
        # Réseau pour choisir quelle carte révéler
        self.reveal_net = DQN(self.state_size, output_size=GRID_ROWS * GRID_COLS)
        self.reveal_optimizer = optim.Adam(self.reveal_net.parameters(), lr=self.lr)
        
        self.training = True
        
    def grid_to_features(self, grid):
        """Convertit une grille en features pour le réseau"""
        values = []
        revealed = []
        
        # Parcourir la grille actuelle
        for row in grid:
            for card in row:
                values.append(card.value if card.revealed else 0)
                revealed.append(1 if card.revealed else 0)
        
        # Padding pour atteindre la taille maximale (GRID_ROWS * GRID_COLS)
        current_size = len(grid) * len(grid[0]) if grid else 0
        max_size = GRID_ROWS * GRID_COLS
        padding_size = max_size - current_size
        
        # Ajouter du padding avec des valeurs neutres (-99 pour indiquer l'absence)
        values.extend([-99] * padding_size)
        revealed.extend([0] * padding_size)
        
        return values + revealed
    
    def get_state(self, grid, discard, other_p_grids):
        """Construit l'état complet du jeu"""
        state = []
        
        # Grille du joueur
        state.extend(self.grid_to_features(grid))
        
        # Carte de défausse
        state.append(discard[-1].value if discard else 0)
        
        # Grilles des adversaires (padding si moins de 3 adversaires)
        for i in range(3):
            if i < len(other_p_grids):
                state.extend(self.grid_to_features(other_p_grids[i]))
            else:
                state.extend([0] * (GRID_ROWS * GRID_COLS * 2))
        
        # Scores actuels (simplifié - on met 0 pour l'instant)
        state.extend([0, 0, 0, 0])
        
        return np.array(state, dtype=np.float32)
    
    def initial_flip(self):
        """Retourne 2 positions aléatoires pour les cartes initiales"""
        positions = [(i, j) for i in range(GRID_ROWS) for j in range(GRID_COLS)]
        selected = random.sample(positions, 2)
        return [[pos[0], pos[1]] for pos in selected]
    
    def choose_source(self, grid, discard=None, other_p_grids=None):
        """Choisit entre piocher ou prendre la défausse"""
        if not discard:
            return 'P'
            
        state = self.get_state(grid, discard, other_p_grids or [])
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Epsilon-greedy
        if self.training and random.random() < self.epsilon:
            return random.choice(['P', 'D'])
        
        with torch.no_grad():
            q_values = self.source_net(state_tensor)
            action = q_values.argmax().item()
        
        return 'P' if action == 0 else 'D'
    
    def choose_keep(self, card, grid, other_p_grids=None):
        """Décide de garder ou non la carte piochée"""
        state = self.get_state(grid, [card], other_p_grids or [])
        # Ajouter la valeur de la carte piochée
        state = np.append(state, card.value)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Epsilon-greedy
        if self.training and random.random() < self.epsilon:
            return random.choice([True, False])
        
        with torch.no_grad():
            q_values = self.keep_net(state_tensor)
            action = q_values.argmax().item()
        
        return action == 1
    
    def choose_position(self, card, grid, other_p_grids=None):
        """Choisit où placer la carte"""
        state = self.get_state(grid, [card], other_p_grids or [])
        state = np.append(state, card.value)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Obtenir toutes les positions valides
        valid_positions = [(i, j) for i in range(len(grid)) for j in range(len(grid[0]))]
        
        if self.training and random.random() < self.epsilon:
            return random.choice(valid_positions)
        
        with torch.no_grad():
            q_values = self.position_net(state_tensor).squeeze()
            
            # Créer un mapping des positions valides vers les indices du réseau
            position_to_idx = {}
            idx_to_position = {}
            
            for i, j in valid_positions:
                # Utiliser l'index basé sur la grille complète originale
                idx = i * GRID_COLS + j
                position_to_idx[(i, j)] = idx
                idx_to_position[idx] = (i, j)
            
            # Masquer les positions invalides
            masked_q_values = torch.full_like(q_values, float('-inf'))
            for pos, idx in position_to_idx.items():
                if idx < len(q_values):
                    masked_q_values[idx] = q_values[idx]
            
            # Trouver le meilleur index parmi les positions valides
            valid_indices = list(position_to_idx.values())
            valid_q_values = [masked_q_values[idx] for idx in valid_indices if idx < len(masked_q_values)]
            
            if valid_q_values:
                best_valid_idx = valid_indices[np.argmax(valid_q_values)]
                # Retrouver la position correspondante
                for pos, idx in position_to_idx.items():
                    if idx == best_valid_idx:
                        return pos
            
            # Sécurité : retourner une position aléatoire si problème
            return random.choice(valid_positions)
    
    def choose_reveal(self, grid):
        """Choisit quelle carte révéler"""
        # Trouver les cartes non révélées
        unrevealed = [(i, j) for i in range(len(grid)) 
                      for j in range(len(grid[0])) 
                      if not grid[i][j].revealed]
        
        if not unrevealed:
            return (0, 0)  # Sécurité
        
        if self.training and random.random() < self.epsilon:
            return random.choice(unrevealed)
        
        state = self.get_state(grid, [], [])
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.reveal_net(state_tensor).squeeze()
            
            # Créer un mapping des positions valides vers les indices du réseau
            position_to_idx = {}
            
            for i, j in unrevealed:
                # Utiliser l'index basé sur la grille complète originale
                idx = i * GRID_COLS + j
                position_to_idx[(i, j)] = idx
            
            # Masquer les positions déjà révélées
            masked_q_values = torch.full_like(q_values, float('-inf'))
            for pos, idx in position_to_idx.items():
                if idx < len(q_values):
                    masked_q_values[idx] = q_values[idx]
            
            # Trouver le meilleur index parmi les positions valides
            valid_indices = list(position_to_idx.values())
            valid_q_values = [masked_q_values[idx] for idx in valid_indices if idx < len(masked_q_values)]
            
            if valid_q_values:
                best_valid_idx = valid_indices[np.argmax(valid_q_values)]
                # Retrouver la position correspondante
                for pos, idx in position_to_idx.items():
                    if idx == best_valid_idx:
                        return pos
            
            # Sécurité : retourner une position aléatoire si problème
            return random.choice(unrevealed)
    
    def remember(self, state, action, reward, next_state, done, action_type):
        """Stocke l'expérience dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done, action_type))
    
    def replay(self):
        """Entraîne les réseaux sur un batch d'expériences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        # Séparer par type d'action
        source_batch = [(s, a, r, ns, d) for s, a, r, ns, d, t in batch if t == 'source']
        keep_batch = [(s, a, r, ns, d) for s, a, r, ns, d, t in batch if t == 'keep']
        position_batch = [(s, a, r, ns, d) for s, a, r, ns, d, t in batch if t == 'position']
        reveal_batch = [(s, a, r, ns, d) for s, a, r, ns, d, t in batch if t == 'reveal']
        
        # Entraîner chaque réseau
        self._train_network(source_batch, self.source_net, self.source_optimizer, 2)
        self._train_network(keep_batch, self.keep_net, self.keep_optimizer, 2)
        self._train_network(position_batch, self.position_net, self.position_optimizer, GRID_ROWS * GRID_COLS)
        self._train_network(reveal_batch, self.reveal_net, self.reveal_optimizer, GRID_ROWS * GRID_COLS)
    
    def _train_network(self, batch, network, optimizer, output_size):
        """Entraîne un réseau spécifique"""
        if not batch:
            return
            
        states = torch.FloatTensor([s for s, _, _, _, _ in batch])
        actions = torch.LongTensor([a for _, a, _, _, _ in batch])
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch])
        next_states = torch.FloatTensor([ns for _, _, _, ns, _ in batch])
        dones = torch.FloatTensor([d for _, _, _, _, d in batch])
        
        current_q_values = network(states).gather(1, actions.unsqueeze(1))
        next_q_values = network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def update_epsilon(self):
        """Décroît epsilon pour l'exploration"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def set_training(self, training):
        """Active/désactive le mode entraînement"""
        self.training = training
        if training:
            self.source_net.train()
            self.keep_net.train()
            self.position_net.train()
            self.reveal_net.train()
        else:
            self.source_net.eval()
            self.keep_net.eval()
            self.position_net.eval()
            self.reveal_net.eval()
    
    def load_model(self, path):
        """Charge un modèle pré-entraîné"""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.source_net.load_state_dict(checkpoint['source_net'])
        self.keep_net.load_state_dict(checkpoint['keep_net'])
        self.position_net.load_state_dict(checkpoint['position_net'])
        self.reveal_net.load_state_dict(checkpoint['reveal_net'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.set_training(False)  # Mode évaluation par défaut
        print(f"Modèle chargé depuis {path}")
