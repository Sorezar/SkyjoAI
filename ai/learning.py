import numpy as np
import random
from ai.base import BaseAI
import torch
import torch.nn as nn
import torch.optim as optim

GRID_ROWS = 3
GRID_COLS = 4

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class LearningAI(BaseAI):
    def __init__(self, model_path=None, epsilon=0.1):
        #self.source_model   = PolicyNetwork(48 + 1, 2)  # grille + valeur carte défausse
        self.source_model   = PolicyNetwork(12 + 1, 2)
        self.keep_model     = PolicyNetwork(12 + 1, 2)  # grille + valeur carte piochée
        self.reveal_model   = PolicyNetwork(12, 12)     # grille
        self.position_model = PolicyNetwork(12 + 1, 12) # grille + valeur carte

        self.optimizers = [
            optim.Adam(self.source_model.parameters(), lr=0.001),
            optim.Adam(self.keep_model.parameters(), lr=0.001),
            optim.Adam(self.reveal_model.parameters(), lr=0.001),
            optim.Adam(self.position_model.parameters(), lr=0.001)
        ]
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.epsilon = epsilon  # exploration
        self.memory = []  # (state, action, reward, model_id)

    def initial_flip(self):
        return [[random.randrange(GRID_ROWS), random.randrange(GRID_COLS)] for _ in range(2)]

    #def flatten_grid(self, grid):
    #    return [cell.value if cell and cell.revealed else -99 for row in grid for cell in row]

    def flatten_grid(self, grid):
        result = []
        for row in grid:
            for cell in row:
                if cell is None:
                    result.append(-99)  # case vide
                elif cell.revealed:
                    result.append(cell.value)
                else:
                    result.append(-99)  # case non révélée
        return result  # Toujours 12 valeurs

    def choose_source(self, grid, discard, other_p_grids=None):
        state = torch.tensor(self.flatten_grid(grid) + [discard[-1].value], dtype=torch.float32)
        probs = self.source_model(state)
        if random.random() < self.epsilon:
            action = random.randint(0, 1)
        else:
            action = torch.argmax(probs).item()
        self.memory.append((state, action, None, 0))
        return 'D' if action == 0 else 'P'

    def choose_keep(self, card, grid, other_p_grids=None):
        state = torch.tensor(self.flatten_grid(grid) + [card.value], dtype=torch.float32)
        probs = self.keep_model(state)
        if random.random() < self.epsilon:
            action = random.randint(0, 1)
        else:
            action = torch.argmax(probs).item()
        self.memory.append((state, action, None, 1))
        return action == 1

    def choose_reveal(self, grid):
        state = torch.tensor(self.flatten_grid(grid), dtype=torch.float32)
        probs = self.reveal_model(state)
        if random.random() < self.epsilon:
            pos = random.randint(0, 11)
        else:
            pos = torch.argmax(probs).item()
        self.memory.append((state, pos, None, 2))
        return divmod(pos, GRID_COLS)

    def choose_position(self, card, grid, other_p_grids=None):
        state = torch.tensor(self.flatten_grid(grid) + [card.value], dtype=torch.float32)
        probs = self.position_model(state)
        if random.random() < self.epsilon:
            pos = random.randint(0, 11)
        else:
            pos = torch.argmax(probs).item()
        self.memory.append((state, pos, None, 3))
        return divmod(pos, GRID_COLS)

    def train(self, reward):
        for state, action, _, model_id in self.memory:
            model = [self.source_model, self.keep_model, self.reveal_model, self.position_model][model_id]
            optimizer = self.optimizers[model_id]
            target = torch.tensor([action], dtype=torch.long)
            pred = model(state.unsqueeze(0))
            loss = self.loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.memory.clear()
