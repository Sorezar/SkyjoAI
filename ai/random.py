import random
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class RandomAI(BaseAI):
    
    def initial_flip(self):
        return [[random.randrange(GRID_ROWS), random.randrange(GRID_COLS)] for _ in range(2)]
    
    # Choix de la source de la carte (pioche ou discard)
    def choose_source(self, grid, discard = None):
        return random.choice(['P', 'D'])
    
    # Choix de garde ou non la carte piochée
    def choose_keep(self, card, grid):
        return random.choice([True, False])
    
    # En cas de garde, choix de la position de la carte piochée
    def choose_position(self, card, grid):
        return random.choice([(i, j) for i in range(len(grid)) for j in range(len(grid[0]))])
    
    # En cas de non garde, choix de la carte existante à révéler
    def choose_reveal(self, grid):
        return random.choice([(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if not grid[i][j].revealed])