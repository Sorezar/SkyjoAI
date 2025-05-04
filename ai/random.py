import random
from ai.base import BaseAI
import config.config as cfg

class RandomAI(BaseAI):
    
    def choose_action(self, player, top_discard, deck):
        source = 'D' if top_discard is not None and top_discard < 5 else 'P'
        unrevealed = [(i,j) for i in range(cfg.GRID_ROWS) for j in range(cfg.GRID_COLS) if not player.grid[i][j].revealed]
        position = random.choice(unrevealed) if unrevealed else None
        
        return {'source': source, 'position': position}