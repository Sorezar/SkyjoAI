import random
from ai.base import BaseAI
import config.config as cfg

class OptimizedAI(BaseAI):
    
    def choose_action(self, player, top_discard, deck):
        # Check if the top discard card can form a column with an existing revealed card
        if top_discard is not None:
            for col in range(cfg.GRID_COLS):
                revealed_values = [player.grid[row][col].value for row in range(cfg.GRID_ROWS) if player.grid[row][col].revealed]
                if top_discard in revealed_values:
                    for row in range(cfg.GRID_ROWS):
                        # Allow replacing an already revealed card
                        if player.grid[row][col].value != top_discard:
                            return {'source': 'D', 'position': (row, col)}

        # Default behavior if no column can be formed
        source = 'D' if top_discard is not None and top_discard < 5 else 'P'
        unrevealed = [(i, j) for i in range(cfg.GRID_ROWS) for j in range(cfg.GRID_COLS) if not player.grid[i][j].revealed]
        position = random.choice(unrevealed) if unrevealed else None
        
        return {'source': source, 'position': position}