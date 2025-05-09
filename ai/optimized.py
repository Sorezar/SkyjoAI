import random
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class OptimizedAI(BaseAI):
    
    def initial_flip(self):
        return [[random.randrange(GRID_ROWS), random.randrange(GRID_COLS)] for _ in range(2)]
    
    def choose_action(self, grid, discard, deck):
        # Check if the top discard card can form a column with an existing revealed card
        if discard is not None:
            for col in range(GRID_COLS):
                revealed_values = [grid[row][col].value for row in range(GRID_ROWS) if col < len(grid[row]) and grid[row][col].revealed]
                if discard.value in revealed_values:
                    for row in range(GRID_ROWS):
                        if col < len(grid[row]) and grid[row][col].value != discard:
                            return {'source': 'D', 'position': (row, col)}

        # Default behavior if no column can be formed
        source = 'D' if discard is not None and discard.value < 5 else 'P'
        unrevealed = [(i, j) for i in range(len(grid)) for j in range(len(grid[i])) if not grid[i][j].revealed]
        position = random.choice(unrevealed) if unrevealed else None
        
        return {'source': source, 'position': position}