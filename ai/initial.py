import random
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class InitialAI(BaseAI):
    
    def initial_flip(self):
        return [[random.randrange(GRID_ROWS), random.randrange(GRID_COLS)] for _ in range(2)]
    
    def choose_action(self, grid, discard):
        source = 'D' if discard and discard.value < 5 else 'P'
        if source == 'P':
            take_draw = random.choice([True, False])
            if take_draw : source = None
        unrevealed = [(i, j) for i in range(GRID_ROWS) for j in range(GRID_COLS) if not grid[i][j].revealed]
        position = random.choice(unrevealed) if unrevealed else None
        return {'source': source, 'position': position}
    
    # Choix de la source de la carte (pioche ou discard)
    def choose_source(self, grid, discard = None):
        return 'D' if discard and discard.value < 5 else 'P'
    
    # Choix de garde ou non la carte piochée
    def choose_keep(self, grid, card, discard = None):
        return card.value < 5
    
    # En cas de garde, choix de la position de la carte piochée
    def choose_position(self, card, keep, grid, discard = None, deck = None):
        
        revealed_positions = [(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j].revealed]
        if revealed_positions:
            max_diff_pos = max(revealed_positions, key=lambda pos: abs(grid[pos[0]][pos[1]].value - card.value))
            max_diff = abs(grid[max_diff_pos[0]][max_diff_pos[1]].value - card.value)
            if max_diff >= 4:
                return max_diff_pos

        # Pour chaque colonne, on vérifie si la carte piochée peut former une colonne avec une carte révélée existante
        # Dans ce cas, on choisit la carte révélée avec la valeur la plus élevée
        for col in range(len(grid[0])):
            column_values = [grid[row][col].value for row in range(len(grid)) if grid[row][col].revealed]
            if card.value in column_values:
                for row in range(len(grid)):
                    if grid[row][col].revealed and grid[row][col].value != card.value:
                        return max(
                            [(r, col) for r in range(len(grid)) if grid[r][col].revealed and grid[r][col].value != card.value],
                            key=lambda pos: grid[pos[0]][pos[1]].value
                        )

        unrevealed_positions = [(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if not grid[i][j].revealed]
        return random.choice(unrevealed_positions) if unrevealed_positions else None
    
    # En cas de non garde, choix de la carte existante à révéler
    def choose_reveal(self, grid):
        
        # On choisit la colonne avec le plus de cartes révélées
        # Si plusieurs colonnes ont le même nombre de cartes révélées, on choisit une colonne au hasard parmi celles-ci
        column_revealed_counts = [
            (col, sum(1 for row in range(len(grid)) if grid[row][col].revealed))
            for col in range(len(grid[0]))
        ]
        max_revealed = max(column_revealed_counts, key=lambda x: x[1])[1]
        candidate_columns = [col for col, count in column_revealed_counts if count == max_revealed]

        unrevealed_positions = [
            (i, j) for j in candidate_columns for i in range(len(grid)) if not grid[i][j].revealed
        ]
        return random.choice(unrevealed_positions) if unrevealed_positions else None