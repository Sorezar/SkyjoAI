import random
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class InitialAI(BaseAI):
    
    def initial_flip(self):
        return [[random.randrange(GRID_ROWS), random.randrange(GRID_COLS)] for _ in range(2)]
    
    # Choix de la source de la carte (pioche ou discard)
    def choose_source(self, grid, discard):
        return 'D' if discard and discard[-1].value < 5 else 'P'
    
    # Choix de garde ou non la carte piochée
    def choose_keep(self, card, grid):
        return card.value < 5
    
    # En cas de garde, choix de la position de la carte piochée
    def choose_position(self, card, grid):
        
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
        # Sinon, on remplace la carte révélée avec la valeur la plus élevée dans la grille sauf si la différence est inférieure à 4
        revealed_positions = [(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j].revealed]
        if revealed_positions:
            max_diff_pos = max(revealed_positions, key=lambda pos: grid[pos[0]][pos[1]].value - card.value)
            max_diff = grid[max_diff_pos[0]][max_diff_pos[1]].value - card.value
            if max_diff >= 4:
                return max_diff_pos

        # Sinon, on choisit une position au hasard parmi les positions non révélées
        unrevealed_positions = [(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if not grid[i][j].revealed]
        return random.choice(unrevealed_positions) if unrevealed_positions else None
    
    # En cas de non garde, choix de la carte existante à révéler
    def choose_reveal(self, grid):
        # Compte le nombre de cartes révélées pour chaque colonne
        column_revealed_counts = []
        for col in range(len(grid[0])):
            revealed_count = sum(1 for row in range(len(grid)) if grid[row][col].revealed)
            column_revealed_counts.append((col, revealed_count))

        # Filtre les colonnes qui ne sont pas entièrement révélées
        partially_revealed_columns = [
            col for col, count in column_revealed_counts
            if any(not grid[row][col].revealed for row in range(len(grid)))
        ]

        # Trouve le nombre maximum de cartes révélées parmi les colonnes partiellement révélées
        max_revealed = max(
            (count for col, count in column_revealed_counts if col in partially_revealed_columns),
            default=0
        )

        # Identifie les colonnes candidates ayant le maximum de cartes révélées
        candidate_columns = [
            col for col, count in column_revealed_counts
            if count == max_revealed and col in partially_revealed_columns
        ]

        # Cherche une position non révélée dans les colonnes candidates
        unrevealed_positions = [
            (i, j) for j in candidate_columns for i in range(len(grid)) if not grid[i][j].revealed
        ]
        return random.choice(unrevealed_positions)