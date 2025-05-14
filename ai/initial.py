import random
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class InitialAI(BaseAI):
    
    def initial_flip(self):
        return [[random.randrange(GRID_ROWS), random.randrange(GRID_COLS)] for _ in range(2)]
    
    def is_it_last_card(self, grid):
        unrevealed_count = sum(1 for row in grid for card in row if not card.revealed)
        if unrevealed_count <= 1:
            print("Il ne reste qu'une carte à révéler.")
        return unrevealed_count <= 1

    def is_it_good_to_take_the_last_card(self, card_value, grid, other_p_grids):
        our_grid_sum = sum(card.value for row in grid for card in row if card.revealed)
        other_players_scores = [
            sum(
                other_card.value
                for other_row in other_grid
                for other_card in other_row
                if other_card.revealed
            )
            + 2 * sum(
                1
                for other_row in other_grid
                for other_card in other_row
                if not other_card.revealed
            )
            for other_grid in other_p_grids
        ]
        
        our_sum = our_grid_sum + card_value
        print(f"Notre score : {our_grid_sum} + {card_value} - Autres joueurs : {other_players_scores}")
        print(f"Is our sum <= aux autres : {our_sum <= min(other_players_scores)}")
        return our_sum <= min(other_players_scores)
    
    # Choix de la source de la carte (pioche ou discard)
    def choose_source(self, grid, discard, other_p_grids):
        if self.is_it_last_card(grid):
            is_discard_good = self.is_it_good_to_take_the_last_card(discard[-1].value, grid, other_p_grids)
            return 'D' if discard and discard[-1].value < 5 and is_discard_good else 'P'
        return 'D' if discard and discard[-1].value < 5 else 'P'
    
    # Choix de garde ou non la carte piochée
    def choose_keep(self, card, grid, other_p_grids):
        # Si il ne nous reste qu'une carte à révéler
        if self.is_it_last_card(grid):
            # Si on a plus qu'une carte, on vérifie que la somme de notre grille + une dummy card (moyenne des cartes restantes) 
            # est inférieure à la somme des autres joueurs et que la carte piochée est plus haute que cette moyenne
            # alors on ne la garde pas, sinon, on garde la carte
            is_keep_good = self.is_it_good_to_take_the_last_card(6, grid, other_p_grids)
            
            # Si on a tellement d'avantage qu'on peut se permettre de ne pas garder la carte
            # en supposant que notre carte non révélée est <= à la moyenne des cartes restantes
            return not is_keep_good
        else :
            # Si ce n'est pas le cas, on garde la carte piochée si elle est plus faible que 5
            return card.value < 5

    def is_all_unrevealed(self, grid):
        return all(not card.revealed for row in grid for card in row)

    # En cas de garde, choix de la position de la carte piochée
    def choose_position(self, card, grid, other_p_grids):
        
        revealed_positions = [(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j].revealed]
        
        # Si on a plus qu'une carte à révéler
        if self.is_it_last_card(grid):
            # Si elle ne nous permet pas de gagner on choisit la carte révélée la plus haute
            if not self.is_it_good_to_take_the_last_card(card.value, grid, other_p_grids):
                print(max(revealed_positions, key=lambda pos: grid[pos[0]][pos[1]].value - card.value))
                return max(revealed_positions, key=lambda pos: grid[pos[0]][pos[1]].value - card.value)
            else:
                last_unrevealed = [(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if not grid[i][j].revealed]
                return last_unrevealed[0]

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
        if not self.is_all_unrevealed(grid):
            max_diff_pos = max(revealed_positions, key=lambda pos: grid[pos[0]][pos[1]].value - card.value)
            max_diff = grid[max_diff_pos[0]][max_diff_pos[1]].value - card.value
            if max_diff >= 4 : return max_diff_pos

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