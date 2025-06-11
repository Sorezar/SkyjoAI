from config.config import GRID_ROWS, GRID_COLS

def create_safe_grid(grid):
    """Crée une copie sécurisée de la grille où seules les cartes révélées sont visibles"""
    safe_grid = []
    for row in grid:
        safe_row = []
        for card in row:
            if card is None:
                safe_row.append(None)
                continue
                
            # Vérification simplifiée : doit être un objet Card valide
            if not (hasattr(card, 'revealed') and hasattr(card, 'value')):
                # Erreur de programmation - ne devrait jamais arriver après correction
                raise TypeError(f"create_safe_grid attend des objets Card, reçu: {type(card)} = {card}")
            
            if card.revealed:
                # Créer une copie de la carte révélée
                safe_card = type('SafeCard', (), {
                    'value': card.value,
                    'revealed': True
                })()
                safe_row.append(safe_card)
            else:
                # Carte masquée - les IA ne voient que le fait qu'elle n'est pas révélée
                safe_card = type('SafeCard', (), {
                    'value': None,  # Valeur cachée
                    'revealed': False
                })()
                safe_row.append(safe_card)
                
        safe_grid.append(safe_row)
    return safe_grid

class Player:
    def __init__(self, id, name, ai):
        self.id    = id
        self.name  = name
        self.ai    = ai
        self.grid  = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    def all_revealed(self):
        return all(c.revealed for row in self.grid for c in row)

    def reveal_all(self):
        for row in self.grid:
            for c in row:
                c.revealed = True

    def round_score(self):
        return sum(c.value for row in self.grid for c in row)

    def take_turn(self, deck, discard, other_p_grids):
        # Créer des grilles sécurisées pour les autres joueurs (sans cartes non révélées)
        safe_other_grids = [create_safe_grid(grid) for grid in other_p_grids]
        
        # Step 1 : Choisir la source de la carte (pioche ou discard)
        source = self.ai.choose_source(self.grid, discard, safe_other_grids)
        card   = discard.pop() if source == 'D' else deck.pop()
        
        # Step 1b (optionnel) : Si on prend la carte du deck, on doit choisir si on la garde ou non
        keep = self.ai.choose_keep(card, self.grid, safe_other_grids)
        #print(f"Player {self.id} - Source: {source} - Card: {card.value}, Keep: {keep}")
        
        if keep or source == 'D':
            # Step 2a : Si on garde la carte, on doit choisir quel carte on va remplacer
            i, j = self.ai.choose_position(card, self.grid, safe_other_grids)
            self.grid[i][j].revealed = True
            #print(f"Player {self.id} - Replacing card at ({i}, {j}) with the value : {self.grid[i][j].value}")
            discard.append(self.grid[i][j])
            self.grid[i][j] = card
            self.grid[i][j].revealed = True
        else:
            # Step 2b : Si on ne garde pas la carte, on doit choisir quelle carte reveler
            i, j = self.ai.choose_reveal(self.grid)
            card.revealed = True
            discard.append(card)
            self.grid[i][j].revealed = True

        return source
