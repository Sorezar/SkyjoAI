from config.config import GRID_ROWS, GRID_COLS

class Player:
    def __init__(self, id, name, ai):
        self.id    = id
        self.name  = name
        self.score = 0
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

    def take_turn(self, deck, discard):
        
        # Step 1 : Choisir la source de la carte (pioche ou discard)
        source = self.ai.choose_source(self.grid, discard)
        card   = discard.pop() if source == 'D' else deck.pop()
        
        # Step 1b (optionnel) : Si on prend la carte du deck, on doit choisir si on la garde ou non
        
        keep = self.ai.choose_keep(card, self.grid)
        print(f"Player {self.id} - Source: {source} - Card: {card.value}, Keep: {keep}")
        
        if keep or source == 'D':
            # Step 2a : Si on garde la carte, on doit choisir quel carte on va remplacer
            i, j = self.ai.choose_position(card, self.grid)
            self.grid[i][j].revealed = True
            print(f"Player {self.id} - Replacing card at ({i}, {j}) with the value : {self.grid[i][j].value}")
            discard.append(self.grid[i][j])
            self.grid[i][j] = card
            self.grid[i][j].revealed = True
        else :
            # Step 2b : Si on ne garde pas la carte, on doit choisir quelle carte révéler
            i, j = self.ai.choose_reveal(self.grid)
            card.revealed = True
            discard.append(card)
            self.grid[i][j].revealed = True


        return source
