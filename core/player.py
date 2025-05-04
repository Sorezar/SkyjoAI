import random
import config.config as cfg
from core import card as c

class Player:
    def __init__(self, name, pos, ai):
        self.name        = name
        self.total_score = 0
        self.pos         = pos
        self.ai          = ai
        self.reset_grid()

    def reset_grid(self):
        self.grid = [[c.Card(random.randint(-2, 12)) for _ in range(cfg.GRID_COLS)] for _ in range(cfg.GRID_ROWS)]
        for _ in range(2):
            i, j = random.randrange(cfg.GRID_ROWS), random.randrange(cfg.GRID_COLS)
            self.grid[i][j].revealed = True

    def all_revealed(self):
        return all(c.revealed for row in self.grid for c in row)

    def reveal_all(self):
        for row in self.grid:
            for c in row:
                c.revealed = True

    def round_score(self):
        return sum(c.value for row in self.grid for c in row)

    def take_turn(self, deck, discard):
        top_discard = discard[-1] if discard else None
        action = self.ai.choose_action(self, top_discard, deck)

        # Choix de la carte
        if action['source'] == 'D' and discard:
            card_val = discard.pop()
        else:
            card_val = deck.pop()

        # Emplacement où jouer
        if action['position']:
            i, j = action['position']
            discard.append(self.grid[i][j].value)
            new_card = c.Card(card_val)
            new_card.revealed = True
            self.grid[i][j] = new_card
        else:
            discard.append(card_val)

        return action['source']
