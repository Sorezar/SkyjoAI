import random
import pygame
import config.config as cfg
from core import card as c

class Player:
    def __init__(self, name, pos):
        self.name = name
        self.total_score = 0
        self.pos = pos
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
        source = ''
        top = discard[-1] if discard else None
        if top is not None and top < 5:
            card_val = discard.pop()
            source = 'D'
        else:
            card_val = deck.pop()
            source = 'P'
        unrevealed = [(i,j) for i in range(cfg.GRID_ROWS) for j in range(cfg.GRID_COLS) if not self.grid[i][j].revealed]
        if unrevealed:
            i,j = random.choice(unrevealed)
            discard.append(self.grid[i][j].value)
            new = c.Card(card_val)
            new.revealed = True
            self.grid[i][j] = new
        else:
            discard.append(card_val)
        return source
