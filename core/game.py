import random
import config.config as cfg
import core.player as pl
from ai.random import RandomAI

class SkyjoGame:
    def __init__(self, players):
        self.players     = players
        self.round       = 1
        self.turns       = 0
        self.log         = []
        self.last_source = ''
        self.current_player_index = 0  # Track the current player's index
        self.start_round()

    def start_round(self):
        self.deck = [v for v in range(-2,13) for _ in range(10)]
        random.shuffle(self.deck)
        self.discard = [self.deck.pop()]
        for p in self.players: p.reset_grid()
        self.current = 0
        self.round_turns = 0
        self.round_over = False

    
    def step(self):
        if self.round_over: return
        
        p = self.players[self.current_player_index]
        self.last_source = p.take_turn(self.deck, self.discard)
        self.current_player_index = (self.current_player_index + 1) % len(self.players)  # Update current player index
        if self.current_player_index == 0:
            self.turns += 1
            
        if any(pl.all_revealed() for pl in self.players):
            self.round_over = True
            for pl in self.players: pl.reveal_all()
            for pl in self.players:
                sc = pl.round_score()
                pl.total_score += sc
                self.log.append((self.round, pl.name, sc, pl.total_score))
            
            # Check and remove columns with identical values
            for pl in self.players:
                for col in range(cfg.GRID_COLS):
                    column_values = [pl.grid[row][col].value for row in range(cfg.GRID_ROWS) if pl.grid[row][col] is not None and pl.grid[row][col].revealed]
                    if len(column_values) == cfg.GRID_ROWS and all(v == column_values[0] for v in column_values):
                        for row in range(cfg.GRID_ROWS):
                            pl.grid[row][col] = None  # Remove the column by setting its cards to None

        if self.round_over and not any(p.total_score >= cfg.MAX_POINTS for p in self.players):
            self.round += 1
            self.start_round()
