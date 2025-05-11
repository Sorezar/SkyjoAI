import random
from config.config import GRID_ROWS, GRID_COLS, MAX_POINTS


class Card:
    def __init__(self, value):
        self.value = value
        self.revealed = False
        
class Scoreboard:
    def __init__(self, players):
        self.players = players
        self.reset()

    def update(self, player_index, score):
        self.scores[player_index].append(score)
        self.total_scores = [sum(scores) for scores in self.scores]

    def reset(self):
        self.scores = [[] for _ in range(len(self.players))]
        self.total_scores = [0 for _ in range(len(self.players))]
    
    def get_winner(self):
        min_score = min(self.total_scores)
        winners = [i for i, score in enumerate(self.total_scores) if score == min_score]
        return winners
        

class SkyjoGame:
    def __init__(self, players, scoreboard):
        self.players = players
        self.scoreboard = scoreboard
        self.round = 1
        self.reset_round()
        
    def reset_round(self):
        self.turns = 0
        self.log   = []
        self.last_source = ''
        self.current_player_index = 0
        self.round_over  = False
        self.finished    = False
        for player in self.players: 
            player.score = 0
            player.grid = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        self.deck = [Card(-2) for _ in range(5)]  + \
                    [Card(-1) for _ in range(10)] + \
                    [Card(0)  for _ in range(15)] + \
                    [Card(v)  for v in range(1, 13) for _ in range(10)]
        random.shuffle(self.deck)
        self.discard = [self.deck.pop()]
        self.give_initial_cards()
        for player in self.players: 
            pos = player.ai.initial_flip()
            for p in pos:
                player.grid[p[0]][p[1]].revealed = True

    def reset(self):
        self.reset_round()
        self.scoreboard.reset()
        self.round = 1

    def is_round_over(self):
        return any(player.all_revealed() for player in self.players)

    def give_initial_cards(self):
        for player in self.players:
            for i in range(GRID_ROWS):
                for j in range(GRID_COLS):
                    player.grid[i][j] = self.deck.pop()
                    
    def is_any_identical_column(self, grid):
        columns = list(zip(*grid))
        indexes = []

        for i, col in enumerate(columns):
            if all(card.revealed for card in col):
                values = [card.value for card in col]
                if len(set(values)) == 1:
                    indexes.append(i)
        return indexes
    
    def delete_column(self, grid, index):
        [[self.discard.append(cell) for j, cell in enumerate(row) if j in index] for row in grid]
        return [[cell for j, cell in enumerate(row) if j not in index] for row in grid]
    
    def step(self):
        p = self.players[self.current_player_index]
        self.last_source = p.take_turn(self.deck, self.discard)
        
        index  = self.is_any_identical_column(p.grid)
        if index: p.grid = self.delete_column(p.grid, index)
            
        if any(pl.all_revealed() for pl in self.players):
            self.round_over = True
            
