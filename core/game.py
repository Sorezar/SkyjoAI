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

    def update(self, scores, first_finisher):
        #print(f"First finisher: {self.players[first_finisher].name}")
        if scores[first_finisher] is not min(scores):
            #print(f"Warning: {self.players[first_finisher].name} is not the lowest scorer.")
            scores[first_finisher] *= 2     
        for i in range(len(self.players)):
            self.scores[i].append(scores[i])
            self.total_scores[i] += scores[i]
        #print(f"Scores: {scores}")
        #print(f"Total scores: {self.total_scores}")

    def reset(self):
        self.scores = [[] for _ in range(len(self.players))]
        self.total_scores = [0 for _ in range(len(self.players))]
    
    def get_winner(self):
        min_score = min(self.total_scores)
        winners = [i for i, score in enumerate(self.total_scores) if score == min_score]
        return winners
        

class SkyjoGame:
    def __init__(self, players, scoreboard):
        self.players    = players
        self.scoreboard = scoreboard
        self.round      = 1
        self.reset_round()
        
    def reset_round(self):
        self.turns = 0
        self.log   = []
        self.last_source    = ''
        self.first_finisher = None
        self.round_over     = False
        self.finished       = False
        self.current_player_index = 0
        for player in self.players: 
            player.score = 0
            player.grid  = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
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

    def refill_deck_if_needed(self):
        """Remele la defausse dans le deck si le deck est vide"""
        if len(self.deck) == 0 and len(self.discard) > 1:
            # Garder la derniere carte de la defausse
            last_discard = self.discard.pop()
            
            # Remettre toutes les autres cartes dans le deck
            self.deck = self.discard[:]
            self.discard = [last_discard]
            
            # Remelanger le deck
            random.shuffle(self.deck)
            
            #print(f"🔄 Deck remelange avec {len(self.deck)} cartes de la defausse")
        elif len(self.deck) == 0 and len(self.discard) <= 1:
            # Cas extreme : plus assez de cartes (ne devrait pas arriver en pratique)
            #print("⚠️ Plus de cartes disponibles - fin forcee de la manche")
            # Forcer la fin de la manche en revelant toutes les cartes du joueur actuel
            self.players[self.current_player_index].reveal_all()
            self.round_over = True
            self.first_finisher = self.current_player_index

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
    
    def delete_column(self, grid, indexes):
        """Supprime les colonnes complètes spécifiées par leurs indices"""
        # Ajouter les cartes des colonnes supprimées à la défausse
        for row in grid:
            for j in indexes:
                if j < len(row):
                    self.discard.append(row[j])
        
        # Créer une nouvelle grille sans les colonnes supprimées
        new_grid = []
        for row in grid:
            new_row = [cell for j, cell in enumerate(row) if j not in indexes]
            new_grid.append(new_row)
        
        return new_grid
    
    def step(self):
        # Verifier et remeler le deck si necessaire avant que le joueur prenne son tour
        self.refill_deck_if_needed()
        
        p = self.players[self.current_player_index]
        other_p_grids = [self.players[i].grid for i in range(len(self.players)) if i != self.current_player_index]
        self.last_source = p.take_turn(self.deck, self.discard, other_p_grids)
        
        index = self.is_any_identical_column(p.grid)
        if index: 
            p.grid = self.delete_column(p.grid, index)
        
        if p.all_revealed():
            self.round_over = True
            self.first_finisher = self.current_player_index
            
        # Passer au joueur suivant
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self.turns += 1
        
        # Verifier si la manche est terminee (tous les joueurs ont joue)
        if self.round_over and self.current_player_index == 0:
            # Calculer les scores de la manche
            scores = [player.round_score() for player in self.players]
            self.scoreboard.update(scores, self.first_finisher)
            
            # Verifier si un joueur a atteint le score maximum
            if any(score >= MAX_POINTS for score in self.scoreboard.total_scores):
                self.finished = True
            else:
                # Commencer une nouvelle manche
                self.round += 1
                self.reset_round()
