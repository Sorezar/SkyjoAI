import random
import config.config as cfg
import core.player as pl

class SkyjoGame:
    def __init__(self):
        positions = [
            (cfg.WIDTH * 0.05, cfg.HEIGHT * 0.05),
            (cfg.WIDTH * 0.80, cfg.HEIGHT * 0.05),
            (cfg.WIDTH * 0.05, cfg.HEIGHT * 0.70),
            (cfg.WIDTH * 0.80, cfg.HEIGHT * 0.70)
        ]
        self.players = [pl.Player(f"IA_{i+1}", positions[i]) for i in range(4)]
        self.round = 1
        self.turns = 0
        self.log = []
        self.last_source = ''
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
        p = self.players[self.current]
        self.last_source = p.take_turn(self.deck, self.discard)
        self.current = (self.current+1)%len(self.players)
        if self.current == 0:
            self.turns += 1
        if any(pl.all_revealed() for pl in self.players):
            self.round_over = True
            for pl in self.players: pl.reveal_all()
            for pl in self.players:
                sc = pl.round_score()
                pl.total_score += sc
                self.log.append((self.round, pl.name, sc, pl.total_score))

        if self.round_over and not any(p.total_score >= cfg.MAX_POINTS for p in self.players):
            self.round += 1
            self.start_round()
