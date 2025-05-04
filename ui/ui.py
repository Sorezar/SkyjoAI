import config.config as cfg
from core import card   as cd
import pygame

class SkyjoUI:
    
    def __init__(self, game):
        self.game = game
        self.FONT = pygame.font.SysFont("arial", int(cfg.HEIGHT * 0.035))
    
    def get_color(self,value):
        if value is None: return cfg.GRAY
        if value in (-2, -1): return cfg.VALUE_COLORS['neg']
        if value == 0: return cfg.VALUE_COLORS['zero']
        if 1 <= value <= 4: return cfg.VALUE_COLORS['low']
        if 5 <= value <= 8: return cfg.VALUE_COLORS['mid']
        return cfg.VALUE_COLORS['high']
    
    def draw_card(self, card, surface, x, y):
        rect = pygame.Rect(x, y, cfg.CARD_WIDTH, cfg.CARD_HEIGHT)
        color = self.get_color(card.value) if card.revealed else cfg.DARK_GRAY
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, cfg.BLACK, rect, 2)
        if card.revealed:
            FONT = pygame.font.SysFont("arial", int(cfg.HEIGHT * 0.035))
            txt = FONT.render(str(card.value), True, cfg.BLACK)
            surface.blit(txt, (x + cfg.CARD_WIDTH/2 - txt.get_width()/2,
                                y + cfg.CARD_HEIGHT/2 - txt.get_height()/2))
    
    def draw_player(self, player, surface, is_current=False):
        FONT = pygame.font.SysFont("arial", int(cfg.HEIGHT * 0.035))
        name_surf = FONT.render(player.name, True, cfg.YELLOW)
        surface.blit(name_surf, (player.pos[0], player.pos[1] - FONT.get_height()))
        
        if is_current:
            highlight_rect = pygame.Rect(
                player.pos[0] - cfg.MARGIN,
                player.pos[1] - cfg.MARGIN - FONT.get_height(),
                cfg.GRID_COLS * (cfg.CARD_WIDTH + cfg.MARGIN) + cfg.MARGIN,
                cfg.GRID_ROWS * (cfg.CARD_HEIGHT + cfg.MARGIN) + FONT.get_height() + cfg.MARGIN
            )
            pygame.draw.rect(surface, cfg.RED, highlight_rect, 3)  # Red border for current player
        
        for i in range(cfg.GRID_ROWS):
            x_offset = 0
            for j in range(cfg.GRID_COLS):
                if player.grid[i][j] is None:  # Skip removed columns
                    continue
                x = player.pos[0] + x_offset * (cfg.CARD_WIDTH + cfg.MARGIN)
                y = player.pos[1] + i * (cfg.CARD_HEIGHT + cfg.MARGIN)
                self.draw_card(player.grid[i][j], surface, x, y)
                x_offset += 1
    
    def draw(self, screen):
        screen.fill(cfg.GRAY)
        txt = self.FONT.render(f"Manche {self.game.round} - Tour complet {self.game.turns}", True, cfg.YELLOW)
        screen.blit(txt, ((cfg.WIDTH - txt.get_width())/2, cfg.MARGIN/2))
        for idx, p in enumerate(self.game.players):
            self.draw_player(p, screen, is_current=(idx == self.game.current_player_index))

        cx, cy = cfg.WIDTH/2, cfg.HEIGHT/2
        self.draw_card(cd.Card(0), screen, cx - cfg.CARD_WIDTH - cfg.MARGIN, cy - cfg.CARD_HEIGHT/2)
        top = self.game.discard[-1]
        c = cd.Card(top); c.revealed = True
        self.draw_card(c, screen, cx + cfg.MARGIN, cy - cfg.CARD_HEIGHT/2)

        src_txt = self.FONT.render(f"Dernière pioche: {'Pioche' if self.game.last_source=='P' else 'Défausse'}", True, cfg.YELLOW)
        screen.blit(src_txt, ((cfg.WIDTH - src_txt.get_width())/2, cy + cfg.CARD_HEIGHT/2 + cfg.MARGIN))

        self.draw_score_table(screen)

    def draw_score_table(self, screen):
        table_width = max(cfg.WIDTH * 0.10, min(cfg.WIDTH * 0.25, cfg.WIDTH * 0.05 * (self.game.round + 2)))
        col_width = table_width / (self.game.round + 2)
        row_height = self.FONT.get_height() + 5
        start_x = (cfg.WIDTH - table_width) / 2
        start_y = cfg.HEIGHT - (len(self.game.players) * row_height) - cfg.MARGIN
        header = self.FONT.render("Total", True, cfg.WHITE)
        screen.blit(header, (start_x + (self.game.round + 1)*col_width, start_y - row_height))
        for i, p in enumerate(self.game.players):
            name_txt = self.FONT.render(p.name, True, cfg.WHITE)
            screen.blit(name_txt, (start_x, start_y + i*row_height))
            total = 0
            for r in range(1, self.game.round + 1):
                entry = next((l for l in self.game.log if l[0]==r and l[1]==p.name), None)
                score = entry[2] if entry else 0
                total += score
                color = cfg.VALUE_COLORS['zero'] if score == 0 else cfg.WHITE
                txt = self.FONT.render(str(score), True, color)
                screen.blit(txt, (start_x + r*col_width, start_y + i*row_height))
            total_txt = self.FONT.render(str(total), True, cfg.WHITE)
            screen.blit(total_txt, (start_x + (self.game.round + 1)*col_width, start_y + i*row_height))

    def show_results(self, screen):
        if any(pl.total_score >= cfg.MAX_POINTS for pl in self.game.players):
            winner = min(self.game.players, key=lambda p: p.total_score)
            win_txt = self.FONT.render(f"Vainqueur: {winner.name}", True, cfg.YELLOW)
            screen.blit(win_txt, ((cfg.WIDTH - win_txt.get_width())/2, cfg.HEIGHT/2))