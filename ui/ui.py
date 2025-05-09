from config.config import WIDTH, HEIGHT, MAX_POINTS, CARD_HEIGHT, CARD_WIDTH, MARGIN, GRID_COLS, GRID_ROWS
from config.config import VALUE_COLORS, GRAY, DARK_GRAY, BLACK, WHITE, RED, YELLOW
from core.game import Card
import pygame


class SkyjoUI:
    
    def __init__(self, game):
        self.game = game
        self.FONT = pygame.font.SysFont("arial", int(HEIGHT * 0.035))
    
    def get_color(self, value):
        if value is None     : return GRAY
        if value in (-2, -1) : return VALUE_COLORS['neg']
        if value == 0        : return VALUE_COLORS['zero']
        if 1 <= value <= 4   : return VALUE_COLORS['low']
        if 5 <= value <= 8   : return VALUE_COLORS['mid']
        return VALUE_COLORS['high']
    
    def draw_card(self, card, surface, x, y):
        rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
        color = self.get_color(card.value) if card.revealed else DARK_GRAY
        pygame.draw.rect(surface, color, rect)
        pygame.draw.rect(surface, BLACK, rect, 2)
        if card.revealed:
            FONT = pygame.font.SysFont("arial", int(HEIGHT * 0.035))
            txt = FONT.render(str(card.value), True, BLACK)
            surface.blit(txt, (x + CARD_WIDTH/2 - txt.get_width()/2,
                               y + CARD_HEIGHT/2 - txt.get_height()/2))
    
    def draw_player(self, player, surface, is_current=False):
        
        positions = [
            (WIDTH * 0.05, HEIGHT * 0.05),
            (WIDTH * 0.80, HEIGHT * 0.05),
            (WIDTH * 0.80, HEIGHT * 0.70),
            (WIDTH * 0.05, HEIGHT * 0.70)
        ]
        
        FONT = pygame.font.SysFont("arial", int(HEIGHT * 0.035))
        name_surf = FONT.render(player.name, True, YELLOW)
        surface.blit(name_surf, (positions[player.id][0], positions[player.id][1] - FONT.get_height()))
        
        if is_current:
            highlight_rect = pygame.Rect(
                positions[player.id][0] - MARGIN,
                positions[player.id][1] - MARGIN - FONT.get_height(),
                GRID_COLS * (CARD_WIDTH + MARGIN) + MARGIN,
                GRID_ROWS * (CARD_HEIGHT + MARGIN) + FONT.get_height() + MARGIN
            )
            pygame.draw.rect(surface, RED, highlight_rect, 3)  # Red border for current player
        
        for i in range(GRID_ROWS):
            x_offset = 0
            for j in range(len(player.grid[i])):  # Use the actual length of the row to avoid index errors
                if player.grid[i][j] is None:  # Skip removed columns
                    continue
                x = positions[player.id][0] + x_offset * (CARD_WIDTH + MARGIN)
                y = positions[player.id][1] + i * (CARD_HEIGHT + MARGIN)
                self.draw_card(player.grid[i][j], surface, x, y)
                x_offset += 1
    
    def draw(self, screen):
        screen.fill(GRAY)
        txt = self.FONT.render(f"Manche {self.game.round} - Tour complet {self.game.turns}", True, YELLOW)
        screen.blit(txt, ((WIDTH - txt.get_width())/2, MARGIN/2))
        for idx, p in enumerate(self.game.players):
            self.draw_player(p, screen, is_current=(idx == self.game.current_player_index))

        cx, cy = WIDTH/2, HEIGHT/2
        self.draw_card(Card(0), screen, cx - CARD_WIDTH - MARGIN, cy - CARD_HEIGHT/2)
        c = self.game.discard[-1]
        c.revealed = True
        self.draw_card(c, screen, cx + MARGIN, cy - CARD_HEIGHT/2)

        src_txt = self.FONT.render(f"Dernière pioche: {'Pioche' if self.game.last_source=='P' else 'Défausse'}", True, YELLOW)
        screen.blit(src_txt, ((WIDTH - src_txt.get_width())/2, cy + CARD_HEIGHT/2 + MARGIN))

        if self.game.round > 1:
            self.draw_score_table(screen)

    def draw_score_table(self, screen):
        table_width = max(WIDTH * 0.10, min(WIDTH * 0.25, WIDTH * 0.05 * (self.game.round + 2)))
        col_width   = table_width / (self.game.round + 2)
        row_height  = self.FONT.get_height() + 5
        
        start_x = (WIDTH - table_width) / 2
        start_y = HEIGHT - (len(self.game.players) * row_height) - MARGIN
        header  = self.FONT.render("Total", True, WHITE)
        screen.blit(header, (start_x + (self.game.round + 1)*col_width, start_y - row_height))
        
        for i, p in enumerate(self.game.players):
            name_txt = self.FONT.render(p.name, True, WHITE)
            screen.blit(name_txt, (start_x, start_y + i*row_height))
            total = 0
            for r in range(1, self.game.round):
                score = self.game.scoreboard.scores[p.id][r-1]
                total += score
                txt = self.FONT.render(str(score), True, WHITE)
                screen.blit(txt, (start_x + r*col_width, start_y + i*row_height))
                
            total_txt = self.FONT.render(str(total), True, WHITE)
            screen.blit(total_txt, (start_x + (self.game.round + 1)*col_width, start_y + i*row_height))

    def show_results(self, screen):
        winner = min(self.game.players, key=lambda p: p.score)
        win_txt = self.FONT.render(f"Vainqueur: {winner.name}", True, YELLOW)
        screen.blit(win_txt, ((WIDTH - win_txt.get_width())/2, HEIGHT/2))