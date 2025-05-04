import pygame
import time
import config.config as cfg
from ui.ui   import SkyjoUI
from core.game import SkyjoGame
from core.player import Player
from ai.random import RandomAI
from ai.qlearning import QLearningAI
from ai.optimized import OptimizedAI


# Initialisation
pygame.init()

if __name__ == "__main__":
    
    pygame.display.set_caption("Skyjo UI & IA")
    screen = pygame.display.set_mode((int(cfg.WIDTH), int(cfg.HEIGHT)))
    clock  = pygame.time.Clock()
    
    positions = [
            (cfg.WIDTH * 0.05, cfg.HEIGHT * 0.05),
            (cfg.WIDTH * 0.80, cfg.HEIGHT * 0.05),
            (cfg.WIDTH * 0.05, cfg.HEIGHT * 0.70),
            (cfg.WIDTH * 0.80, cfg.HEIGHT * 0.70)
        ]
    players     = [
        Player(f"IA_1", positions[0], OptimizedAI()),
        Player(f"IA_2", positions[1], OptimizedAI()),
        Player(f"IA_3", positions[2], OptimizedAI()),
        Player(f"IA_4", positions[3], OptimizedAI())
    ]
    
    game   = SkyjoGame(players)
    ui     = SkyjoUI(game)
    auto   = False
    last   = time.time()
    
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: break
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE: game.step()
                if ev.key == pygame.K_a: auto=not auto
                
        if auto and not any(pl.total_score>=cfg.MAX_POINTS for pl in game.players) and time.time()-last> cfg.AUTO_PLAY_DELAY:
            game.step()
            last=time.time()
        
        ui.draw(screen)
        if any(pl.total_score>=cfg.MAX_POINTS for pl in game.players):
            ui.show_results(screen)
            
        pygame.display.flip()
        clock.tick(cfg.FPS)

    pygame.quit()
