import pygame
import time
import config.config as cfg
from ui.ui   import SkyjoUI
from core.game import SkyjoGame


# Initialisation
pygame.init()

if __name__ == "__main__":
    
    pygame.display.set_caption("Skyjo UI & IA")
    screen = pygame.display.set_mode((int(cfg.WIDTH), int(cfg.HEIGHT)))
    clock  = pygame.time.Clock()
    game   = SkyjoGame()
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
            game.step(); last=time.time()
        
        ui.draw(screen)
        if any(pl.total_score>=cfg.MAX_POINTS for pl in game.players):
            ui.show_results(screen)
            
        pygame.display.flip()
        clock.tick(cfg.FPS)

    pygame.quit()
