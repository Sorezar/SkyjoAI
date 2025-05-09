from config.config import WIDTH, HEIGHT, AUTO_PLAY_DELAY, MAX_POINTS, FPS
from ui.ui         import SkyjoUI
from core.game     import SkyjoGame, Scoreboard
from core.player   import Player
from ai.optimized  import OptimizedAI
from ai.random     import RandomAI
from ai.initial    import InitialAI

import pygame
import time
import json
import sys

pygame.init()  # Initialize pygame before using any of its features

def simulate_games(num_games, players):
    metrics = {
        "total_games": num_games,
        "wins_by_ai": {},
        "wins_by_initial_cards": {}
    }

    for _ in range(num_games):
        print(f"Simulating game {_ + 1}/{num_games}")
        game = SkyjoGame(players)
        while not any(pl.score >= MAX_POINTS for pl in game.players):
            game.step()

        winner = min(game.players, key=lambda p: p.score)
        ia_name = winner.ai.__class__.__name__
        if ia_name not in metrics["wins_by_ai"]:
            metrics["wins_by_ai"][ia_name] = 0
        metrics["wins_by_ai"][ia_name] += 1
        
        print(f"Winner: {winner.name} with score {winner.score}")

        initial_card_sum = sum(c.value for row in winner.grid for c in row if c is not None)
        metrics["wins_by_initial_cards"].setdefault(initial_card_sum, 0)
        metrics["wins_by_initial_cards"][initial_card_sum] += 1

    with open("simulation_results.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    
    players = [
        Player(0, f"IA_1", InitialAI()),
        Player(1, f"IA_2", InitialAI()),
        Player(2, f"IA_3", InitialAI()),
        Player(3, f"IA_4", InitialAI())
    ]
    
    if len(sys.argv) > 1 and sys.argv[1] == "simulate":
        num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
        simulate_games(num_games, players)
    else:
        pygame.display.set_caption("Skyjo AI")
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock  = pygame.time.Clock()
        auto   = False
        last   = time.time()
        score  = Scoreboard(players)
        game   = SkyjoGame(players, score)
        ui     = SkyjoUI(game)

        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_SPACE : 
                        game.step()
                    if ev.key == pygame.K_a : 
                        auto = not auto
                    if ev.key == pygame.K_r and game.finished:
                        game.reset()

            if auto and not game.finished and time.time() - last > AUTO_PLAY_DELAY:
                game.step()
                last = time.time()

            ui.draw(screen)
            if game.finished:
                ui.show_results(screen)
            
            pygame.display.flip()
            clock.tick(FPS)
