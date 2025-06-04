from config.config import WIDTH, HEIGHT, AUTO_PLAY_DELAY, MAX_POINTS, FPS
from ui.ui         import SkyjoUI
from core.game     import SkyjoGame, Scoreboard
from core.player   import Player
from ai.random     import RandomAI
from ai.initial    import InitialAI
from ai.ml_xgboost_enhanced import XGBoostEnhancedAI

import pygame
import time
import json
import sys
import os

pygame.init()  # Initialize pygame before using any of its features

def play(game, ui, scoreboard):
    # Faire jouer un seul joueur
    game.step()
    ui.draw()
    pygame.display.flip()  # Forcer l'actualisation de l'affichage
    time.sleep(0.1)  # Petit délai pour voir l'action
                
    # Vérifier si la manche est terminée
    if game.round_over and game.current_player_index == 0:
        scores = []
        for pl in game.players:
            pl.reveal_all()
            scores.append(pl.round_score())

        scoreboard.update(scores, game.first_finisher)

        if any(s >= MAX_POINTS for s in scoreboard.total_scores):
            game.finished = True
        else:
            game.round += 1
            game.reset_round()

if __name__ == "__main__":
    
    # Créer l'IA DeepAI
    testai = XGBoostEnhancedAI()
    
    players = [
        Player(0, f"TestAI", testai),
        Player(1, f"IA_2", InitialAI()),
        Player(2, f"IA_3", InitialAI()),
        Player(3, f"IA_4", InitialAI())
    ]

    pygame.display.set_caption("Skyjo AI - Test")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock  = pygame.time.Clock()
    auto   = False
    last   = time.time()
    score  = Scoreboard(players)
    game   = SkyjoGame(players, score)
    ui     = SkyjoUI(game, score, screen)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE : 
                    play(game, ui, score)
                if ev.key == pygame.K_a : 
                    auto = not auto
                if ev.key == pygame.K_r and game.finished:
                    game.reset()

        if auto and not game.finished and time.time() - last > AUTO_PLAY_DELAY:
            play(game, ui, score)
            last = time.time()
                
        ui.draw()
        if game.finished:
            ui.show_results()
            
        pygame.display.flip()
        clock.tick(FPS) 