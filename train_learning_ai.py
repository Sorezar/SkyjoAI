from core.game import SkyjoGame, Scoreboard
from core.player import Player
from ai.learning import LearningAI
from ai.random import RandomAI
from ai.initial import InitialAI
import torch

NUM_GAMES = 10000

ai = LearningAI()
#opponents = [InitialAI(), InitialAI(), InitialAI()]
opponents = [RandomAI(), RandomAI(), RandomAI()]
players = [Player(0, "AI", ai)] + [Player(i+1, f"IA_{i+1}", opponents[i]) for i in range(3)]

for episode in range(NUM_GAMES):
    print(f"\n=== Partie {episode + 1}/{NUM_GAMES} ===")
    score = Scoreboard(players)
    game = SkyjoGame(players, score)
    
    while not any(s >= 100 for s in score.total_scores):
        game.step()
        game.current_player_index = (game.current_player_index + 1) % len(game.players)
        if game.current_player_index == 0:
            game.turns += 1

    # À la fin de la partie
    round_score = players[0].round_score()
    reward = -round_score  # Pénalité = score (on cherche à minimiser)
    ai.train(reward)

    if (episode + 1) % 100 == 0:
        print(f"Partie {episode+1} - Score de l'IA: {round_score}")
