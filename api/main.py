# -*- coding: utf-8 -*-
"""
API FastAPI pour Skyjo AI - Backend pour l'interface React
Expose les fonctionnalites du moteur de jeu Python via une API REST
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import logging
from datetime import datetime

# Imports du moteur de jeu existant
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.game import SkyjoGame, Scoreboard
from core.player import Player

# Imports des IA disponibles (avec gestion d'erreur)
from ai.initial import InitialAI

# Imports optionnels avec gestion d'erreur
optional_ai_imports = {}

try:
    from ai.advanced_dominant import AdvancedDominantAI
    optional_ai_imports["advanced_dominant"] = AdvancedDominantAI
except ImportError as e:
    print(f"WARNING: AdvancedDominantAI non disponible: {e}")

try:
    from ai.unsupervised_pattern_ai import UnsupervisedPatternAI
    optional_ai_imports["unsupervised_pattern"] = UnsupervisedPatternAI
except ImportError as e:
    print(f"WARNING: UnsupervisedPatternAI non disponible: {e}")

try:
    from ai.hybrid_elite_ai import HybridEliteAI
    optional_ai_imports["hybrid_elite"] = HybridEliteAI
except ImportError as e:
    print(f"WARNING: HybridEliteAI non disponible: {e}")

try:
    from ai.adaptive_ml_ai import AdaptiveMLAI
    optional_ai_imports["adaptive_ml"] = AdaptiveMLAI
except ImportError as e:
    print(f"WARNING: AdaptiveMLAI non disponible: {e}")

try:
    from ai.champion_elite_ai import ChampionEliteAI
    optional_ai_imports["champion_elite"] = ChampionEliteAI
except ImportError as e:
    print(f"WARNING: ChampionEliteAI non disponible: {e}")

# Imports optionnels
try:
    from ai.ml_xgboost import XGBoostSkyjoAI
    XGBOOST_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: XGBoostSkyjoAI non disponible: {e}")
    XGBOOST_AVAILABLE = False

try:
    from ai.ml_xgboost_enhanced import XGBoostEnhancedAI
    XGBOOST_ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: XGBoostEnhancedAI non disponible: {e}")
    XGBOOST_ENHANCED_AVAILABLE = False

# Configuration du logging avec encodage UTF-8
import io

# Forcer l'encodage UTF-8 pour éviter les erreurs charmap sous Windows
if sys.platform.startswith('win'):
    # Rediriger stdout et stderr vers UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Skyjo AI API",
    description="API REST pour le jeu Skyjo avec IA",
    version="1.0.0"
)

# Configuration CORS pour React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# MODELES PYDANTIC
# ===============================

class AIConfig(BaseModel):
    id: str
    name: str
    type: str

class GameConfig(BaseModel):
    players: List[AIConfig]
    game_id: Optional[str] = None

class CardData(BaseModel):
    value: int
    revealed: bool
    id: str

class PlayerData(BaseModel):
    id: int
    name: str
    isAI: bool
    score: int
    current_round_score: int
    grid: List[List[CardData]]

class GameState(BaseModel):
    game_id: str
    players: List[PlayerData]
    current_player: int
    turn: int
    round: int
    phase: str
    deck_count: int
    discard_pile: List[CardData]
    last_action: Optional[str]
    finished: bool
    scoreboard: Optional[Dict[str, Any]] = None

# ===============================
# REGISTRE DES PARTIES ACTIVES
# ===============================

active_games: Dict[str, Dict[str, Any]] = {}

# ===============================
# MAPPING DES IA DISPONIBLES
# ===============================

AI_CLASSES = {
    "initial": InitialAI
}

# Ajouter les IA optionnelles disponibles
AI_CLASSES.update(optional_ai_imports)

if XGBOOST_AVAILABLE:
    AI_CLASSES["xgboost"] = XGBoostSkyjoAI

if XGBOOST_ENHANCED_AVAILABLE:
    AI_CLASSES["xgboost_enhanced"] = XGBoostEnhancedAI

# ===============================
# FONCTIONS UTILITAIRES
# ===============================

def card_to_dict(card):
    """Convertit une carte du jeu en dictionnaire"""
    if hasattr(card, 'value'):
        # Carte du moteur de jeu (objet Card)
        return {
            "value": card.value,
            "revealed": getattr(card, 'revealed', True),
            "id": f"{card.value}_{uuid.uuid4().hex[:8]}"
        }
    else:
        # Carte simple (nombre)
        return {
            "value": card,
            "revealed": True,
            "id": f"{card}_{uuid.uuid4().hex[:8]}"
        }

def grid_to_dict(grid):
    """Convertit une grille de cartes en format dictionnaire"""
    result = []
    for row_idx, row in enumerate(grid):
        row_data = []
        for col_idx, card in enumerate(row):
            card_data = {
                "value": card.value,
                "revealed": card.revealed,
                "id": f"{row_idx}-{col_idx}"
            }
            row_data.append(card_data)
        result.append(row_data)
    return result

def player_to_dict(player, player_id, scoreboard):
    """Convertit un joueur en dictionnaire avec les bons scores"""
    # Score total des manches terminées
    total_score = scoreboard.total_scores[player_id] if player_id < len(scoreboard.total_scores) else 0
    
    # Score de la manche en cours (seulement les cartes révélées)
    current_round_score = sum(
        card.value for row in player.grid for card in row 
        if card and card.revealed
    )
    
    return {
        "id": player_id,
        "name": player.name,
        "isAI": True,
        "score": total_score,  # Score total des manches terminées
        "current_round_score": current_round_score,  # Score de la manche en cours
        "grid": grid_to_dict(player.grid)
    }

def game_to_state(game, game_id):
    """Convertit l'etat du jeu en format API"""
    # Récupérer les statistiques du jeu
    game_data = active_games.get(game_id, {})
    turn_count = game_data.get("turn_count", 1)
    
    return {
        "game_id": game_id,
        "players": [player_to_dict(player, idx, game.scoreboard) for idx, player in enumerate(game.players)],
        "current_player": game.current_player_index,
        "turn": turn_count,
        "round": game.round,
        "phase": "playing" if not game.finished else "finished",
        "deck_count": len(game.deck) if hasattr(game, 'deck') and game.deck else 0,
        "discard_pile": [card_to_dict(card) for card in getattr(game, 'discard', [])],
        "last_action": game_data.get("last_action", "En cours..."),
        "finished": game.finished,
        "scoreboard": {
            "total_scores": game.scoreboard.total_scores,
            "round_scores": game.scoreboard.scores
        }
    }

# ===============================
# ENDPOINTS API
# ===============================

@app.get("/")
@app.head("/")
async def root():
    return {
        "message": "Skyjo AI API v1.0",
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "available_ais": len(AI_CLASSES),
        "active_games": len(active_games)
    }

@app.get("/ai/available")
async def get_available_ais():
    """Retourne la liste des IA disponibles avec leurs statistiques"""
    ai_list = []
    
    ai_stats = {
        "initial": {"avgScore": 20.9, "winRate": 20, "level": "Debutant"},
        "machine_learning": {"avgScore": 32.52, "winRate": 8, "level": "Intermediaire"},
        "unsupervised_deep": {"avgScore": 28.5, "winRate": 12, "level": "Avance"},
        "advanced_dominant": {"avgScore": 19.72, "winRate": 16, "level": "Expert"},
        "unsupervised_pattern": {"avgScore": 27.76, "winRate": 28, "level": "Expert"},
        "hybrid_elite": {"avgScore": 25.3, "winRate": 22, "level": "Expert"},
        "adaptive_ml": {"avgScore": 26.8, "winRate": 18, "level": "Expert"},
        "champion_elite": {"avgScore": 24.1, "winRate": 25, "level": "Expert"},
        "xgboost": {"avgScore": 18.2, "winRate": 35, "level": "Champion"},
        "xgboost_enhanced": {"avgScore": 13.68, "winRate": 56, "level": "Champion"}
    }
    
    for ai_id, ai_class in AI_CLASSES.items():
        stats = ai_stats.get(ai_id, {"avgScore": 25.0, "winRate": 15, "level": "Intermediaire"})
        
        ai_info = {
            "id": ai_id,
            "name": ai_class.__name__,
            "available": True,
            "avgScore": stats["avgScore"],
            "winRate": stats["winRate"],
            "level": stats["level"],
            "description": f"IA {stats['level']} - Score moyen: {stats['avgScore']}"
        }
        ai_list.append(ai_info)
    
    return {"available_ais": ai_list}

@app.post("/game/create")
async def create_game(config: GameConfig):
    """Cree une nouvelle partie avec les IA specifiees"""
    try:
        game_id = config.game_id or str(uuid.uuid4())
        
        # Creer les joueurs IA
        players = []
        for idx, ai_config in enumerate(config.players):
            if ai_config.type not in AI_CLASSES:
                raise HTTPException(status_code=400, detail=f"IA non disponible: {ai_config.type}")
            
            ai_instance = AI_CLASSES[ai_config.type]()
            player = Player(idx, ai_config.name, ai_instance)
            players.append(player)
        
        # Creer le scoreboard et le jeu
        scoreboard = Scoreboard(players)
        game = SkyjoGame(players, scoreboard)
        
        # Sauvegarder la partie
        active_games[game_id] = {
            "game": game,
            "created_at": datetime.now(),
            "turn_count": 1,
            "last_action": "Partie creee"
        }
        
        logger.info(f"Partie creee: {game_id} avec {len(players)} joueurs")
        
        return {
            "game_id": game_id,
            "status": "created",
            "players_count": len(players),
            "initial_state": game_to_state(game, game_id)
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la creation de la partie: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur creation partie: {str(e)}")

@app.get("/game/{game_id}/state")
async def get_game_state(game_id: str):
    """Recupere l'etat actuel de la partie"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Partie non trouvee")
    
    game_data = active_games[game_id]
    game = game_data["game"]
    
    return game_to_state(game, game_id)

@app.post("/game/{game_id}/step")
async def game_step(game_id: str):
    """Fait avancer la partie d'un tour (action de l'IA courante)"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Partie non trouvee")
    
    try:
        game_data = active_games[game_id]
        game = game_data["game"]
        
        if game.finished:
            return {"status": "game_finished", "state": game_to_state(game, game_id)}
        
        # Sauvegarder l'état avant le tour
        current_player = game.players[game.current_player_index]
        player_name = current_player.name
        
        # Score de la manche en cours avant le tour
        current_round_score_before = sum(
            card.value for row in current_player.grid for card in row 
            if card and card.revealed
        )
        
        # Capturer l'état initial pour les logs détaillés
        initial_deck_size = len(game.deck)
        initial_discard_top = game.discard[-1].value if game.discard else None
        
        # Variables pour capturer les détails de l'action
        source_used = None
        keep_decision = None
        replaced_card_value = None
        new_card_value = None
        revealed_card_value = None
        
        # Hook personnalisé pour capturer les décisions
        original_choose_source = current_player.ai.choose_source
        original_choose_keep = current_player.ai.choose_keep
        original_choose_position = current_player.ai.choose_position
        original_choose_reveal = current_player.ai.choose_reveal
        
        def capture_source(grid, discard, other_grids):
            nonlocal source_used
            decision = original_choose_source(grid, discard, other_grids)
            source_used = "Défausse" if decision == 'D' else "Pioche"
            return decision
        
        def capture_keep(card, grid, other_grids):
            nonlocal keep_decision, new_card_value
            decision = original_choose_keep(card, grid, other_grids)
            keep_decision = "Oui" if decision else "Non"
            new_card_value = card.value
            return decision
        
        def capture_position(card, grid, other_grids):
            nonlocal replaced_card_value, new_card_value
            position = original_choose_position(card, grid, other_grids)
            i, j = position
            if i < len(grid) and j < len(grid[i]) and grid[i][j] is not None:
                replaced_card_value = grid[i][j].value
            new_card_value = card.value
            return position
        
        def capture_reveal(grid):
            nonlocal revealed_card_value
            position = original_choose_reveal(grid)
            i, j = position
            if i < len(grid) and j < len(grid[i]) and grid[i][j] is not None:
                revealed_card_value = grid[i][j].value
            return position
        
        # Remplacer temporairement les méthodes
        current_player.ai.choose_source = capture_source
        current_player.ai.choose_keep = capture_keep
        current_player.ai.choose_position = capture_position
        current_player.ai.choose_reveal = capture_reveal
        
        try:
            # Exécuter un tour
            game.step()
        finally:
            # Restaurer les méthodes originales
            current_player.ai.choose_source = original_choose_source
            current_player.ai.choose_keep = original_choose_keep
            current_player.ai.choose_position = original_choose_position
            current_player.ai.choose_reveal = original_choose_reveal
        
        # Calculer les changements après le tour
        current_round_score_after = sum(
            card.value for row in current_player.grid for card in row 
            if card and card.revealed
        )
        
        score_change = current_round_score_after - current_round_score_before
        
        # Construire le message de log détaillé
        log_parts = []
        
        # 1. Source
        log_parts.append(f"Source : {source_used}")
        
        # 2. Keep (si source = Pioche)
        if source_used == "Pioche" and keep_decision is not None:
            log_parts.append(f"Keep : {keep_decision}")
        
        # 3. Carte remplacée (si Keep=Oui ou Source=Défausse)
        if ((source_used == "Pioche" and keep_decision == "Oui") or 
            (source_used == "Défausse")) and replaced_card_value is not None:
            log_parts.append(f"Carte remplacée : {replaced_card_value} par {new_card_value}")
        
        # 4. Carte révélée (si Keep=Non et source=Pioche)
        if source_used == "Pioche" and keep_decision == "Non" and revealed_card_value is not None:
            log_parts.append(f"Carte révélée : {revealed_card_value}")
        
        # 5. Score final
        score_change_str = f"+{score_change}" if score_change > 0 else (f"{score_change}" if score_change < 0 else "+0")
        log_parts.append(f"Score : {current_round_score_after}({score_change_str})")
        
        # Assembler le message final
        action_desc = f"{player_name} | " + " | ".join(log_parts)
        
        # Ajouter des informations sur la manche et le tour
        action_desc += f" | Manche {game.round}, Tour {game_data['turn_count'] + 1}"
        
        # Vérifier si la manche est terminée
        if game.round_over:
            action_desc += " | 🏁 Manche terminée!"
        
        # Mettre à jour les métadonnées
        game_data["turn_count"] += 1
        game_data["last_action"] = action_desc
        
        logger.info(f"Tour exécuté: {action_desc}")
        
        return {
            "status": "step_completed",
            "state": game_to_state(game, game_id)
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'execution du tour: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur execution tour: {str(e)}")

@app.post("/game/{game_id}/auto-play")
async def auto_play_game(game_id: str, steps: int = 10):
    """Fait jouer automatiquement plusieurs tours"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Partie non trouvee")
    
    try:
        game_data = active_games[game_id]
        game = game_data["game"]
        
        steps_executed = 0
        while not game.finished and steps_executed < steps:
            game.step()
            steps_executed += 1
        
        game_data["turn_count"] += steps_executed
        game_data["last_action"] = f"{steps_executed} tours automatiques executes"
        
        return {
            "status": "auto_play_completed",
            "steps_executed": steps_executed,
            "state": game_to_state(game, game_id)
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'auto-play: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/game/{game_id}")
async def delete_game(game_id: str):
    """Supprime une partie"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Partie non trouvee")
    
    del active_games[game_id]
    logger.info(f"Partie supprimee: {game_id}")
    
    return {"status": "deleted", "game_id": game_id}

@app.get("/games")
async def list_games():
    """Liste toutes les parties actives"""
    games_list = []
    for game_id, game_data in active_games.items():
        games_list.append({
            "game_id": game_id,
            "created_at": game_data["created_at"].isoformat(),
            "turn_count": game_data["turn_count"],
            "players_count": len(game_data["game"].players),
            "finished": game_data["game"].finished
        })
    
    return {"active_games": games_list}

if __name__ == "__main__":
    import uvicorn
    print("=== Demarrage de l'API Skyjo AI sur http://localhost:8000 ===")
    print(f"IA disponibles: {list(AI_CLASSES.keys())}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 