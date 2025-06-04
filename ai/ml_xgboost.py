import numpy as np
import pandas as pd
import random
import os
import pickle
from collections import Counter
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class XGBoostSkyjoAI(BaseAI):
    """
    XGBoost AI pour Skyjo avec features engineering avancé
    - Features repensées pour capturer les patterns stratégiques
    - Modèles spécialisés par type de décision
    - Entraînement sur données optimales et adversariales
    """
    
    def __init__(self):
        # Modèles XGBoost pour chaque type de décision
        self.source_model = None
        self.keep_model = None
        self.position_model = None
        self.reveal_model = None
        
        # Scalers pour normalisation
        self.source_scaler = StandardScaler()
        self.keep_scaler = StandardScaler()
        self.position_scaler = StandardScaler()
        self.reveal_scaler = StandardScaler()
        
        # Données d'entraînement
        self.training_data = {
            'source': {'X': [], 'y': []},
            'keep': {'X': [], 'y': []},
            'position': {'X': [], 'y': []},
            'reveal': {'X': [], 'y': []}
        }
        
        # Configuration XGBoost optimisée
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist',
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }
        
        # Métriques de performance
        self.decision_count = 0
        self.successful_predictions = 0
        
        # Dimensions attendues pour les features
        self.expected_feature_dims = {
            'source': 60,
            'keep': 65,
            'position': 70,
            'reveal': 65
        }
        
    def safe_get_card_value(self, card):
        """Récupère la valeur d'une carte de manière sécurisée"""
        try:
            if card is not None and hasattr(card, 'value'):
                return card.value
            return 0
        except:
            return 0
    
    def safe_is_revealed(self, card):
        """Vérifie si une carte est révélée de manière sécurisée"""
        try:
            if card is not None and hasattr(card, 'revealed'):
                return card.revealed
            return False
        except:
            return False
    
    def extract_basic_grid_features(self, grid):
        """Extrait les features de base de notre grille"""
        try:
            revealed_values = []
            hidden_count = 0
            total_score = 0
            
            if grid:
                for i in range(min(GRID_ROWS, len(grid))):
                    for j in range(min(GRID_COLS, len(grid[i]) if i < len(grid) else 0)):
                        if i < len(grid) and j < len(grid[i]) and grid[i][j] is not None:
                            card = grid[i][j]
                            value = self.safe_get_card_value(card)
                            revealed = self.safe_is_revealed(card)
                            
                            if revealed:
                                revealed_values.append(value)
                                total_score += value
                            else:
                                hidden_count += 1
            
            # Features statistiques
            features = [
                len(revealed_values),  # Nombre de cartes révélées
                hidden_count,  # Nombre de cartes cachées
                total_score,  # Score révélé actuel
                np.mean(revealed_values) if revealed_values else 0,  # Score moyen révélé
                np.std(revealed_values) if len(revealed_values) > 1 else 0,  # Écart-type des scores
                min(revealed_values) if revealed_values else 0,  # Meilleure carte révélée
                max(revealed_values) if revealed_values else 0,  # Pire carte révélée
                sum(1 for v in revealed_values if v <= 0),  # Nombre de cartes négatives
                sum(1 for v in revealed_values if v <= 3),  # Nombre de bonnes cartes
                sum(1 for v in revealed_values if v >= 8),  # Nombre de mauvaises cartes
                total_score / max(1, len(revealed_values)),  # Score normalisé par cartes révélées
                (12 - len(revealed_values)) * 3  # Score potentiel des cartes cachées (conservateur)
            ]
            
            return features
            
        except Exception as e:
            # Fallback avec features par défaut
            return [0.0] * 12
    
    def extract_column_features(self, grid):
        """Extrait les features des colonnes"""
        try:
            column_features = []
            
            for col in range(GRID_COLS):
                col_values = []
                col_revealed = 0
                col_hidden = 0
                
                if grid:
                    for row in range(min(GRID_ROWS, len(grid))):
                        if (row < len(grid) and col < len(grid[row]) and 
                            grid[row][col] is not None):
                            card = grid[row][col]
                            value = self.safe_get_card_value(card)
                            revealed = self.safe_is_revealed(card)
                            
                            if revealed:
                                col_values.append(value)
                                col_revealed += 1
                            else:
                                col_hidden += 1
                
                # Potentiel de la colonne
                if col_values:
                    if len(set(col_values)) == 1:  # Colonne identique
                        column_potential = 1.0 - (col_values[0] / 12.0)
                    else:
                        column_potential = 1.0 - (np.mean(col_values) / 12.0)
                else:
                    column_potential = 0.5
                
                column_features.extend([
                    col_revealed,  # Cartes révélées dans la colonne
                    col_hidden,   # Cartes cachées dans la colonne
                    column_potential  # Potentiel de la colonne
                ])
            
            return column_features
            
        except Exception as e:
            # Fallback : 4 colonnes × 3 features = 12 features
            return [0.0] * 12
    
    def extract_opponent_features(self, other_grids):
        """Extrait les features des adversaires"""
        try:
            opp_features = []
            
            for i in range(3):  # Maximum 3 adversaires
                if i < len(other_grids) and other_grids[i]:
                    opp_revealed = []
                    opp_total = 0
                    
                    grid = other_grids[i]
                    for row_idx in range(min(GRID_ROWS, len(grid))):
                        for col_idx in range(min(GRID_COLS, len(grid[row_idx]) if row_idx < len(grid) else 0)):
                            if (row_idx < len(grid) and col_idx < len(grid[row_idx]) and 
                                grid[row_idx][col_idx] is not None):
                                card = grid[row_idx][col_idx]
                                value = self.safe_get_card_value(card)
                                revealed = self.safe_is_revealed(card)
                                
                                if revealed:
                                    opp_revealed.append(value)
                                    opp_total += value
                    
                    # Features adversaire
                    opp_features.extend([
                        len(opp_revealed),  # Cartes révélées de l'adversaire
                        opp_total,  # Score révélé de l'adversaire
                        np.mean(opp_revealed) if opp_revealed else 0,  # Score moyen adversaire
                        min(opp_revealed) if opp_revealed else 0,  # Meilleure carte adversaire
                        max(opp_revealed) if opp_revealed else 0   # Pire carte adversaire
                    ])
                else:
                    opp_features.extend([0, 0, 0, 0, 0])  # Adversaire absent
            
            return opp_features
            
        except Exception as e:
            # Fallback : 3 adversaires × 5 features = 15 features
            return [0.0] * 15
    
    def extract_discard_features(self, discard):
        """Extrait les features de la défausse"""
        try:
            if not discard:
                return [0.0] * 10
            
            recent_discard = []
            for i in range(min(10, len(discard))):
                card = discard[-(i+1)]
                value = self.safe_get_card_value(card)
                recent_discard.append(value)
            
            discard_features = [
                len(discard),  # Taille de la défausse
                recent_discard[0] if recent_discard else 0,  # Dernière carte défausse
                np.mean(recent_discard) if recent_discard else 0,  # Moyenne récente
                np.std(recent_discard) if len(recent_discard) > 1 else 0,  # Variabilité
                sum(1 for v in recent_discard if v <= 3),  # Bonnes cartes récentes
                sum(1 for v in recent_discard if v >= 8),  # Mauvaises cartes récentes
                min(recent_discard) if recent_discard else 0,  # Meilleure récente
                max(recent_discard) if recent_discard else 0,  # Pire récente
                sum(recent_discard) if recent_discard else 0,  # Somme récente
                len(set(recent_discard)) if recent_discard else 0  # Diversité récente
            ]
            
            return discard_features
            
        except Exception as e:
            return [0.0] * 10
    
    def extract_context_features(self, grid, other_grids, discard):
        """Extrait les features contextuelles"""
        try:
            # Phase de jeu
            total_revealed_all = 0
            total_cards_all = 0
            
            all_grids = [grid] + other_grids
            for g in all_grids:
                if g:
                    for row_idx in range(min(GRID_ROWS, len(g))):
                        for col_idx in range(min(GRID_COLS, len(g[row_idx]) if row_idx < len(g) else 0)):
                            if (row_idx < len(g) and col_idx < len(g[row_idx]) and 
                                g[row_idx][col_idx] is not None):
                                total_cards_all += 1
                                if self.safe_is_revealed(g[row_idx][col_idx]):
                                    total_revealed_all += 1
            
            game_progress = total_revealed_all / max(1, total_cards_all)
            
            # Estimation des cartes restantes dans le deck
            all_revealed_values = []
            
            # Collecter toutes les cartes révélées
            for g in all_grids:
                if g:
                    for row in g:
                        if row:
                            for card in row:
                                if card is not None and self.safe_is_revealed(card):
                                    all_revealed_values.append(self.safe_get_card_value(card))
            
            if discard:
                for card in discard:
                    all_revealed_values.append(self.safe_get_card_value(card))
            
            # Estimation probabiliste du deck restant
            deck_composition = Counter()
            deck_composition.update([-2] * 5)
            deck_composition.update([-1] * 10)
            deck_composition.update([0] * 15)
            for value in range(1, 13):
                deck_composition.update([value] * 10)
            
            for value in all_revealed_values:
                if deck_composition[value] > 0:
                    deck_composition[value] -= 1
            
            remaining_total = sum(deck_composition.values())
            expected_deck_value = sum(value * count for value, count in deck_composition.items()) / max(1, remaining_total)
            
            contextual_features = [
                game_progress,  # Progression du jeu (0-1)
                1 if game_progress < 0.3 else 0,  # Flag début de partie
                1 if 0.3 <= game_progress < 0.7 else 0,  # Flag milieu de partie
                1 if game_progress >= 0.7 else 0,  # Flag fin de partie
                expected_deck_value,  # Valeur attendue du deck restant
                remaining_total,  # Cartes restantes dans le deck
                len(all_revealed_values),  # Total cartes révélées (toutes grilles)
                sum(all_revealed_values) / max(1, len(all_revealed_values)),  # Score global moyen
                sum(1 for v in all_revealed_values if v <= 0),  # Cartes négatives globales
                sum(1 for v in all_revealed_values if v >= 8)   # Cartes élevées globales
            ]
            
            return contextual_features
            
        except Exception as e:
            return [0.0] * 10
    
    def extract_card_features(self, current_card, expected_deck_value, revealed_values):
        """Extrait les features pour une carte spécifique"""
        try:
            if current_card is None:
                return [0.0] * 6
            
            card_value = self.safe_get_card_value(current_card)
            
            card_features = [
                card_value,  # Valeur de la carte
                1 if card_value <= 0 else 0,  # Flag carte négative
                1 if card_value <= 3 else 0,  # Flag bonne carte
                1 if card_value >= 8 else 0,  # Flag mauvaise carte
                card_value - expected_deck_value,  # Différence vs moyenne deck
                1 if (revealed_values and card_value < np.mean(revealed_values)) else 0  # Meilleure que moyenne
            ]
            
            return card_features
            
        except Exception as e:
            return [0.0] * 6
    
    def extract_position_features(self, position, grid):
        """Extrait les features pour une position spécifique"""
        try:
            if position is None or not grid:
                return [0.0] * 5
            
            row, col = position
            if (row >= len(grid) or col >= len(grid[row]) or 
                grid[row][col] is None):
                return [0.0] * 5
            
            current_card_at_pos = grid[row][col]
            current_value = self.safe_get_card_value(current_card_at_pos)
            
            position_features = [
                current_value,  # Carte actuelle à la position
                1 if (row == 0 or row == GRID_ROWS-1) and (col == 0 or col == GRID_COLS-1) else 0,  # Flag coin
                1 if row == 0 or row == GRID_ROWS-1 or col == 0 or col == GRID_COLS-1 else 0,  # Flag bord
                row,  # Position ligne
                col   # Position colonne
            ]
            
            return position_features
            
        except Exception as e:
            return [0.0] * 5
    
    def extract_advanced_features(self, grid, discard, other_grids, current_card=None, position=None):
        """Extraction de features avancées pour XGBoost"""
        try:
            features = []
            
            # Features de base (12)
            basic_features = self.extract_basic_grid_features(grid)
            features.extend(basic_features)
            
            # Features de colonnes (12)
            column_features = self.extract_column_features(grid)
            features.extend(column_features)
            
            # Features adversaires (15)
            opponent_features = self.extract_opponent_features(other_grids)
            features.extend(opponent_features)
            
            # Features défausse (10)
            discard_features = self.extract_discard_features(discard)
            features.extend(discard_features)
            
            # Features contextuelles (10)
            context_features = self.extract_context_features(grid, other_grids, discard)
            features.extend(context_features)
            
            # Récupérer les valeurs révélées pour les features de carte
            revealed_values = []
            expected_deck_value = context_features[4] if len(context_features) > 4 else 3.0
            
            if grid:
                for i in range(min(GRID_ROWS, len(grid))):
                    for j in range(min(GRID_COLS, len(grid[i]) if i < len(grid) else 0)):
                        if (i < len(grid) and j < len(grid[i]) and grid[i][j] is not None and 
                            self.safe_is_revealed(grid[i][j])):
                            revealed_values.append(self.safe_get_card_value(grid[i][j]))
            
            # Features pour carte actuelle (6)
            card_features = self.extract_card_features(current_card, expected_deck_value, revealed_values)
            features.extend(card_features)
            
            # Features pour position (5)
            position_features = self.extract_position_features(position, grid)
            features.extend(position_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            # Fallback avec features par défaut
            total_features = 12 + 12 + 15 + 10 + 10 + 6 + 5  # 70 features
            return np.array([0.0] * total_features, dtype=np.float32)
    
    def initial_flip(self):
        """Flip initial stratégique"""
        # Coins pour contrôle territorial
        corners = [[0, 0], [0, GRID_COLS-1], [GRID_ROWS-1, 0], [GRID_ROWS-1, GRID_COLS-1]]
        return random.sample(corners, 2)
    
    def choose_source(self, grid, discard, other_grids):
        """Choix de source avec XGBoost"""
        if not discard:
            return 'P'
        
        if self.source_model is None:
            # Fallback à une heuristique simple
            discard_value = self.safe_get_card_value(discard[-1])
            return 'D' if discard_value <= 4 else 'P'
        
        try:
            # Extraire les features
            features = self.extract_advanced_features(grid, discard, other_grids)
            
            # Ajuster la dimension si nécessaire
            expected_dim = self.expected_feature_dims['source']
            if len(features) != expected_dim:
                if len(features) > expected_dim:
                    features = features[:expected_dim]
                else:
                    features = np.pad(features, (0, expected_dim - len(features)), 'constant')
            
            features_scaled = self.source_scaler.transform([features])
            
            # Prédiction XGBoost
            prob_take_discard = self.source_model.predict_proba(features_scaled)[0][1]
            
            # Décision avec un peu d'exploration
            exploration_rate = 0.1
            if random.random() < exploration_rate:
                choice = random.choice(['D', 'P'])
            else:
                choice = 'D' if prob_take_discard > 0.5 else 'P'
            
            self.decision_count += 1
            return choice
            
        except Exception as e:
            # Fallback en cas d'erreur
            discard_value = self.safe_get_card_value(discard[-1])
            return 'D' if discard_value <= 4 else 'P'
    
    def choose_keep(self, card, grid, other_grids):
        """Décision de garde avec XGBoost"""
        if self.keep_model is None:
            # Fallback heuristique
            card_value = self.safe_get_card_value(card)
            return card_value <= 3
        
        try:
            # Extraire les features
            features = self.extract_advanced_features(grid, [], other_grids, card)
            
            # Ajuster la dimension si nécessaire
            expected_dim = self.expected_feature_dims['keep']
            if len(features) != expected_dim:
                if len(features) > expected_dim:
                    features = features[:expected_dim]
                else:
                    features = np.pad(features, (0, expected_dim - len(features)), 'constant')
            
            features_scaled = self.keep_scaler.transform([features])
            
            # Prédiction XGBoost
            prob_keep = self.keep_model.predict_proba(features_scaled)[0][1]
            
            # Décision avec exploration
            exploration_rate = 0.1
            if random.random() < exploration_rate:
                decision = random.choice([True, False])
            else:
                decision = prob_keep > 0.5
            
            self.decision_count += 1
            return decision
            
        except Exception as e:
            # Fallback heuristique
            card_value = self.safe_get_card_value(card)
            return card_value <= 3
    
    def choose_position(self, card, grid, other_grids):
        """Choix de position avec XGBoost"""
        if not grid:
            return None
        
        # Positions disponibles
        available_positions = []
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                try:
                    if (i < len(grid) and j < len(grid[i]) and 
                        grid[i][j] is not None and self.safe_is_revealed(grid[i][j])):
                        available_positions.append((i, j))
                except:
                    continue
        
        if not available_positions:
            return None
        
        if self.position_model is None:
            # Fallback : chercher la meilleure amélioration
            best_position = None
            best_improvement = -float('inf')
            
            card_value = self.safe_get_card_value(card)
            for row, col in available_positions:
                try:
                    current_value = self.safe_get_card_value(grid[row][col])
                    improvement = current_value - card_value
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_position = (row, col)
                except:
                    continue
            
            return best_position or random.choice(available_positions)
        
        try:
            # Évaluer chaque position avec XGBoost
            position_scores = []
            
            for row, col in available_positions:
                try:
                    features = self.extract_advanced_features(grid, [], other_grids, card, (row, col))
                    
                    # Ajuster la dimension si nécessaire
                    expected_dim = self.expected_feature_dims['position']
                    if len(features) != expected_dim:
                        if len(features) > expected_dim:
                            features = features[:expected_dim]
                        else:
                            features = np.pad(features, (0, expected_dim - len(features)), 'constant')
                    
                    features_scaled = self.position_scaler.transform([features])
                    
                    # Prédiction de la "qualité" de cette position
                    position_quality = self.position_model.predict_proba(features_scaled)[0][1]
                    position_scores.append((position_quality, (row, col)))
                except:
                    # En cas d'erreur, score neutre
                    position_scores.append((0.5, (row, col)))
            
            # Choisir la meilleure position avec un peu d'exploration
            position_scores.sort(reverse=True)
            
            exploration_rate = 0.15
            if random.random() < exploration_rate:
                # Exploration : choisir parmi les 3 meilleures
                top_positions = position_scores[:min(3, len(position_scores))]
                chosen = random.choice(top_positions)[1]
            else:
                # Exploitation : meilleure position
                chosen = position_scores[0][1]
            
            self.decision_count += 1
            return chosen
            
        except Exception as e:
            # Fallback
            return random.choice(available_positions)
    
    def choose_reveal(self, grid):
        """Choix de révélation avec XGBoost"""
        if not grid:
            return None
        
        # Positions non révélées
        unrevealed_positions = []
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                try:
                    if (i < len(grid) and j < len(grid[i]) and 
                        grid[i][j] is not None and not self.safe_is_revealed(grid[i][j])):
                        unrevealed_positions.append((i, j))
                except:
                    continue
        
        if not unrevealed_positions:
            return None
        
        if self.reveal_model is None:
            # Fallback : stratégie de colonnes
            column_scores = {}
            for col in range(GRID_COLS):
                revealed_count = 0
                for row in range(GRID_ROWS):
                    try:
                        if (row < len(grid) and col < len(grid[row]) and 
                            grid[row][col] is not None and self.safe_is_revealed(grid[row][col])):
                            revealed_count += 1
                    except:
                        continue
                
                column_scores[col] = revealed_count
            
            # Privilégier les colonnes avec plus de cartes révélées
            best_positions = []
            max_revealed = max(column_scores.values()) if column_scores else 0
            
            for row, col in unrevealed_positions:
                if column_scores.get(col, 0) == max_revealed:
                    best_positions.append((row, col))
            
            return random.choice(best_positions) if best_positions else random.choice(unrevealed_positions)
        
        try:
            # Évaluer chaque position avec XGBoost
            reveal_scores = []
            
            for row, col in unrevealed_positions:
                try:
                    features = self.extract_advanced_features(grid, [], [], position=(row, col))
                    
                    # Ajuster la dimension si nécessaire
                    expected_dim = self.expected_feature_dims['reveal']
                    if len(features) != expected_dim:
                        if len(features) > expected_dim:
                            features = features[:expected_dim]
                        else:
                            features = np.pad(features, (0, expected_dim - len(features)), 'constant')
                    
                    features_scaled = self.reveal_scaler.transform([features])
                    
                    # Prédiction de l'utilité de révéler cette position
                    reveal_utility = self.reveal_model.predict_proba(features_scaled)[0][1]
                    reveal_scores.append((reveal_utility, (row, col)))
                except:
                    # En cas d'erreur, score neutre
                    reveal_scores.append((0.5, (row, col)))
            
            # Choisir la meilleure position
            reveal_scores.sort(reverse=True)
            
            exploration_rate = 0.2
            if random.random() < exploration_rate:
                top_positions = reveal_scores[:min(3, len(reveal_scores))]
                chosen = random.choice(top_positions)[1]
            else:
                chosen = reveal_scores[0][1]
            
            self.decision_count += 1
            return chosen
            
        except Exception as e:
            # Fallback
            return random.choice(unrevealed_positions)
    
    def collect_training_data(self, num_games=1000):
        """Collecte des données d'entraînement en observant InitialAI"""
        print(f"📊 Collecte de données d'entraînement sur {num_games} parties...")
        
        from ai.initial import InitialAI
        from core.game import SkyjoGame, Scoreboard
        from core.player import Player
        
        # Réinitialiser les données
        for key in self.training_data:
            self.training_data[key]['X'].clear()
            self.training_data[key]['y'].clear()
        
        collected_samples = {'source': 0, 'keep': 0, 'position': 0, 'reveal': 0}
        
        for game_num in range(num_games):
            try:
                # Créer une partie avec InitialAI comme référence
                reference_ai = InitialAI()
                players = [
                    Player(0, "Reference", reference_ai),
                    Player(1, "Opp1", InitialAI()),
                    Player(2, "Opp2", InitialAI()),
                    Player(3, "Opp3", InitialAI())
                ]
                
                scoreboard = Scoreboard(players)
                game = SkyjoGame(players, scoreboard)
                
                # Observer les décisions pendant la partie
                step_count = 0
                while not game.finished and step_count < 1000:
                    if not game.round_over and game_num < num_games * 0.7:  # Seulement les premiers 70% pour l'entraînement
                        current_player = game.players[game.current_player_index]
                        
                        if current_player.name == "Reference":
                            other_grids = [p.grid for i, p in enumerate(game.players) if i != game.current_player_index]
                            
                            # Observer la décision de source
                            if game.discard and len(game.discard) > 0:
                                try:
                                    features = self.extract_advanced_features(current_player.grid, game.discard, other_grids)
                                    decision = reference_ai.choose_source(current_player.grid, game.discard, other_grids)
                                    label = 1 if decision == 'D' else 0
                                    
                                    # Ajuster les dimensions pour source
                                    expected_dim = self.expected_feature_dims['source']
                                    if len(features) != expected_dim:
                                        if len(features) > expected_dim:
                                            features = features[:expected_dim]
                                        else:
                                            features = np.pad(features, (0, expected_dim - len(features)), 'constant')
                                    
                                    self.training_data['source']['X'].append(features)
                                    self.training_data['source']['y'].append(label)
                                    collected_samples['source'] += 1
                                except Exception as e:
                                    continue
                    
                    game.step()
                    step_count += 1
                
                # Affichage de progression
                if game_num % 200 == 0:
                    print(f"   Collecté {game_num}/{num_games} parties...")
                    print(f"   Échantillons: Source={collected_samples['source']}, Keep={collected_samples['keep']}, "
                          f"Position={collected_samples['position']}, Reveal={collected_samples['reveal']}")
                    
            except Exception as e:
                continue
        
        print(f"✅ Collecte terminée!")
        print(f"   Source: {len(self.training_data['source']['X'])} échantillons")
        print(f"   Keep: {len(self.training_data['keep']['X'])} échantillons")
        print(f"   Position: {len(self.training_data['position']['X'])} échantillons")
        print(f"   Reveal: {len(self.training_data['reveal']['X'])} échantillons")
    
    def train_xgboost_models(self):
        """Entraîne les modèles XGBoost"""
        print("🚀 Entraînement des modèles XGBoost...")
        
        models_trained = 0
        
        for decision_type in ['source']:  # Commencer par source seulement
            X = self.training_data[decision_type]['X']
            y = self.training_data[decision_type]['y']
            
            if len(X) < 50:  # Minimum de données requis
                print(f"⚠️ Pas assez de données pour {decision_type} ({len(X)} échantillons)")
                continue
            
            try:
                # Conversion en numpy arrays
                X = np.array(X)
                y = np.array(y)
                
                print(f"📊 {decision_type}: {X.shape[0]} échantillons, {X.shape[1]} features")
                
                # Vérifier l'équilibre des classes
                unique, counts = np.unique(y, return_counts=True)
                print(f"   Classes: {dict(zip(unique, counts))}")
                
                # Split train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Normalisation
                scaler = getattr(self, f"{decision_type}_scaler")
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Entraînement XGBoost
                model = xgb.XGBClassifier(**self.xgb_params)
                model.fit(X_train_scaled, y_train)
                
                # Évaluation
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                print(f"✅ {decision_type.capitalize()}: Train={train_score:.3f}, Test={test_score:.3f}")
                
                # Sauvegarder le modèle
                setattr(self, f"{decision_type}_model", model)
                models_trained += 1
                
            except Exception as e:
                print(f"❌ Erreur pour {decision_type}: {e}")
        
        if models_trained > 0:
            self.save_models()
            print(f"🎯 {models_trained}/4 modèles entraînés avec succès!")
        else:
            print("❌ Aucun modèle n'a pu être entraîné")
    
    def save_models(self):
        """Sauvegarde les modèles XGBoost"""
        try:
            os.makedirs("ml_models", exist_ok=True)
            
            model_data = {
                'source_model': self.source_model,
                'keep_model': self.keep_model,
                'position_model': self.position_model,
                'reveal_model': self.reveal_model,
                'source_scaler': self.source_scaler,
                'keep_scaler': self.keep_scaler,
                'position_scaler': self.position_scaler,
                'reveal_scaler': self.reveal_scaler,
                'xgb_params': self.xgb_params,
                'expected_feature_dims': self.expected_feature_dims
            }
            
            with open("ml_models/xgboost_skyjo.pkl", 'wb') as f:
                pickle.dump(model_data, f)
            
            print("💾 Modèles XGBoost sauvegardés!")
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde XGBoost: {e}")
    
    def load_models(self):
        """Charge les modèles XGBoost sauvegardés"""
        model_path = "ml_models/xgboost_skyjo.pkl"
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.source_model = model_data.get('source_model')
                self.keep_model = model_data.get('keep_model')
                self.position_model = model_data.get('position_model')
                self.reveal_model = model_data.get('reveal_model')
                self.source_scaler = model_data.get('source_scaler', StandardScaler())
                self.keep_scaler = model_data.get('keep_scaler', StandardScaler())
                self.position_scaler = model_data.get('position_scaler', StandardScaler())
                self.reveal_scaler = model_data.get('reveal_scaler', StandardScaler())
                self.xgb_params = model_data.get('xgb_params', self.xgb_params)
                self.expected_feature_dims = model_data.get('expected_feature_dims', self.expected_feature_dims)
                
                models_loaded = sum(1 for model in [self.source_model, self.keep_model, 
                                                  self.position_model, self.reveal_model] if model is not None)
                
                print(f"📂 Modèles XGBoost chargés! ({models_loaded}/4 modèles)")
                return True
                
            except Exception as e:
                print(f"❌ Erreur lors du chargement XGBoost: {e}")
                return False
        return False 