import numpy as np
import pandas as pd
import random
import os
import pickle
from collections import Counter, defaultdict
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class XGBoostEnhancedAI(BaseAI):
    """
    XGBoost Enhanced AI pour Skyjo - Version optimisée:
    - Feature engineering avancé avec patterns temporels
    - Ensemble de modèles avec voting
    - Adaptation dynamique aux adversaires
    - Apprentissage en ligne avec feedback
    """
    
    def __init__(self):
        # Modèles XGBoost spécialisés avec ensemble
        self.source_ensemble = None
        self.keep_ensemble = None
        self.position_ensemble = None
        self.reveal_ensemble = None
        
        # Scalers robustes 
        self.source_scaler = RobustScaler()
        self.keep_scaler = RobustScaler()
        self.position_scaler = RobustScaler()
        self.reveal_scaler = RobustScaler()
        
        # Données d'entraînement enrichies
        self.training_data = {
            'source': {'X': [], 'y': [], 'weights': []},
            'keep': {'X': [], 'y': [], 'weights': []},
            'position': {'X': [], 'y': [], 'weights': []},
            'reveal': {'X': [], 'y': [], 'weights': []}
        }
        
        # Configuration XGBoost optimisée pour ensemble
        self.xgb_base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 5,
            'learning_rate': 0.08,
            'n_estimators': 150,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': 42,
            'tree_method': 'hist',
            'reg_alpha': 0.05,
            'reg_lambda': 0.8,
            'min_child_weight': 3
        }
        
        # Variantes XGBoost pour l'ensemble
        self.xgb_variants = [
            {**self.xgb_base_params, 'max_depth': 4, 'learning_rate': 0.1},
            {**self.xgb_base_params, 'max_depth': 6, 'learning_rate': 0.06},
            {**self.xgb_base_params, 'n_estimators': 200, 'subsample': 0.9}
        ]
        
        # Variantes pour ensemble
        self.xgb_variants = [
            {**self.xgb_base_params, 'max_depth': 4, 'learning_rate': 0.12},
            {**self.xgb_base_params, 'max_depth': 6, 'learning_rate': 0.06},
            {**self.xgb_base_params, 'subsample': 0.9, 'colsample_bytree': 0.9}
        ]
        
        # Métriques et adaptation
        self.decision_history = []
        self.opponent_patterns = defaultdict(list)
        self.game_phase_performance = defaultdict(float)
        self.adaptation_factor = 0.1
        
        # Dimensions des features étendues
        self.expected_feature_dims = {
            'source': 85,
            'keep': 90, 
            'position': 95,
            'reveal': 80
        }
        
        # Historique pour patterns temporels
        self.state_history = []
        self.max_history = 20
        
        self.load_models()
    
    def safe_get_card_value(self, card):
        """Récupère la valeur d'une carte de manière sécurisée"""
        try:
            if card is not None and hasattr(card, 'value'):
                return card.value
            return 0
        except:
            print(f"Erreur lors de la récupération de la valeur de la carte: {card}")
            return 0
    
    def safe_is_revealed(self, card):
        """Vérifie si une carte est révélée de manière sécurisée"""
        try:
            if card is not None and hasattr(card, 'revealed'):
                return card.revealed
            return False
        except:
            print(f"Erreur lors de la vérification de la révélation de la carte: {card}")
            return False
    
    def extract_enhanced_grid_features(self, grid):
        """Extrait des features avancées de la grille"""
        try:
            revealed_values = []
            hidden_positions = []
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
                                hidden_positions.append((i, j))
            
            # Features statistiques de base
            num_revealed = len(revealed_values)
            num_hidden = len(hidden_positions)
            
            basic_features = [
                num_revealed,  # Nombre de cartes révélées
                num_hidden,    # Nombre de cartes cachées
                total_score,   # Score révélé actuel
                np.mean(revealed_values) if revealed_values else 0,
                np.std(revealed_values) if len(revealed_values) > 1 else 0,
                np.median(revealed_values) if revealed_values else 0,
                min(revealed_values) if revealed_values else 0,
                max(revealed_values) if revealed_values else 0,
                sum(1 for v in revealed_values if v <= 0),   # Cartes négatives
                sum(1 for v in revealed_values if v <= 2),   # Cartes excellentes
                sum(1 for v in revealed_values if v <= 5),   # Cartes bonnes
                sum(1 for v in revealed_values if v >= 8),   # Cartes mauvaises
                sum(1 for v in revealed_values if v >= 10),  # Cartes très mauvaises
            ]
            
            # Features avancées
            if revealed_values:
                # Distribution des valeurs
                value_distribution = [sum(1 for v in revealed_values if i <= v < i+3) for i in range(-2, 12, 3)]
                # Momentum (tendance récente)
                if len(revealed_values) > 3:
                    recent_trend = np.mean(revealed_values[-3:]) - np.mean(revealed_values[:-3])
                else:
                    recent_trend = 0
                
                advanced_features = value_distribution + [
                    recent_trend,
                    num_revealed / 12.0,  # Progression révélée normalisée  
                    total_score / max(1, num_revealed),  # Score moyen par carte révélée
                    len(set(revealed_values)) / max(1, len(revealed_values)),  # Diversité
                ]
            else:
                advanced_features = [0.0] * 9
            
            return basic_features + advanced_features
            
        except Exception as e:
            return [0.0] * 22
    
    def extract_advanced_column_features(self, grid):
        """Features avancées des colonnes avec patterns"""
        try:
            column_features = []
            
            for col in range(GRID_COLS):
                col_values = []
                col_revealed = 0
                col_pattern_score = 0
                
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
                
                # Analyse des patterns de colonne
                if col_values:
                    # Pattern identique
                    if len(set(col_values)) == 1:
                        pattern_bonus = 2.0
                    # Pattern croissant/décroissant
                    elif len(col_values) > 1:
                        is_sorted = all(col_values[i] <= col_values[i+1] for i in range(len(col_values)-1))
                        is_reverse_sorted = all(col_values[i] >= col_values[i+1] for i in range(len(col_values)-1))
                        pattern_bonus = 1.5 if (is_sorted or is_reverse_sorted) else 1.0
                    else:
                        pattern_bonus = 1.0
                    
                    col_potential = (1.0 - np.mean(col_values) / 12.0) * pattern_bonus
                    col_variance = np.var(col_values)
                    col_completion = col_revealed / GRID_ROWS
                else:
                    col_potential = 0.5
                    col_variance = 0
                    col_completion = 0
                
                column_features.extend([
                    col_revealed,      # Cartes révélées
                    col_completion,    # Taux de complétion
                    col_potential,     # Potentiel de la colonne  
                    col_variance,      # Variance des valeurs
                    pattern_bonus if col_values else 0  # Bonus de pattern
                ])
            
            return column_features
            
        except Exception as e:
            return [0.0] * (GRID_COLS * 5)
    
    def extract_opponent_intelligence_features(self, other_grids):
        """Features intelligentes des adversaires"""
        try:
            opp_features = []
            
            for i in range(3):  # Maximum 3 adversaires
                if i < len(other_grids) and other_grids[i]:
                    grid = other_grids[i]
                    opp_revealed = []
                    opp_patterns = {}
                    
                    # Collecte des données adversaire
                    for row_idx in range(min(GRID_ROWS, len(grid))):
                        for col_idx in range(min(GRID_COLS, len(grid[row_idx]) if row_idx < len(grid) else 0)):
                            if (row_idx < len(grid) and col_idx < len(grid[row_idx]) and 
                                grid[row_idx][col_idx] is not None):
                                card = grid[row_idx][col_idx]
                                value = self.safe_get_card_value(card)
                                revealed = self.safe_is_revealed(card)
                                
                                if revealed:
                                    opp_revealed.append(value)
                    
                    # Features adversaire
                    if opp_revealed:
                        opp_score = sum(opp_revealed)
                        opp_mean = np.mean(opp_revealed)
                        opp_risk = sum(1 for v in opp_revealed if v >= 7) / len(opp_revealed)
                        opp_advantage = sum(1 for v in opp_revealed if v <= 3) / len(opp_revealed)
                        opp_progress = len(opp_revealed) / 12.0
                        
                        # Estimation du potentiel de l'adversaire
                        estimated_final = opp_score + (12 - len(opp_revealed)) * 3.5  # Estimation conservatrice
                        threat_level = max(0, (25 - estimated_final) / 25.0)  # Niveau de menace
                        
                        opp_features.extend([
                            len(opp_revealed),
                            opp_score,
                            opp_mean,
                            opp_risk,
                            opp_advantage,
                            opp_progress,
                            estimated_final,
                            threat_level
                        ])
                    else:
                        opp_features.extend([0.0] * 8)
                else:
                    opp_features.extend([0.0] * 8)
            
            return opp_features
            
        except Exception as e:
            return [0.0] * 24
    
    def extract_temporal_features(self):
        """Features basées sur l'historique temporel"""
        try:
            if len(self.state_history) < 2:
                return [0.0] * 8
            
            # Analyser la tendance des scores
            recent_states = self.state_history[-5:]  # 5 derniers états
            
            score_trend = 0
            reveal_trend = 0
            
            if len(recent_states) >= 2:
                scores = [state.get('score', 0) for state in recent_states]
                reveals = [state.get('reveals', 0) for state in recent_states]
                
                if len(scores) > 1:
                    score_trend = scores[-1] - scores[0]
                if len(reveals) > 1:
                    reveal_trend = reveals[-1] - reveals[0]
            
            # Patterns de décision récents
            recent_decisions = [state.get('last_decision', 'none') for state in recent_states]
            decision_consistency = len(set(recent_decisions)) / max(1, len(recent_decisions))
            
            # Vitesse de jeu
            game_speed = len(self.state_history) / max(1, self.state_history[-1].get('turn', 1))
            
            return [
                score_trend,
                reveal_trend, 
                decision_consistency,
                game_speed,
                len(self.state_history),  # Nombre total de tours
                len(recent_states),       # Historique récent disponible
                0,  # Reserved pour future feature
                0   # Reserved pour future feature
            ]
            
        except Exception as e:
            return [0.0] * 8
    
    def extract_probabilistic_features(self, revealed_cards):
        """Features probabilistes basées sur les cartes révélées"""
        try:
            # Distribution initiale du deck
            initial_deck = Counter()
            initial_deck.update([-2] * 5)
            initial_deck.update([-1] * 10) 
            initial_deck.update([0] * 15)
            for value in range(1, 13):
                initial_deck.update([value] * 10)
            
            # Soustraire les cartes révélées
            remaining_deck = initial_deck.copy()
            for card_value in revealed_cards:
                if remaining_deck[card_value] > 0:
                    remaining_deck[card_value] -= 1
            
            total_remaining = sum(remaining_deck.values())
            if total_remaining == 0:
                return [0.0] * 10
            
            # Probabilités
            probabilities = {value: count/total_remaining for value, count in remaining_deck.items()}
            
            # Features probabilistes
            prob_excellent = sum(probabilities.get(v, 0) for v in [-2, -1, 0, 1, 2])
            prob_good = sum(probabilities.get(v, 0) for v in [3, 4, 5])
            prob_neutral = sum(probabilities.get(v, 0) for v in [6, 7])
            prob_bad = sum(probabilities.get(v, 0) for v in [8, 9, 10, 11, 12])
            
            expected_value = sum(v * p for v, p in probabilities.items())
            value_variance = sum((v - expected_value)**2 * p for v, p in probabilities.items())
            
            # Entropie de la distribution
            entropy = -sum(p * np.log(p + 1e-10) for p in probabilities.values())
            
            # Asymétrie de la distribution
            remaining_low = sum(remaining_deck[v] for v in range(-2, 4))
            remaining_high = sum(remaining_deck[v] for v in range(8, 13))
            distribution_skew = (remaining_low - remaining_high) / max(1, total_remaining)
            
            return [
                prob_excellent, prob_good, prob_neutral, prob_bad,
                expected_value, value_variance, entropy, distribution_skew,
                total_remaining / 150.0,  # Fraction de deck restante
                len(set(revealed_cards)) / 15.0  # Diversité des cartes révélées
            ]
            
        except Exception as e:
            return [0.0] * 10
    
    def collect_all_revealed_cards(self, grid, other_grids, discard=None):
        """Collecte toutes les cartes révélées"""
        revealed_cards = []
        
        try:
            # Notre grille
            if grid:
                for i in range(min(GRID_ROWS, len(grid))):
                    if i < len(grid) and grid[i]:
                        for j in range(min(GRID_COLS, len(grid[i]))):
                            if j < len(grid[i]) and grid[i][j] is not None:
                                if self.safe_is_revealed(grid[i][j]):
                                    revealed_cards.append(self.safe_get_card_value(grid[i][j]))
            
            # Grilles adversaires
            for opp_grid in other_grids:
                if opp_grid:
                    for i in range(min(GRID_ROWS, len(opp_grid))):
                        if i < len(opp_grid) and opp_grid[i]:
                            for j in range(min(GRID_COLS, len(opp_grid[i]))):
                                if j < len(opp_grid[i]) and opp_grid[i][j] is not None:
                                    if self.safe_is_revealed(opp_grid[i][j]):
                                        revealed_cards.append(self.safe_get_card_value(opp_grid[i][j]))
            
            # Défausse
            if discard:
                for card in discard:
                    revealed_cards.append(self.safe_get_card_value(card))
                        
        except Exception as e:
            pass
        
        return revealed_cards
    
    def extract_comprehensive_features(self, grid, discard, other_grids, current_card=None, position=None):
        """Extraction complète des features"""
        try:
            # Features de base
            grid_features = self.extract_enhanced_grid_features(grid)
            column_features = self.extract_advanced_column_features(grid)
            opponent_features = self.extract_opponent_intelligence_features(other_grids)
            temporal_features = self.extract_temporal_features()
            
            # Features probabilistes
            revealed_cards = self.collect_all_revealed_cards(grid, other_grids, discard)
            prob_features = self.extract_probabilistic_features(revealed_cards)
            
            # Features contextuelles
            context_features = []
            
            # Défausse
            if discard and len(discard) > 0:
                discard_value = self.safe_get_card_value(discard[-1])
                discard_trend = 0
                if len(discard) > 1:
                    recent_discard = [self.safe_get_card_value(card) for card in discard[-3:]]
                    discard_trend = np.mean(recent_discard)
                
                context_features.extend([
                    discard_value / 12.0,  # Valeur normalisée
                    discard_trend / 12.0,  # Tendance de la défausse
                    len(discard) / 100.0   # Taille de la défausse normalisée
                ])
            else:
                context_features.extend([0.0, 0.0, 0.0])
            
            # Carte actuelle (si fournie)
            if current_card:
                card_value = self.safe_get_card_value(current_card)
                context_features.extend([
                    card_value / 12.0,
                    1.0 if card_value <= 2 else 0.0,  # Excellente carte
                    1.0 if card_value >= 8 else 0.0   # Mauvaise carte
                ])
            else:
                context_features.extend([0.0, 0.0, 0.0])
            
            # Position (si fournie)
            if position:
                i, j = position
                context_features.extend([
                    i / (GRID_ROWS - 1),  # Position verticale normalisée
                    j / (GRID_COLS - 1),  # Position horizontale normalisée
                    1.0 if i == 0 or i == GRID_ROWS-1 else 0.0,  # Bordure verticale
                    1.0 if j == 0 or j == GRID_COLS-1 else 0.0   # Bordure horizontale
                ])
            else:
                context_features.extend([0.0, 0.0, 0.0, 0.0])
            
            # Combiner toutes les features
            all_features = (grid_features + column_features + opponent_features + 
                          temporal_features + prob_features + context_features)
            
            return np.array(all_features, dtype=np.float32)
            
        except Exception as e:
            # Fallback avec dimension appropriée
            return np.zeros(85, dtype=np.float32)
    
    def initial_flip(self):
        """Choix initial optimisé"""
        return [[0, 0], [GRID_ROWS-1, GRID_COLS-1]]
    
    def choose_source(self, grid, discard, other_grids):
        """Choix de source avec modèle ensemble"""
        try:
            if not discard:
                return 'P'
            
            features = self.extract_comprehensive_features(grid, discard, other_grids)
            
            if self.source_ensemble is not None:
                # Prédiction avec ensemble
                prob_discard = self.predict_with_ensemble(features, 'source')
                
                # Adaptation dynamique basée sur l'historique
                if len(self.decision_history) > 10:
                    recent_success = np.mean([d.get('success', 0.5) for d in self.decision_history[-10:]])
                    prob_discard += (recent_success - 0.5) * self.adaptation_factor
                
                return 'D' if prob_discard > 0.5 else 'P'
            else:
                # Fallback heuristique amélioré
                discard_value = self.safe_get_card_value(discard[-1])
                if discard_value <= 2:
                    return 'D'
                elif discard_value <= 5:
                    return 'D' if random.random() > 0.3 else 'P'
                else:
                    return 'P'
                    
        except Exception as e:
            return 'P'
    
    def choose_keep(self, card, grid, other_grids):
        """Décision de garde avec modèle ensemble"""
        try:
            features = self.extract_comprehensive_features(grid, [], other_grids, current_card=card)
            
            if self.keep_ensemble is not None:
                prob_keep = self.predict_with_ensemble(features, 'keep')
                return prob_keep > 0.5
            else:
                # Fallback heuristique
                card_value = self.safe_get_card_value(card)
                return card_value <= 4
                
        except Exception as e:
            return False
    
    def choose_position(self, card, grid, other_grids):
        """Choix de position optimisé"""
        try:
            if not grid or len(grid) == 0:
                return (0, 0)
            
            best_position = (0, 0)
            best_score = float('-inf')
            
            # Évaluer toutes les positions
            for i in range(min(GRID_ROWS, len(grid))):
                if i < len(grid) and grid[i]:
                    for j in range(min(GRID_COLS, len(grid[i]))):
                        if j < len(grid[i]) and grid[i][j] is not None:
                            features = self.extract_comprehensive_features(
                                grid, [], other_grids, current_card=card, position=(i, j)
                            )
                            
                            if self.position_ensemble is not None:
                                score = self.predict_with_ensemble(features, 'position')
                            else:
                                # Fallback avec gain immédiat
                                current_value = self.safe_get_card_value(grid[i][j])
                                card_value = self.safe_get_card_value(card)
                                score = current_value - card_value
                            
                            if score > best_score:
                                best_score = score
                                best_position = (i, j)
            
            return best_position
            
        except Exception as e:
            return (0, 0) if grid and len(grid) > 0 and len(grid[0]) > 0 else None
    
    def choose_reveal(self, grid):
        """Choix de révélation optimisé"""
        try:
            if not grid or len(grid) == 0:
                return None
            
            unrevealed_positions = []
            for i in range(min(GRID_ROWS, len(grid))):
                if i < len(grid) and grid[i]:
                    for j in range(min(GRID_COLS, len(grid[i]))):
                        if j < len(grid[i]) and grid[i][j] is not None:
                            if not self.safe_is_revealed(grid[i][j]):
                                unrevealed_positions.append((i, j))
            
            if not unrevealed_positions:
                return None
            
            best_position = unrevealed_positions[0]
            best_score = float('-inf')
            
            for i, j in unrevealed_positions:
                features = self.extract_comprehensive_features(
                    grid, [], [], position=(i, j)
                )
                
                if self.reveal_ensemble is not None:
                    score = self.predict_with_ensemble(features, 'reveal')
                else:
                    # Fallback: préférer les coins et bordures
                    score = 0
                    if i == 0 or i == GRID_ROWS - 1:
                        score += 2
                    if j == 0 or j == GRID_COLS - 1:
                        score += 2
                
                if score > best_score:
                    best_score = score
                    best_position = (i, j)
            
            return best_position
            
        except Exception as e:
            return None
    
    def predict_with_ensemble(self, features, model_type):
        """Prédiction avec modèle ensemble"""
        try:
            # Assurer la bonne dimension
            expected_dim = self.expected_feature_dims.get(model_type, 85)
            if len(features) < expected_dim:
                features = np.pad(features, (0, expected_dim - len(features)), 'constant')
            elif len(features) > expected_dim:
                features = features[:expected_dim]
            
            features = features.reshape(1, -1)
            
            # Normalisation
            scaler = getattr(self, f"{model_type}_scaler")
            if hasattr(scaler, 'transform'):
                features = scaler.transform(features)
            
            # Prédiction ensemble
            ensemble = getattr(self, f"{model_type}_ensemble")
            if ensemble is not None:
                prediction = ensemble.predict_proba(features)[0][1]  # Probabilité classe positive
                return prediction
            else:
                return 0.5
                
        except Exception as e:
            return 0.5
    
    def save_models(self):
        """Sauvegarde les modèles"""
        try:
            os.makedirs("ml_models", exist_ok=True)
            
            model_data = {
                'source_ensemble': self.source_ensemble,
                'keep_ensemble': self.keep_ensemble,
                'position_ensemble': self.position_ensemble,
                'reveal_ensemble': self.reveal_ensemble,
                'source_scaler': self.source_scaler,
                'keep_scaler': self.keep_scaler,
                'position_scaler': self.position_scaler,
                'reveal_scaler': self.reveal_scaler,
                'training_data': self.training_data,
                'decision_history': self.decision_history,
                'state_history': self.state_history
            }
            
            with open("ml_models/xgboost_enhanced.pkl", "wb") as f:
                pickle.dump(model_data, f)
                
            print("✅ Modèles XGBoost Enhanced sauvegardés")
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
    
    def load_models(self):
        """Charge les modèles sauvegardés"""
        try:
            if os.path.exists("ml_models/xgboost_skyjo.pkl"):
                with open("ml_models/xgboost_skyjo.pkl", "rb") as f:
                    model_data = pickle.load(f)
                
                self.source_ensemble = model_data.get('source_ensemble')
                self.keep_ensemble = model_data.get('keep_ensemble')
                self.position_ensemble = model_data.get('position_ensemble')
                self.reveal_ensemble = model_data.get('reveal_ensemble')
                self.source_scaler = model_data.get('source_scaler', RobustScaler())
                self.keep_scaler = model_data.get('keep_scaler', RobustScaler())
                self.position_scaler = model_data.get('position_scaler', RobustScaler())
                self.reveal_scaler = model_data.get('reveal_scaler', RobustScaler())
                self.training_data = model_data.get('training_data', {
                    'source': {'X': [], 'y': [], 'weights': []},
                    'keep': {'X': [], 'y': [], 'weights': []},
                    'position': {'X': [], 'y': [], 'weights': []},
                    'reveal': {'X': [], 'y': [], 'weights': []}
                })
                self.decision_history = model_data.get('decision_history', [])
                self.state_history = model_data.get('state_history', [])
                
                print("✅ Modèles XGBoost Enhanced chargés")
            else:
                print("📝 Nouveaux modèles XGBoost Enhanced - entraînement requis")
                
        except Exception as e:
            print(f"⚠️ Erreur chargement: {e}")

    def collect_training_data(self, num_games=1000):
        """Collecte des données d'entraînement en mode sécurisé avec collecteur complet"""
        from core.game import SkyjoGame, Scoreboard  
        from core.player import Player, create_safe_grid
        from ai.initial import InitialAI
        
        print(f"📊 Collecte de données d'entraînement sur {num_games} parties (MODE SÉCURISÉ)...")
        
        # Réinitialiser les données
        for key in self.training_data:
            self.training_data[key]['X'].clear()
            self.training_data[key]['y'].clear()
        
        collected_samples = {'source': 0, 'keep': 0, 'position': 0, 'reveal': 0}
        
        # Créer un collecteur personnalisé qui intercepte toutes les décisions
        class DataCollectorAI(InitialAI):
            def __init__(self, collector):
                super().__init__()
                self.collector = collector
            
            def choose_source(self, grid, discard, other_grids):
                decision = super().choose_source(grid, discard, other_grids)
                self.collector.record_source_decision(grid, discard, other_grids, decision)
                return decision
            
            def choose_keep(self, card, grid, other_grids):
                decision = super().choose_keep(card, grid, other_grids)
                self.collector.record_keep_decision(card, grid, other_grids, decision)
                return decision
            
            def choose_position(self, card, grid, other_grids):
                position = super().choose_position(card, grid, other_grids)
                self.collector.record_position_decision(card, grid, other_grids, position)
                return position
            
            def choose_reveal(self, grid):
                position = super().choose_reveal(grid)
                self.collector.record_reveal_decision(grid, position)
                return position
        
        # Méthodes pour enregistrer les décisions
        def record_source_decision(self, grid, discard, other_grids, decision):
            try:
                if discard and len(discard) > 0:
                    features = self.extract_comprehensive_features(grid, discard, other_grids)
                    label = 1 if decision == 'D' else 0
                    
                    if len(features) > 0 and not np.all(features == 0):
                        self.training_data['source']['X'].append(features)
                        self.training_data['source']['y'].append(label)
                        collected_samples['source'] += 1
            except Exception as e:
                pass
        
        def record_keep_decision(self, card, grid, other_grids, decision):
            try:
                features = self.extract_comprehensive_features(grid, [], other_grids, current_card=card)
                label = 1 if decision else 0
                
                if len(features) > 0 and not np.all(features == 0):
                    self.training_data['keep']['X'].append(features)
                    self.training_data['keep']['y'].append(label)
                    collected_samples['keep'] += 1
            except Exception as e:
                pass
        
        def record_position_decision(self, card, grid, other_grids, position):
            try:
                if position and grid:
                    i, j = position
                    if i < len(grid) and j < len(grid[i]) and grid[i][j] is not None:
                        current_value = self.safe_get_card_value(grid[i][j])
                        card_value = self.safe_get_card_value(card)
                        improvement = current_value - card_value
                        
                        label = 1 if improvement > 0 else 0
                        features = self.extract_comprehensive_features(grid, [], other_grids, current_card=card, position=position)
                        
                        if len(features) > 0 and not np.all(features == 0):
                            self.training_data['position']['X'].append(features)
                            self.training_data['position']['y'].append(label)
                            collected_samples['position'] += 1
            except Exception as e:
                pass
        
        def record_reveal_decision(self, grid, position):
            try:
                if position and grid:
                    i, j = position
                    # Label: 1 si position stratégique (bordures/coins), 0 sinon
                    is_strategic = (i == 0 or i == GRID_ROWS-1 or j == 0 or j == GRID_COLS-1)
                    label = 1 if is_strategic else 0
                    
                    features = self.extract_comprehensive_features(grid, [], [], position=position)
                    
                    if len(features) > 0 and not np.all(features == 0):
                        self.training_data['reveal']['X'].append(features)
                        self.training_data['reveal']['y'].append(label)
                        collected_samples['reveal'] += 1
            except Exception as e:
                pass
        
        # Ajouter les méthodes au collecteur
        self.record_source_decision = record_source_decision.__get__(self, type(self))
        self.record_keep_decision = record_keep_decision.__get__(self, type(self))
        self.record_position_decision = record_position_decision.__get__(self, type(self))
        self.record_reveal_decision = record_reveal_decision.__get__(self, type(self))
        
        for game_num in range(num_games):
            try:
                # Créer une partie avec notre collecteur personnalisé
                collector_ai = DataCollectorAI(self)
                players = [
                    Player(0, "Collector", collector_ai),
                    Player(1, "Opp1", InitialAI()),
                    Player(2, "Opp2", InitialAI()),
                    Player(3, "Opp3", InitialAI())
                ]
                
                scoreboard = Scoreboard(players)
                game = SkyjoGame(players, scoreboard)
                
                # Jouer la partie complète pour capturer toutes les décisions
                step_count = 0
                while not game.finished and step_count < 1000:
                    game.step()
                    step_count += 1
                
                # Affichage de progression
                if (game_num + 1) % 200 == 0:
                    print(f"   Collecté {game_num + 1}/{num_games} parties...")
                    print(f"   Échantillons: Source={collected_samples['source']}, Keep={collected_samples['keep']}, "
                          f"Position={collected_samples['position']}, Reveal={collected_samples['reveal']}")
                    
            except Exception as e:
                continue
        
        print("✅ Collecte terminée!")
        print(f"   Source: {len(self.training_data['source']['X'])} échantillons")
        print(f"   Keep: {len(self.training_data['keep']['X'])} échantillons")
        print(f"   Position: {len(self.training_data['position']['X'])} échantillons")
        print(f"   Reveal: {len(self.training_data['reveal']['X'])} échantillons")
        
        return True

    def train_models(self):
        """Entraîne les modèles ensemble avec les données collectées"""
        print("🧠 Entraînement des modèles XGBoost Enhanced...")
        
        models_trained = 0
        
        # Entraîner TOUS les modèles
        for decision_type in ['source', 'keep', 'position', 'reveal']:
            X = self.training_data[decision_type]['X']
            y = self.training_data[decision_type]['y']
            
            if len(X) < 50:  # Minimum de données requis
                print(f"⚠️ Pas assez de données pour {decision_type} ({len(X)} échantillons)")
                continue
            
            try:
                print(f"🚀 Entraînement du modèle {decision_type}...")
                X = np.array(X)
                y = np.array(y)
                
                print(f"📊 {decision_type}: {X.shape[0]} échantillons, {X.shape[1]} features")
                
                # Vérifier l'équilibre des classes
                unique, counts = np.unique(y, return_counts=True)
                class_dist = dict(zip(unique, counts))
                print(f"   Classes: {class_dist}")
                
                # Skip si une seule classe
                if len(unique) < 2:
                    print(f"⚠️ Une seule classe pour {decision_type}, skip")
                    continue
                
                # Split train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Normalisation
                scaler = getattr(self, f"{decision_type}_scaler")
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Créer ensemble de modèles
                estimators = []
                for i, params in enumerate(self.xgb_variants):
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_train_scaled, y_train)
                    estimators.append((f'xgb_{i}', model))
                
                # Voting ensemble
                ensemble = VotingClassifier(estimators=estimators, voting='soft')
                ensemble.fit(X_train_scaled, y_train)
                
                # Évaluation
                train_score = ensemble.score(X_train_scaled, y_train)
                test_score = ensemble.score(X_test_scaled, y_test)
                print(f"✅ {decision_type}: Train={train_score:.3f}, Test={test_score:.3f}")
                
                # Sauvegarder le modèle
                setattr(self, f"{decision_type}_ensemble", ensemble)
                models_trained += 1
                
            except Exception as e:
                print(f"❌ Erreur entraînement {decision_type}: {e}")
                continue
        
        # Sauvegarder les modèles
        self.save_models()
        print("💾 Modèles XGBoost Enhanced sauvegardés!")
        print(f"🎯 {models_trained}/4 modèles entraînés avec succès")
        
        return models_trained > 0

