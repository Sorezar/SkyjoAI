import random
import math
import numpy as np
from collections import deque, defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class AdaptiveMLAI(BaseAI):
    """
    AdaptiveMLAI - Modèle ML avec apprentissage en ligne:
    - Apprentissage continu pendant les parties
    - Adaptation aux adversaires et contextes
    - Modèles légers pour prédictions rapides
    - Système de feedback immédiat
    - Optimisation dynamique des paramètres
    """
    
    def __init__(self):
        # Modèles ML adaptatifs
        self.source_model = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, random_state=42)
        self.keep_model = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.01, random_state=42)
        self.position_model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
        
        # Scalers adaptatifs
        #self.source_scaler = StandardScaler()
        #self.keep_scaler = StandardScaler()
        #self.position_scaler = StandardScaler()
        self.source_scaler = RobustScaler()
        self.keep_scaler = RobustScaler()
        self.position_scaler = RobustScaler()
        
        # Buffers d'apprentissage en ligne
        self.source_buffer = {'X': deque(maxlen=100), 'y': deque(maxlen=100)}
        self.keep_buffer = {'X': deque(maxlen=100), 'y': deque(maxlen=100)}
        self.position_buffer = {'X': deque(maxlen=50), 'y': deque(maxlen=50)}
        
        # Métriques d'adaptation
        self.games_played = 0
        self.learning_rate = 0.05
        self.adaptation_threshold = 10  # Nombre de samples avant adaptation
        
        # Performance tracking
        self.decision_outcomes = defaultdict(list)
        self.context_performance = defaultdict(float)
        self.opponent_profiles = defaultdict(list)
        
        # État d'apprentissage
        self.models_initialized = False
        self.last_predictions = {}
        self.last_contexts = {}
        
        # Paramètres dynamiques
        self.aggression_level = 0.5
        self.risk_tolerance = 0.6
        self.exploration_rate = 0.15
        
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
    
    def extract_adaptive_features(self, grid, discard, other_grids, decision_type="general"):
        """Extraction de features adaptatives"""
        try:
            features = []
            
            # Features de base de notre état
            our_revealed = []
            our_hidden = 0
            our_score = 0
            
            if grid:
                for i in range(min(GRID_ROWS, len(grid))):
                    for j in range(min(GRID_COLS, len(grid[i]) if i < len(grid) else 0)):
                        if i < len(grid) and j < len(grid[i]) and grid[i][j] is not None:
                            if self.safe_is_revealed(grid[i][j]):
                                value = self.safe_get_card_value(grid[i][j])
                                our_revealed.append(value)
                                our_score += value
                            else:
                                our_hidden += 1
            
            # Features numériques de base
            features.extend([
                len(our_revealed),  # Cartes révélées
                our_hidden,         # Cartes cachées
                our_score,          # Score actuel
                np.mean(our_revealed) if our_revealed else 0,     # Score moyen
                np.std(our_revealed) if len(our_revealed) > 1 else 0,  # Variabilité
                sum(1 for v in our_revealed if v <= 2),          # Excellentes cartes
                sum(1 for v in our_revealed if v >= 8),          # Mauvaises cartes
                len(our_revealed) / 12.0,                        # Progression
            ])
            
            # Features des adversaires
            opponent_features = []
            for opp_grid in other_grids[:3]:  # Max 3 adversaires
                if opp_grid:
                    opp_revealed = []
                    for i in range(min(GRID_ROWS, len(opp_grid))):
                        for j in range(min(GRID_COLS, len(opp_grid[i]) if i < len(opp_grid) else 0)):
                            if i < len(opp_grid) and j < len(opp_grid[i]) and opp_grid[i][j] is not None:
                                if self.safe_is_revealed(opp_grid[i][j]):
                                    value = self.safe_get_card_value(opp_grid[i][j])
                                    opp_revealed.append(value)
                    
                    if opp_revealed:
                        opp_score = sum(opp_revealed)
                        estimated_final = opp_score + (12 - len(opp_revealed)) * 3.5
                        threat = max(0, (25 - estimated_final) / 25.0)
                        
                        opponent_features.extend([
                            len(opp_revealed),
                            opp_score,
                            estimated_final,
                            threat,
                            len(opp_revealed) / 12.0  # Progression adversaire
                        ])
                    else:
                        opponent_features.extend([0, 0, 30, 0, 0])
                else:
                    opponent_features.extend([0, 0, 30, 0, 0])
            
            # Padding pour avoir exactement 3 adversaires
            while len(opponent_features) < 15:  # 3 adversaires * 5 features
                opponent_features.extend([0, 0, 30, 0, 0])
            
            features.extend(opponent_features[:15])
            
            # Features de la défausse
            if discard and len(discard) > 0:
                discard_features = [
                    self.safe_get_card_value(discard[-1]),  # Dernière carte
                    len(discard),                           # Taille défausse
                ]
                
                # Tendance récente de la défausse
                if len(discard) >= 3:
                    recent = [self.safe_get_card_value(card) for card in discard[-3:]]
                    discard_features.append(np.mean(recent))
                else:
                    discard_features.append(0)
            else:
                discard_features = [0, 0, 0]
            
            features.extend(discard_features)
            
            # Features contextuelles
            total_revealed = len(our_revealed) + sum(opponent_features[i] for i in range(0, 15, 5))
            game_progress = total_revealed / (4 * 12) if total_revealed > 0 else 0
            
            context_features = [
                game_progress,                    # Progression globale
                self.aggression_level,            # Niveau d'agressivité adaptatif
                self.risk_tolerance,              # Tolérance au risque adaptative
                self.games_played / 100.0,       # Expérience normalisée
            ]
            
            features.extend(context_features)
            
            # Features spécifiques au type de décision
            if decision_type == "source" and discard and len(discard) > 0:
                card_value = self.safe_get_card_value(discard[-1])
                features.extend([
                    card_value / 12.0,            # Valeur normalisée
                    1.0 if card_value <= 2 else 0.0,  # Excellente carte
                    1.0 if card_value >= 8 else 0.0,  # Mauvaise carte
                ])
            elif decision_type == "keep":
                # Features ajoutées dans choose_keep
                features.extend([0, 0, 0])
            else:
                features.extend([0, 0, 0])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            # Fallback avec features par défaut
            return np.zeros(29, dtype=np.float32)
    
    def initialize_models_if_needed(self):
        """Initialise les modèles avec des données synthétiques si nécessaire"""
        if not self.models_initialized and len(self.source_buffer['X']) < 5:
            # Générer quelques échantillons synthétiques pour l'initialisation
            for _ in range(10):
                # Features synthétiques
                synthetic_features = np.random.normal(0.5, 0.2, 29)
                synthetic_features = np.clip(synthetic_features, 0, 1)
                
                # Labels synthétiques basés sur des heuristiques simples
                discard_value = synthetic_features[-3] * 12  # Valeur de carte de défausse
                
                # Source decision (prendre défausse si bonne carte)
                source_label = 1 if discard_value <= 4 else 0
                self.source_buffer['X'].append(synthetic_features)
                self.source_buffer['y'].append(source_label)
                
                # Keep decision
                keep_label = 1 if discard_value <= 5 else 0
                self.keep_buffer['X'].append(synthetic_features)
                self.keep_buffer['y'].append(keep_label)
            
            self.train_models()
            self.models_initialized = True
    
    def train_models(self):
        """Entraîne/met à jour les modèles avec les nouvelles données"""
        try:
            # Entraîner modèle source
            if len(self.source_buffer['X']) >= 5:
                X = np.array(list(self.source_buffer['X']))
                y = np.array(list(self.source_buffer['y']))
                
                # Normalisation
                if hasattr(self.source_scaler, 'n_features_in_') or len(X) > 10:
                    X_scaled = self.source_scaler.fit_transform(X) if len(X) > 10 else self.source_scaler.transform(X)
                else:
                    X_scaled = self.source_scaler.fit_transform(X)
                
                # Entraînement/mise à jour du modèle
                if hasattr(self.source_model, 'partial_fit'):
                    if not hasattr(self.source_model, 'classes_'):
                        self.source_model.partial_fit(X_scaled, y, classes=[0, 1])
                    else:
                        self.source_model.partial_fit(X_scaled, y)
                else:
                    self.source_model.fit(X_scaled, y)
            
            # Entraîner modèle keep
            if len(self.keep_buffer['X']) >= 5:
                X = np.array(list(self.keep_buffer['X']))
                y = np.array(list(self.keep_buffer['y']))
                
                if hasattr(self.keep_scaler, 'n_features_in_') or len(X) > 10:
                    X_scaled = self.keep_scaler.fit_transform(X) if len(X) > 10 else self.keep_scaler.transform(X)
                else:
                    X_scaled = self.keep_scaler.fit_transform(X)
                
                if hasattr(self.keep_model, 'partial_fit'):
                    if not hasattr(self.keep_model, 'classes_'):
                        self.keep_model.partial_fit(X_scaled, y, classes=[0, 1])
                    else:
                        self.keep_model.partial_fit(X_scaled, y)
                else:
                    self.keep_model.fit(X_scaled, y)
            
            # Entraîner modèle position (moins fréquent)
            if len(self.position_buffer['X']) >= 8:
                X = np.array(list(self.position_buffer['X']))
                y = np.array(list(self.position_buffer['y']))
                
                X_scaled = self.position_scaler.fit_transform(X)
                self.position_model.fit(X_scaled, y)
                
        except Exception as e:
            pass
    
    def predict_with_model(self, features, model_type):
        """Prédiction avec gestion d'erreurs"""
        try:
            model = getattr(self, f"{model_type}_model")
            scaler = getattr(self, f"{model_type}_scaler")
            
            # Assurer que les features ont la bonne dimension
            features = features.reshape(1, -1)
            
            # Normalisation
            if hasattr(scaler, 'transform'):
                try:
                    features_scaled = scaler.transform(features)
                except:
                    features_scaled = features
            else:
                features_scaled = features
            
            # Prédiction
            if hasattr(model, 'predict_proba') and hasattr(model, 'classes_'):
                prob = model.predict_proba(features_scaled)[0]
                return prob[1] if len(prob) > 1 else 0.5
            else:
                return 0.5
                
        except Exception as e:
            return 0.5
    
    def record_decision_outcome(self, decision_type, features, decision, outcome):
        """Enregistre le résultat d'une décision pour l'apprentissage"""
        try:
            buffer = getattr(self, f"{decision_type}_buffer")
            
            # Convertir la décision en label
            if decision_type in ["source", "keep"]:
                label = 1 if (decision == 'D' if decision_type == "source" else decision) else 0
            else:
                label = outcome  # Pour position, outcome est déjà un score
            
            buffer['X'].append(features)
            buffer['y'].append(label)
            
            # Entraînement périodique
            if len(buffer['X']) % self.adaptation_threshold == 0:
                self.train_models()
                
        except Exception as e:
            pass
    
    def adapt_parameters(self, game_result):
        """Adapte les paramètres selon les résultats"""
        try:
            self.games_played += 1
            
            # Ajuster l'agressivité selon les résultats
            if game_result == 'win':
                self.aggression_level = min(0.9, self.aggression_level + 0.02)
                self.risk_tolerance = min(0.8, self.risk_tolerance + 0.01)
            elif game_result == 'loss':
                self.aggression_level = max(0.2, self.aggression_level - 0.02)
                self.risk_tolerance = max(0.4, self.risk_tolerance - 0.01)
            
            # Réduire l'exploration avec l'expérience
            self.exploration_rate = max(0.05, self.exploration_rate * 0.995)
            
        except Exception as e:
            pass
    
    def initial_flip(self):
        """Stratégie initiale"""
        return [[0, 0], [GRID_ROWS-1, GRID_COLS-1]]
    
    def choose_source(self, grid, discard, other_grids):
        """Choix de source avec ML adaptatif"""
        try:
            if not discard:
                return 'P'
            
            self.initialize_models_if_needed()
            
            # Extraire features
            features = self.extract_adaptive_features(grid, discard, other_grids, "source")
            
            # Prédiction ML
            ml_score = self.predict_with_model(features, 'source')
            
            # Heuristique de fallback
            discard_value = self.safe_get_card_value(discard[-1])
            heuristic_score = max(0, (6 - discard_value) / 8.0)
            
            # Combinaison adaptative
            if self.models_initialized and len(self.source_buffer['X']) > 20:
                final_score = 0.7 * ml_score + 0.3 * heuristic_score
            else:
                final_score = 0.3 * ml_score + 0.7 * heuristic_score
            
            # Exploration occasionnelle
            if random.random() < self.exploration_rate:
                final_score += random.uniform(-0.2, 0.2)
            
            # Seuil adaptatif
            threshold = 0.5 * self.aggression_level
            decision = 'D' if final_score > threshold else 'P'
            
            # Enregistrer pour apprentissage futur
            self.last_predictions['source'] = final_score
            self.last_contexts['source'] = {
                'features': features,
                'decision': decision,
                'discard_value': discard_value
            }
            
            return decision
            
        except Exception as e:
            return 'P'
    
    def choose_keep(self, card, grid, other_grids):
        """Décision de garde avec ML adaptatif"""
        try:
            card_value = self.safe_get_card_value(card)
            
            self.initialize_models_if_needed()
            
            # Extraire features
            features = self.extract_adaptive_features(grid, [card], other_grids, "keep")
            # Ajouter la valeur de la carte aux features
            features = np.append(features[:-3], [card_value / 12.0, 1.0 if card_value <= 2 else 0.0, 1.0 if card_value >= 8 else 0.0])
            
            # Prédiction ML
            ml_score = self.predict_with_model(features, 'keep')
            
            # Heuristique
            heuristic_score = max(0, (5 - card_value) / 7.0)
            
            # Combinaison
            if self.models_initialized and len(self.keep_buffer['X']) > 20:
                final_score = 0.7 * ml_score + 0.3 * heuristic_score
            else:
                final_score = 0.3 * ml_score + 0.7 * heuristic_score
            
            # Exploration
            if random.random() < self.exploration_rate:
                final_score += random.uniform(-0.15, 0.15)
            
            threshold = 0.5 * self.risk_tolerance
            decision = final_score > threshold
            
            # Enregistrer
            self.last_predictions['keep'] = final_score
            self.last_contexts['keep'] = {
                'features': features,
                'decision': decision,
                'card_value': card_value
            }
            
            return decision
            
        except Exception as e:
            return card_value <= 4
    
    def choose_position(self, card, grid, other_grids):
        """Choix de position avec évaluation ML"""
        try:
            if not grid or len(grid) == 0:
                return (0, 0)
            
            card_value = self.safe_get_card_value(card)
            
            best_position = (0, 0)
            best_score = float('-inf')
            
            for i in range(min(GRID_ROWS, len(grid))):
                if i < len(grid) and grid[i]:
                    for j in range(min(GRID_COLS, len(grid[i]))):
                        if j < len(grid[i]) and grid[i][j] is not None:
                            current_value = self.safe_get_card_value(grid[i][j])
                            is_revealed = self.safe_is_revealed(grid[i][j])
                            
                            # Score de base
                            if is_revealed:
                                score = current_value - card_value
                            else:
                                score = 4.0 - card_value  # Estimation
                            
                            # Bonus position
                            if i == 0 or i == GRID_ROWS - 1:
                                score += 1
                            if j == 0 or j == GRID_COLS - 1:
                                score += 1
                            
                            # Ajustement adaptatif
                            score *= (1 + self.aggression_level * 0.2)
                            
                            if score > best_score:
                                best_score = score
                                best_position = (i, j)
            
            return best_position
            
        except Exception as e:
            return (0, 0) if grid and len(grid) > 0 and len(grid[0]) > 0 else None
    
    def choose_reveal(self, grid):
        """Choix de révélation adaptatif"""
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
            
            # Stratégie adaptative
            if self.aggression_level > 0.7:
                # Plus agressif: préférer les coins
                corner_positions = [(i, j) for i, j in unrevealed_positions 
                                  if (i == 0 or i == GRID_ROWS-1) and (j == 0 or j == GRID_COLS-1)]
                if corner_positions:
                    return random.choice(corner_positions)
            
            # Stratégie standard: bordures
            border_positions = [(i, j) for i, j in unrevealed_positions 
                              if i == 0 or i == GRID_ROWS-1 or j == 0 or j == GRID_COLS-1]
            if border_positions:
                return random.choice(border_positions)
            
            return random.choice(unrevealed_positions)
            
        except Exception as e:
            return None
    
    def get_performance_stats(self):
        """Statistiques de performance"""
        stats = {
            'games_played': self.games_played,
            'models_initialized': self.models_initialized,
            'aggression_level': self.aggression_level,
            'risk_tolerance': self.risk_tolerance,
            'exploration_rate': self.exploration_rate,
            'buffer_sizes': {
                'source': len(self.source_buffer['X']),
                'keep': len(self.keep_buffer['X']),
                'position': len(self.position_buffer['X'])
            }
        }
        return stats 