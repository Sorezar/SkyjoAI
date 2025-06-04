import random
import math
import numpy as np
from collections import defaultdict, deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class UnsupervisedPatternAI(BaseAI):
    """
    UnsupervisedPatternAI - Inspiré du comportement d'UnsupervisedDeepAI:
    - Stratégie risquée avec exploration stochastique
    - Analyse de patterns émergents
    - Adaptation dynamique aux adversaires
    - Apprentissage de patterns de victoire
    - Système probabiliste pour décisions risquées mais payantes
    """
    
    def __init__(self):
        # Paramètres de base inspirés d'UnsupervisedDeepAI
        self.exploration_rate = 0.25      # Taux d'exploration stochastique élevé
        self.risk_tolerance = 0.85        # Tolérance au risque élevée
        self.pattern_memory_size = 50     # Mémoire des patterns
        self.adaptation_factor = 0.15     # Vitesse d'adaptation
        
        # Système de patterns émergents
        self.victory_patterns = deque(maxlen=20)     # Patterns de victoire
        self.defeat_patterns = deque(maxlen=20)      # Patterns de défaite
        self.state_transitions = defaultdict(list)   # Transitions d'état
        self.opponent_behaviors = defaultdict(list)  # Comportements adversaires
        
        # Modèles ML légers pour patterns
        self.pattern_classifier = RandomForestClassifier(
            n_estimators=50, max_depth=8, random_state=42
        )
        self.risk_predictor = RandomForestClassifier(
            n_estimators=30, max_depth=6, random_state=42
        )
        self.scaler = StandardScaler()
        
        # Métriques d'apprentissage
        self.games_played = 0
        self.risky_decisions = 0
        self.successful_risks = 0
        self.pattern_matches = 0
        
        # État interne pour adaptation
        self.current_game_state = {}
        self.decision_confidence = 0.5
        
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
    
    def encode_game_state_pattern(self, grid, discard, other_grids):
        """Encode l'état du jeu pour la détection de patterns"""
        try:
            state_vector = []
            
            # Notre état
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
            
            # Features de notre état
            state_vector.extend([
                len(our_revealed),                                    # Cartes révélées
                our_hidden,                                          # Cartes cachées
                our_score / max(1, len(our_revealed)),              # Score moyen
                np.std(our_revealed) if len(our_revealed) > 1 else 0, # Variance
                sum(1 for v in our_revealed if v <= 2),             # Excellentes cartes
                sum(1 for v in our_revealed if v >= 8),             # Mauvaises cartes
                our_score,                                           # Score total
            ])
            
            # État de la défausse
            if discard and len(discard) > 0:
                recent_discard = [self.safe_get_card_value(card) for card in discard[-3:]]
                state_vector.extend([
                    len(recent_discard),
                    np.mean(recent_discard) if recent_discard else 0,
                    self.safe_get_card_value(discard[-1])
                ])
            else:
                state_vector.extend([0, 0, 0])
            
            # État des adversaires (agrégé)
            total_opp_revealed = 0
            total_opp_score = 0
            min_opp_threat = float('inf')
            
            for opp_grid in other_grids:
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
                        total_opp_revealed += len(opp_revealed)
                        total_opp_score += opp_score
                        
                        # Estimer la menace de cet adversaire
                        estimated_final = opp_score + (12 - len(opp_revealed)) * 3.5
                        min_opp_threat = min(min_opp_threat, estimated_final)
            
            state_vector.extend([
                total_opp_revealed,
                total_opp_score,
                min_opp_threat if min_opp_threat != float('inf') else 30,
                total_opp_score / max(1, total_opp_revealed),  # Score moyen adversaires
            ])
            
            # Métrique de pression temporelle
            progress = (len(our_revealed) + total_opp_revealed) / (len(other_grids) + 1) / 12.0
            state_vector.append(progress)
            
            return np.array(state_vector, dtype=np.float32)
            
        except Exception as e:
            # Fallback avec état minimal
            return np.zeros(15, dtype=np.float32)
    
    def detect_emerging_pattern(self, state_vector):
        """Détecte des patterns émergents dans l'état"""
        try:
            # Comparaison avec patterns de victoire connus
            if len(self.victory_patterns) > 5:
                similarities = []
                for victory_state in self.victory_patterns:
                    if len(victory_state) == len(state_vector):
                        # Similarité cosinus simple
                        dot_product = np.dot(state_vector, victory_state)
                        norms = np.linalg.norm(state_vector) * np.linalg.norm(victory_state)
                        similarity = dot_product / max(norms, 1e-10)
                        similarities.append(similarity)
                
                if similarities:
                    max_similarity = max(similarities)
                    if max_similarity > 0.7:  # Seuil de pattern
                        self.pattern_matches += 1
                        return True, max_similarity
            
            return False, 0.0
            
        except Exception as e:
            return False, 0.0
    
    def calculate_risk_reward_ratio(self, action_type, context):
        """Calcule le ratio risque/récompense pour une action"""
        try:
            base_risk = 0.5
            base_reward = 0.5
            
            if action_type == "take_discard":
                discard_value = context.get('discard_value', 5)
                # Plus la carte est bonne, plus la récompense est élevée
                base_reward = max(0.1, (6 - discard_value) / 8.0)
                # Risque d'exposition
                base_risk = 0.3 + context.get('game_progress', 0) * 0.3
                
            elif action_type == "keep_card":
                card_value = context.get('card_value', 5)
                base_reward = max(0.1, (5 - card_value) / 7.0)
                base_risk = 0.2 + context.get('opponent_pressure', 0) * 0.4
                
            elif action_type == "risky_position":
                gain = context.get('potential_gain', 0)
                base_reward = min(0.9, gain / 10.0)
                base_risk = 0.4 + context.get('uncertainty', 0) * 0.3
            
            # Ajustement basé sur patterns
            pattern_detected, pattern_strength = self.detect_emerging_pattern(
                context.get('state_vector', np.zeros(15))
            )
            
            if pattern_detected:
                base_reward *= (1.0 + pattern_strength * 0.5)  # Bonus pattern
                base_risk *= (1.0 - pattern_strength * 0.2)    # Réduction risque
            
            ratio = base_reward / max(base_risk, 0.1)
            return ratio
            
        except Exception as e:
            return 1.0
    
    def should_take_risk(self, risk_reward_ratio, exploration_bonus=0):
        """Décide si prendre un risque selon UnsupervisedDeepAI style"""
        try:
            # Seuil de base ajusté par l'exploration
            base_threshold = 0.8 - exploration_bonus
            
            # Ajustement adaptatif basé sur succès récents
            if self.risky_decisions > 10:
                success_rate = self.successful_risks / self.risky_decisions
                adaptive_adjustment = (success_rate - 0.5) * 0.3
                base_threshold += adaptive_adjustment
            
            # Facteur stochastique (caractéristique d'UnsupervisedDeepAI)
            stochastic_factor = random.gauss(0, 0.15)  # Bruit gaussien
            final_threshold = base_threshold + stochastic_factor
            
            # Décision finale
            should_risk = risk_reward_ratio > final_threshold
            
            if should_risk:
                self.risky_decisions += 1
            
            return should_risk
            
        except Exception as e:
            return random.random() < self.exploration_rate
    
    def initial_flip(self):
        """Stratégie initiale avec exploration"""
        # Mélange de stratégique et d'exploration comme UnsupervisedDeepAI
        strategic_positions = [[0, 0], [0, GRID_COLS-1], [GRID_ROWS-1, 0], [GRID_ROWS-1, GRID_COLS-1]]
        exploratory_positions = [[1, 1], [1, 2], [2, 1], [2, 2]]
        
        if random.random() < self.exploration_rate:
            # Choix exploratoire
            return random.sample(strategic_positions + exploratory_positions, 2)
        else:
            # Choix stratégique
            return random.sample(strategic_positions, 2)
    
    def choose_source(self, grid, discard, other_grids):
        """Choix de source avec analyse pattern et risque"""
        try:
            if not discard:
                return 'P'
            
            # Encoder l'état
            state_vector = self.encode_game_state_pattern(grid, discard, other_grids)
            discard_value = self.safe_get_card_value(discard[-1])
            
            # Analyser le contexte
            context = {
                'discard_value': discard_value,
                'state_vector': state_vector,
                'game_progress': np.sum(state_vector[:2]) / 24.0,  # Progression approximative
                'opponent_pressure': state_vector[-4] if len(state_vector) > 4 else 0
            }
            
            # Calculer ratio risque/récompense
            risk_reward_ratio = self.calculate_risk_reward_ratio("take_discard", context)
            
            # Bonus d'exploration pour reproduire le comportement UnsupervisedDeepAI
            exploration_bonus = self.exploration_rate * random.random()
            
            # Décision risquée
            if self.should_take_risk(risk_reward_ratio, exploration_bonus):
                return 'D'
            
            # Fallback: décision conservatrice avec un peu d'aléatoire
            if discard_value <= 2:
                return 'D'
            elif discard_value <= 5 and random.random() < 0.4:
                return 'D'
            else:
                return 'P'
                
        except Exception as e:
            return 'P'
    
    def choose_keep(self, card, grid, other_grids):
        """Décision de garde avec analyse pattern"""
        try:
            card_value = self.safe_get_card_value(card)
            
            # Encoder l'état
            state_vector = self.encode_game_state_pattern(grid, [card], other_grids)
            
            context = {
                'card_value': card_value,
                'state_vector': state_vector,
                'opponent_pressure': state_vector[-4] if len(state_vector) > 4 else 0
            }
            
            risk_reward_ratio = self.calculate_risk_reward_ratio("keep_card", context)
            
            # Décision avec exploration stochastique
            if card_value <= 0:
                return True  # Toujours garder les cartes négatives
            elif card_value <= 3:
                return self.should_take_risk(risk_reward_ratio * 1.5)  # Bonus pour bonnes cartes
            elif card_value <= 6:
                return self.should_take_risk(risk_reward_ratio)
            else:
                # Même les mauvaises cartes peuvent être gardées si pattern détecté
                pattern_detected, _ = self.detect_emerging_pattern(state_vector)
                if pattern_detected and random.random() < self.exploration_rate:
                    return True
                return False
                
        except Exception as e:
            return card_value <= 4
    
    def choose_position(self, card, grid, other_grids):
        """Choix de position avec pattern et exploration"""
        try:
            if not grid or len(grid) == 0:
                return (0, 0)
            
            card_value = self.safe_get_card_value(card)
            state_vector = self.encode_game_state_pattern(grid, [], other_grids)
            
            best_position = (0, 0)
            best_score = float('-inf')
            
            for i in range(min(GRID_ROWS, len(grid))):
                if i < len(grid) and grid[i]:
                    for j in range(min(GRID_COLS, len(grid[i]))):
                        if j < len(grid[i]) and grid[i][j] is not None:
                            score = self.evaluate_position_pattern(
                                i, j, card_value, grid, state_vector
                            )
                            
                            if score > best_score:
                                best_score = score
                                best_position = (i, j)
            
            return best_position
            
        except Exception as e:
            return (0, 0) if grid and len(grid) > 0 and len(grid[0]) > 0 else None
    
    def evaluate_position_pattern(self, i, j, card_value, grid, state_vector):
        """Évalue une position avec analyse pattern"""
        try:
            current_card = grid[i][j]
            current_value = self.safe_get_card_value(current_card) if current_card else 12
            is_revealed = self.safe_is_revealed(current_card) if current_card else False
            
            # Score de base
            if is_revealed:
                base_score = current_value - card_value
            else:
                # Estimation probabiliste avec biais d'exploration
                expected_value = 4.5 + random.gauss(0, 1.5)  # Bruit stochastique
                base_score = expected_value - card_value
            
            # Bonus de position
            position_bonus = 0
            if i == 0 or i == GRID_ROWS - 1:  # Bordures
                position_bonus += 1
            if j == 0 or j == GRID_COLS - 1:  # Côtés  
                position_bonus += 1
            
            # Bonus pattern
            pattern_bonus = 0
            pattern_detected, pattern_strength = self.detect_emerging_pattern(state_vector)
            if pattern_detected:
                # Les patterns encouragent des choix plus risqués
                if not is_revealed:  # Remplacer carte cachée = plus risqué
                    pattern_bonus = pattern_strength * 3
                elif current_value > card_value:  # Amélioration visible
                    pattern_bonus = pattern_strength * 2
            
            # Facteur d'exploration aléatoire (caractéristique d'UnsupervisedDeepAI)
            exploration_noise = random.gauss(0, 0.8) * self.exploration_rate
            
            total_score = base_score + position_bonus + pattern_bonus + exploration_noise
            
            return total_score
            
        except Exception as e:
            return random.random()  # Score aléatoire en cas d'erreur
    
    def choose_reveal(self, grid):
        """Choix de révélation avec exploration stochastique"""
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
            
            # Mélange stratégique et exploration
            if random.random() < self.exploration_rate:
                # Choix exploratoire pur
                return random.choice(unrevealed_positions)
            else:
                # Choix avec scoring
                best_position = unrevealed_positions[0]
                best_score = float('-inf')
                
                for i, j in unrevealed_positions:
                    score = self.evaluate_reveal_position(i, j, grid)
                    if score > best_score:
                        best_score = score
                        best_position = (i, j)
                
                return best_position
            
        except Exception as e:
            return None
    
    def evaluate_reveal_position(self, i, j, grid):
        """Évalue une position pour révélation"""
        try:
            score = 0
            
            # Bonus position stratégique
            if i == 0 or i == GRID_ROWS - 1:
                score += 2
            if j == 0 or j == GRID_COLS - 1:
                score += 2
            
            # Bonus complétion colonne
            col_revealed = sum(1 for row in range(min(GRID_ROWS, len(grid)))
                             if (row < len(grid) and j < len(grid[row]) and 
                                 grid[row][j] is not None and
                                 self.safe_is_revealed(grid[row][j])))
            
            if col_revealed == GRID_ROWS - 1:
                score += 5
            elif col_revealed >= GRID_ROWS - 2:
                score += 3
            
            # Facteur d'exploration
            exploration_bonus = random.gauss(0, 1.0) * self.exploration_rate
            
            return score + exploration_bonus
            
        except Exception as e:
            return random.random()
    
    def record_game_outcome(self, won, final_score, game_state):
        """Enregistre le résultat pour apprentissage"""
        try:
            self.games_played += 1
            
            if won:
                self.victory_patterns.append(game_state)
                # Ajuster les paramètres pour succès
                self.exploration_rate = max(0.15, self.exploration_rate * 0.98)
                if self.risky_decisions > 0:
                    self.successful_risks += 1
            else:
                self.defeat_patterns.append(game_state)
                # Augmenter légèrement l'exploration après échec
                self.exploration_rate = min(0.35, self.exploration_rate * 1.02)
            
            # Maintenir l'équilibre exploration/exploitation
            if self.games_played % 10 == 0:
                self.adapt_parameters()
                
        except Exception as e:
            pass
    
    def adapt_parameters(self):
        """Adaptation des paramètres basée sur l'historique"""
        try:
            if self.games_played > 20:
                # Calculer le taux de succès des risques
                if self.risky_decisions > 0:
                    risk_success_rate = self.successful_risks / self.risky_decisions
                    
                    # Ajuster la tolérance au risque
                    if risk_success_rate > 0.6:
                        self.risk_tolerance = min(0.95, self.risk_tolerance * 1.05)
                    elif risk_success_rate < 0.4:
                        self.risk_tolerance = max(0.7, self.risk_tolerance * 0.95)
                
                # Ajuster l'exploration
                if len(self.victory_patterns) > len(self.defeat_patterns):
                    # Plus de victoires que de défaites
                    self.exploration_rate *= 0.95  # Réduire exploration
                else:
                    self.exploration_rate *= 1.05  # Augmenter exploration
                
                # Borner l'exploration
                self.exploration_rate = np.clip(self.exploration_rate, 0.1, 0.4)
                
        except Exception as e:
            pass
    
    def get_performance_stats(self):
        """Retourne les statistiques de performance"""
        stats = {
            'games_played': self.games_played,
            'risky_decisions': self.risky_decisions,
            'successful_risks': self.successful_risks,
            'risk_success_rate': self.successful_risks / max(1, self.risky_decisions),
            'pattern_matches': self.pattern_matches,
            'victory_patterns': len(self.victory_patterns),
            'exploration_rate': self.exploration_rate,
            'risk_tolerance': self.risk_tolerance
        }
        return stats 