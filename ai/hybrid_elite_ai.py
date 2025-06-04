import random
import math
import numpy as np
from collections import Counter, defaultdict, deque
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class HybridEliteAI(BaseAI):
    """
    HybridEliteAI - Combine les meilleures stratégies:
    - Analyse probabiliste avancée d'AdvancedDominantAI
    - Features engineering de XGBoostEnhancedAI  
    - Système de décision multicritères
    - Adaptation dynamique selon le contexte
    - Optimisation pour performances vs InitialAI
    """
    
    def __init__(self):
        # Paramètres optimisés basés sur les champions
        self.aggression_factor = 1.6
        self.endgame_threshold = 0.65
        self.victory_opportunity_bonus = 5.5
        self.risk_tolerance = 0.80
        self.consistency_factor = 0.75
        
        # Système hybride de décision
        self.decision_weights = {
            'probabilistic': 0.4,  # Poids de l'analyse probabiliste
            'heuristic': 0.35,     # Poids des heuristiques
            'pattern': 0.25        # Poids de la détection de patterns
        }
        
        # Métriques d'adaptation
        self.games_played = 0
        self.performance_history = deque(maxlen=20)
        self.context_performance = defaultdict(list)
        
        # Système de features comme XGBoostEnhanced
        self.feature_cache = {}
        self.decision_confidence = 0.5
        
        # Métriques de performance
        self.decision_count = 0
        self.successful_decisions = 0
        self.adaptation_rate = 0.1
        
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
    
    def extract_comprehensive_state(self, grid, discard, other_grids):
        """Extraction d'état complète inspirée de XGBoostEnhanced"""
        try:
            state = {}
            
            # Analyse de notre grille
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
            
            state['our_revealed'] = our_revealed
            state['our_hidden'] = our_hidden
            state['our_score'] = our_score
            state['our_progress'] = len(our_revealed) / 12.0
            
            # Analyse des adversaires
            opponent_states = []
            for opp_grid in other_grids:
                opp_revealed = []
                opp_score = 0
                
                if opp_grid:
                    for i in range(min(GRID_ROWS, len(opp_grid))):
                        for j in range(min(GRID_COLS, len(opp_grid[i]) if i < len(opp_grid) else 0)):
                            if i < len(opp_grid) and j < len(opp_grid[i]) and opp_grid[i][j] is not None:
                                if self.safe_is_revealed(opp_grid[i][j]):
                                    value = self.safe_get_card_value(opp_grid[i][j])
                                    opp_revealed.append(value)
                                    opp_score += value
                
                if opp_revealed:
                    estimated_final = opp_score + (12 - len(opp_revealed)) * 3.5
                    threat_level = max(0, (25 - estimated_final) / 25.0)
                    
                    opponent_states.append({
                        'revealed_count': len(opp_revealed),
                        'score': opp_score,
                        'estimated_final': estimated_final,
                        'threat_level': threat_level,
                        'progress': len(opp_revealed) / 12.0
                    })
            
            state['opponents'] = opponent_states
            
            # Analyse de la défausse
            if discard and len(discard) > 0:
                recent_values = [self.safe_get_card_value(card) for card in discard[-3:]]
                state['discard'] = {
                    'current_value': self.safe_get_card_value(discard[-1]),
                    'recent_trend': np.mean(recent_values) if recent_values else 0,
                    'size': len(discard)
                }
            else:
                state['discard'] = {'current_value': 0, 'recent_trend': 0, 'size': 0}
            
            # Phase de jeu
            total_revealed = len(our_revealed) + sum(len(opp.get('revealed_count', 0)) for opp in opponent_states)
            total_possible = (len(other_grids) + 1) * 12
            game_progress = total_revealed / total_possible if total_possible > 0 else 0
            
            if game_progress < 0.2:
                state['phase'] = 'opening'
            elif game_progress < 0.5:
                state['phase'] = 'mid'
            elif game_progress < 0.8:
                state['phase'] = 'endgame'
            else:
                state['phase'] = 'final'
            
            state['game_progress'] = game_progress
            
            return state
            
        except Exception as e:
            # État minimal en cas d'erreur
            return {
                'our_revealed': [], 'our_hidden': 12, 'our_score': 0, 'our_progress': 0,
                'opponents': [], 'discard': {'current_value': 0, 'recent_trend': 0, 'size': 0},
                'phase': 'mid', 'game_progress': 0.5
            }
    
    def estimate_card_probabilities(self, revealed_cards):
        """Estimation probabiliste comme AdvancedDominantAI"""
        try:
            # Distribution initiale du deck Skyjo
            card_counts = Counter()
            card_counts.update([-2] * 5)
            card_counts.update([-1] * 10)
            card_counts.update([0] * 15)
            for value in range(1, 13):
                card_counts.update([value] * 10)
            
            # Soustraire les cartes révélées
            for card_value in revealed_cards:
                if card_counts[card_value] > 0:
                    card_counts[card_value] -= 1
            
            total_remaining = sum(card_counts.values())
            if total_remaining == 0:
                return {}
            
            probabilities = {value: count/total_remaining for value, count in card_counts.items()}
            return probabilities
            
        except Exception as e:
            return {i: 1.0/15 for i in range(-2, 13)}
    
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
    
    def detect_victory_opportunity(self, state):
        """Détection d'opportunité de victoire hybride"""
        try:
            our_score = state['our_score']
            our_hidden = state['our_hidden']
            
            # Estimation de notre score final
            expected_hidden = 3.5  # Estimation conservatrice
            our_estimated_final = our_score + (our_hidden * expected_hidden * self.consistency_factor)
            
            # Analyse des adversaires
            min_opponent_threat = float('inf')
            avg_opponent_threat = 0
            
            if state['opponents']:
                threats = [opp['estimated_final'] for opp in state['opponents']]
                min_opponent_threat = min(threats)
                avg_opponent_threat = np.mean(threats)
            
            # Opportunités multiples
            if our_estimated_final < min_opponent_threat - 3:
                return True, "commanding_lead", min_opponent_threat - our_estimated_final
            
            if our_estimated_final < avg_opponent_threat - 1.5 and state['phase'] in ['endgame', 'final']:
                return True, "late_advantage", avg_opponent_threat - our_estimated_final
            
            if our_hidden <= 2 and our_estimated_final < min_opponent_threat:
                return True, "sprint_finish", min_opponent_threat - our_estimated_final
            
            return False, "no_opportunity", 0
            
        except Exception as e:
            return False, "error", 0
    
    def calculate_decision_score(self, decision_type, context, state):
        """Système de scoring multicritères"""
        try:
            scores = {}
            
            # Score probabiliste (style AdvancedDominantAI)
            scores['probabilistic'] = self.calculate_probabilistic_score(decision_type, context, state)
            
            # Score heuristique (règles expertes)
            scores['heuristic'] = self.calculate_heuristic_score(decision_type, context, state)
            
            # Score pattern (détection d'opportunités)
            scores['pattern'] = self.calculate_pattern_score(decision_type, context, state)
            
            # Score final pondéré
            final_score = sum(scores[key] * self.decision_weights[key] for key in scores)
            
            # Ajustement adaptatif basé sur la performance
            if len(self.performance_history) > 5:
                recent_performance = np.mean(self.performance_history[-5:])
                if recent_performance > 0.6:  # Bonne performance récente
                    final_score *= 1.1
                elif recent_performance < 0.4:  # Mauvaise performance
                    final_score *= 0.9
            
            return final_score, scores
            
        except Exception as e:
            return 0.5, {}
    
    def calculate_probabilistic_score(self, decision_type, context, state):
        """Score basé sur l'analyse probabiliste"""
        try:
            if decision_type == "source":
                discard_value = context.get('discard_value', 5)
                # Prendre la défausse est bon si la carte est meilleure que l'espérance
                expected_pile = 4.0  # Estimation
                return max(0, (expected_pile - discard_value) / 10.0)
            
            elif decision_type == "keep":
                card_value = context.get('card_value', 5)
                return max(0, (6 - card_value) / 8.0)
            
            elif decision_type == "position":
                current_value = context.get('current_value', 5)
                card_value = context.get('card_value', 5)
                return max(0, (current_value - card_value) / 12.0)
            
            return 0.5
            
        except Exception as e:
            return 0.5
    
    def calculate_heuristic_score(self, decision_type, context, state):
        """Score basé sur les heuristiques expertes"""
        try:
            base_score = 0.5
            
            if decision_type == "source":
                discard_value = context.get('discard_value', 5)
                if discard_value <= 1:
                    base_score = 0.9
                elif discard_value <= 3:
                    base_score = 0.7
                elif discard_value <= 5:
                    base_score = 0.6
                else:
                    base_score = 0.2
            
            elif decision_type == "keep":
                card_value = context.get('card_value', 5)
                if card_value <= 0:
                    base_score = 0.95
                elif card_value <= 3:
                    base_score = 0.8
                elif card_value <= 6:
                    base_score = 0.5
                else:
                    base_score = 0.1
            
            # Ajustement selon la phase
            if state['phase'] == 'final':
                base_score *= 1.2  # Plus agressif en fin de partie
            elif state['phase'] == 'opening':
                base_score *= 0.9  # Plus conservateur au début
            
            return np.clip(base_score, 0, 1)
            
        except Exception as e:
            return 0.5
    
    def calculate_pattern_score(self, decision_type, context, state):
        """Score basé sur la détection de patterns"""
        try:
            base_score = 0.5
            
            # Détecter opportunité de victoire
            opportunity, opp_type, margin = self.detect_victory_opportunity(state)
            
            if opportunity:
                if decision_type == "source":
                    discard_value = context.get('discard_value', 5)
                    if discard_value <= 5:  # Être plus agressif si opportunité
                        base_score += 0.3 * min(1, margin / 5.0)
                
                elif decision_type == "keep":
                    card_value = context.get('card_value', 5)
                    if card_value <= 7:  # Garder plus de cartes si opportunité
                        base_score += 0.25 * min(1, margin / 5.0)
            
            # Pression des adversaires
            if state['opponents']:
                max_threat = max(opp['threat_level'] for opp in state['opponents'])
                if max_threat > 0.7:  # Adversaire menaçant
                    base_score += 0.2  # Être plus agressif
            
            return np.clip(base_score, 0, 1)
            
        except Exception as e:
            return 0.5
    
    def initial_flip(self):
        """Stratégie initiale optimisée"""
        # Stratégie en diagonale pour maximiser l'information
        return [[0, 0], [GRID_ROWS-1, GRID_COLS-1]]
    
    def choose_source(self, grid, discard, other_grids):
        """Choix de source avec système hybride"""
        try:
            if not discard:
                return 'P'
            
            self.decision_count += 1
            
            # Extraire l'état complet
            state = self.extract_comprehensive_state(grid, discard, other_grids)
            discard_value = self.safe_get_card_value(discard[-1])
            
            # Contexte de décision
            context = {
                'discard_value': discard_value,
                'game_progress': state['game_progress'],
                'phase': state['phase']
            }
            
            # Calculer le score de décision
            score, detailed_scores = self.calculate_decision_score("source", context, state)
            
            # Seuil adaptatif
            base_threshold = 0.55
            
            # Ajustement selon la phase
            if state['phase'] == 'endgame':
                base_threshold *= 0.9  # Plus agressif
            elif state['phase'] == 'final':
                base_threshold *= 0.8  # Très agressif
            
            decision = 'D' if score > base_threshold else 'P'
            
            # Enregistrer pour l'adaptation
            self.record_decision_context(decision, context, state)
            
            return decision
            
        except Exception as e:
            return 'P'
    
    def choose_keep(self, card, grid, other_grids):
        """Décision de garde avec système hybride"""
        try:
            card_value = self.safe_get_card_value(card)
            
            # Extraire l'état
            state = self.extract_comprehensive_state(grid, [], other_grids)
            
            context = {
                'card_value': card_value
            }
            
            # Calculer le score
            score, _ = self.calculate_decision_score("keep", context, state)
            
            # Seuil adaptatif
            base_threshold = 0.5
            
            # Opportunité de victoire
            opportunity, _, margin = self.detect_victory_opportunity(state)
            if opportunity and card_value <= 7:
                base_threshold *= 0.8  # Plus tolérant si opportunité
            
            return score > base_threshold
            
        except Exception as e:
            return card_value <= 4
    
    def choose_position(self, card, grid, other_grids):
        """Choix de position avec évaluation hybride"""
        try:
            if not grid or len(grid) == 0:
                return (0, 0)
            
            card_value = self.safe_get_card_value(card)
            state = self.extract_comprehensive_state(grid, [], other_grids)
            
            best_position = (0, 0)
            best_score = float('-inf')
            
            for i in range(min(GRID_ROWS, len(grid))):
                if i < len(grid) and grid[i]:
                    for j in range(min(GRID_COLS, len(grid[i]))):
                        if j < len(grid[i]) and grid[i][j] is not None:
                            current_value = self.safe_get_card_value(grid[i][j])
                            is_revealed = self.safe_is_revealed(grid[i][j])
                            
                            context = {
                                'current_value': current_value,
                                'card_value': card_value,
                                'is_revealed': is_revealed,
                                'position': (i, j)
                            }
                            
                            score, _ = self.calculate_decision_score("position", context, state)
                            
                            # Bonus positionnels
                            if i == 0 or i == GRID_ROWS - 1:
                                score += 0.1
                            if j == 0 or j == GRID_COLS - 1:
                                score += 0.1
                            
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
            
            # Préférer les coins et bordures
            best_position = unrevealed_positions[0]
            best_score = 0
            
            for i, j in unrevealed_positions:
                score = 0
                
                # Bonus position
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
                
                if score > best_score:
                    best_score = score
                    best_position = (i, j)
            
            return best_position
            
        except Exception as e:
            return None
    
    def record_decision_context(self, decision, context, state):
        """Enregistre le contexte pour l'apprentissage adaptatif"""
        try:
            key = f"{state['phase']}_{context.get('discard_value', 0)}"
            self.context_performance[key].append({
                'decision': decision,
                'context': context,
                'timestamp': self.decision_count
            })
        except Exception as e:
            pass
    
    def adapt_weights(self, game_outcome):
        """Adaptation des poids basée sur les résultats"""
        try:
            self.games_played += 1
            performance = 1.0 if game_outcome == 'win' else 0.0
            self.performance_history.append(performance)
            
            # Ajustement léger des poids selon la performance
            if len(self.performance_history) >= 5:
                recent_perf = np.mean(self.performance_history[-5:])
                
                if recent_perf > 0.7:  # Bonne performance
                    # Augmenter légèrement le poids de la stratégie dominante
                    if self.decision_weights['probabilistic'] < 0.5:
                        self.decision_weights['probabilistic'] += self.adaptation_rate
                elif recent_perf < 0.3:  # Mauvaise performance
                    # Équilibrer les poids
                    self.decision_weights['heuristic'] += self.adaptation_rate
                
                # Normaliser les poids
                total = sum(self.decision_weights.values())
                for key in self.decision_weights:
                    self.decision_weights[key] /= total
                    
        except Exception as e:
            pass
    
    def get_performance_stats(self):
        """Statistiques de performance"""
        stats = {
            'games_played': self.games_played,
            'decision_count': self.decision_count,
            'performance_history': list(self.performance_history),
            'decision_weights': self.decision_weights.copy(),
            'win_rate': np.mean(self.performance_history) if self.performance_history else 0
        }
        return stats 