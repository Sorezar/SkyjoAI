import random
import math
import numpy as np
from collections import Counter, defaultdict
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class ChampionEliteAI(BaseAI):
    """
    ChampionEliteAI - Spécialement optimisé pour dominer InitialAI:
    - Analyse des faiblesses d'InitialAI et contre-stratégies
    - Système de scoring ultra-précis pour battere les 20.9 points
    - Stratégies anti-InitialAI avec sabotage intelligent
    - Optimisation pour scores <13 (objectif de performance)
    - Adaptation spécialisée contre adversaires prévisibles
    """
    
    def __init__(self):
        # Paramètres spécialisés anti-InitialAI
        self.target_score = 13  # Objectif de performance
        self.initiali_baseline = 20.9  # Score de référence d'InitialAI
        
        # Système de scoring ultra-précis
        self.precision_factor = 0.9    # Précision dans les estimations
        self.aggression_ceiling = 1.4   # Niveau d'agressivité contrôlé
        self.victory_threshold = 0.7    # Seuil pour détection d'opportunités
        
        # Anti-patterns spécialisés
        self.initiali_counter_strategies = {
            'early_aggression': 0.3,      # Contrer agressivité précoce
            'endgame_rush': 0.8,          # Contre-attaque en fin de partie
            'card_hoarding': 0.6,         # Contrer accumulation de cartes
            'risk_aversion': 0.9          # Exploiter la frilosité
        }
        
        # Métriques de précision
        self.score_estimates = []
        self.actual_outcomes = []
        self.estimation_accuracy = 0.85
        
        # Système d'optimisation continue
        self.performance_target = self.target_score
        self.adjustment_rate = 0.05
        self.games_analyzed = 0
        
        # Cache de calculs
        self.probability_cache = {}
        self.context_cache = {}
        
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
    
    def analyze_initiali_behavior(self, other_grids):
        """Analyse le comportement typique d'InitialAI pour le contrer"""
        try:
            initiali_patterns = {
                'conservative_opening': 0,
                'predictable_keeps': 0,
                'suboptimal_positions': 0,
                'late_reveals': 0
            }
            
            for opp_grid in other_grids:
                if opp_grid:
                    revealed_count = 0
                    revealed_positions = []
                    revealed_values = []
                    
                    for i in range(min(GRID_ROWS, len(opp_grid))):
                        for j in range(min(GRID_COLS, len(opp_grid[i]) if i < len(opp_grid) else 0)):
                            if i < len(opp_grid) and j < len(opp_grid[i]) and opp_grid[i][j] is not None:
                                if self.safe_is_revealed(opp_grid[i][j]):
                                    revealed_count += 1
                                    revealed_positions.append((i, j))
                                    revealed_values.append(self.safe_get_card_value(opp_grid[i][j]))
                    
                    # Détecter patterns InitialAI
                    if revealed_count < 4:  # Ouverture conservatrice
                        initiali_patterns['conservative_opening'] += 1
                    
                    if revealed_values and np.mean(revealed_values) > 4:  # Garde cartes moyennes
                        initiali_patterns['predictable_keeps'] += 1
                    
                    # Positions sous-optimales (évite les bordures)
                    border_reveals = sum(1 for i, j in revealed_positions 
                                       if i == 0 or i == GRID_ROWS-1 or j == 0 or j == GRID_COLS-1)
                    if border_reveals < revealed_count * 0.5:
                        initiali_patterns['suboptimal_positions'] += 1
            
            return initiali_patterns
            
        except Exception as e:
            return {'conservative_opening': 0, 'predictable_keeps': 0, 'suboptimal_positions': 0, 'late_reveals': 0}
    
    def estimate_ultra_precise_score(self, grid, probabilities):
        """Estimation ultra-précise du score pour battre InitialAI"""
        try:
            revealed_score = 0
            unrevealed_count = 0
            position_bonuses = 0
            
            if grid:
                for i in range(min(GRID_ROWS, len(grid))):
                    for j in range(min(GRID_COLS, len(grid[i]) if i < len(grid) else 0)):
                        if i < len(grid) and j < len(grid[i]) and grid[i][j] is not None:
                            if self.safe_is_revealed(grid[i][j]):
                                value = self.safe_get_card_value(grid[i][j])
                                revealed_score += value
                                
                                # Bonus pour positions optimales
                                if i == 0 or i == GRID_ROWS-1 or j == 0 or j == GRID_COLS-1:
                                    position_bonuses += 0.1
                            else:
                                unrevealed_count += 1
            
            # Estimation super-précise des cartes cachées
            if probabilities:
                expected_hidden = sum(value * prob for value, prob in probabilities.items())
                # Ajustement selon notre précision historique
                expected_hidden *= self.estimation_accuracy
            else:
                expected_hidden = 3.2  # Légèrement optimiste vs InitialAI
            
            # Score final ultra-précis
            estimated_total = (revealed_score + 
                             (unrevealed_count * expected_hidden * self.precision_factor) - 
                             position_bonuses)
            
            # Calibrage pour battre InitialAI
            target_margin = self.initiali_baseline - self.performance_target
            if estimated_total > self.performance_target + target_margin * 0.5:
                estimated_total *= 1.05  # Légère surestimation pour encourager l'agressivité
            
            return estimated_total
            
        except Exception as e:
            return 25  # Estimation conservatrice
    
    def detect_anti_initiali_opportunity(self, grid, other_grids, probabilities):
        """Détecte les opportunités spécifiques pour battre InitialAI"""
        try:
            our_estimated_score = self.estimate_ultra_precise_score(grid, probabilities)
            
            # Analyser comportement InitialAI
            initiali_patterns = self.analyze_initiali_behavior(other_grids)
            
            # Estimer scores adversaires avec biais anti-InitialAI
            opponent_estimates = []
            for opp_grid in other_grids:
                if opp_grid:
                    opp_score = self.estimate_opponent_score_vs_initiali(opp_grid, probabilities)
                    opponent_estimates.append(opp_score)
            
            if not opponent_estimates:
                return False, "no_opponents", 0
            
            min_opponent = min(opponent_estimates)
            avg_opponent = np.mean(opponent_estimates)
            
            # Opportunités spécialisées
            margin = min_opponent - our_estimated_score
            
            # 1. Opportunité de domination totale
            if our_estimated_score < self.target_score and margin > 2:
                return True, "crushing_victory", margin
            
            # 2. Exploitation de la frilosité InitialAI
            if (initiali_patterns['conservative_opening'] >= 2 and 
                our_estimated_score < self.initiali_baseline - 2):
                return True, "exploit_conservatism", margin
            
            # 3. Sprint final contre InitialAI
            unrevealed = sum(1 for i in range(min(GRID_ROWS, len(grid)))
                           for j in range(min(GRID_COLS, len(grid[i]) if i < len(grid) else 0))
                           if (i < len(grid) and j < len(grid[i]) and grid[i][j] is not None and
                               not self.safe_is_revealed(grid[i][j])))
            
            if unrevealed <= 3 and our_estimated_score < min_opponent + 1:
                return True, "sprint_finish", margin
            
            # 4. Exploitation des erreurs InitialAI
            if initiali_patterns['suboptimal_positions'] >= 2:
                return True, "exploit_mistakes", margin * 1.2
            
            return False, "no_opportunity", margin
            
        except Exception as e:
            return False, "error", 0
    
    def estimate_opponent_score_vs_initiali(self, opponent_grid, probabilities):
        """Estime le score adversaire avec biais pour détecter InitialAI"""
        try:
            revealed_score = 0
            unrevealed_count = 0
            initiali_indicators = 0
            
            if opponent_grid:
                for i in range(min(GRID_ROWS, len(opponent_grid))):
                    for j in range(min(GRID_COLS, len(opponent_grid[i]) if i < len(opponent_grid) else 0)):
                        if i < len(opponent_grid) and j < len(opponent_grid[i]) and opponent_grid[i][j] is not None:
                            if self.safe_is_revealed(opponent_grid[i][j]):
                                value = self.safe_get_card_value(opponent_grid[i][j])
                                revealed_score += value
                                
                                # Détecter comportement InitialAI
                                if value > 3 and value < 7:  # Garde cartes moyennes
                                    initiali_indicators += 1
                                if not (i == 0 or i == GRID_ROWS-1 or j == 0 or j == GRID_COLS-1):
                                    initiali_indicators += 0.5  # Évite bordures
                            else:
                                unrevealed_count += 1
            
            # Estimation selon le type d'adversaire détecté
            if initiali_indicators > 2:  # Probable InitialAI
                expected_hidden = 4.2  # InitialAI a tendance à garder cartes moyennes
            else:
                expected_hidden = 3.5  # Adversaire plus avancé
            
            return revealed_score + (unrevealed_count * expected_hidden)
            
        except Exception as e:
            return 30  # Estimation par défaut
    
    def calculate_anti_initiali_score(self, decision_type, context, opportunity_info):
        """Calcul de score spécialisé anti-InitialAI"""
        try:
            base_score = 0.5
            opportunity, opp_type, margin = opportunity_info
            
            if decision_type == "source":
                discard_value = context.get('discard_value', 5)
                
                # Scoring ultra-précis pour battre 20.9
                if discard_value <= 0:
                    base_score = 0.95  # Presque toujours prendre
                elif discard_value <= 2:
                    base_score = 0.85  # Très favorable
                elif discard_value <= 4:
                    base_score = 0.7   # Favorable
                elif discard_value <= 6:
                    base_score = 0.4   # Neutre
                else:
                    base_score = 0.15  # Défavorable
                
                # Bonus d'opportunité anti-InitialAI
                if opportunity:
                    if opp_type == "crushing_victory":
                        base_score += 0.3
                    elif opp_type == "exploit_conservatism":
                        base_score += 0.25  # Exploiter frilosité
                    elif opp_type == "sprint_finish":
                        base_score += 0.4   # Aggressif en fin
                    elif opp_type == "exploit_mistakes":
                        base_score += 0.2
                
            elif decision_type == "keep":
                card_value = context.get('card_value', 5)
                
                # Seuils optimisés pour battre InitialAI
                if card_value <= 0:
                    base_score = 0.98
                elif card_value <= 2:
                    base_score = 0.9
                elif card_value <= 4:
                    base_score = 0.75
                elif card_value <= 6:
                    base_score = 0.45
                elif card_value <= 8:
                    base_score = 0.2
                else:
                    base_score = 0.05
                
                # Ajustement selon opportunité
                if opportunity and card_value <= 7:
                    base_score += 0.15 * min(1, margin / 3.0)
            
            # Ajustement de précision
            base_score *= (0.8 + self.estimation_accuracy * 0.4)
            
            return np.clip(base_score, 0, 1)
            
        except Exception as e:
            return 0.5
    
    def calculate_card_probabilities_cached(self, revealed_cards):
        """Calcul de probabilités avec cache pour optimisation"""
        cache_key = tuple(sorted(revealed_cards))
        
        if cache_key in self.probability_cache:
            return self.probability_cache[cache_key]
        
        try:
            # Distribution initiale
            card_counts = Counter()
            card_counts.update([-2] * 5)
            card_counts.update([-1] * 10)
            card_counts.update([0] * 15)
            for value in range(1, 13):
                card_counts.update([value] * 10)
            
            # Soustraire cartes révélées
            for card_value in revealed_cards:
                if card_counts[card_value] > 0:
                    card_counts[card_value] -= 1
            
            total_remaining = sum(card_counts.values())
            if total_remaining == 0:
                return {}
            
            probabilities = {value: count/total_remaining for value, count in card_counts.items()}
            
            # Cache le résultat
            if len(self.probability_cache) < 100:  # Limiter taille cache
                self.probability_cache[cache_key] = probabilities
            
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
    
    def initial_flip(self):
        """Ouverture optimisée anti-InitialAI"""
        # InitialAI utilise souvent une stratégie conservatrice
        # Nous utilisons une ouverture agressive mais calculée
        return [[0, 0], [GRID_ROWS-1, GRID_COLS-1]]
    
    def choose_source(self, grid, discard, other_grids):
        """Choix de source ultra-optimisé pour battre InitialAI"""
        try:
            if not discard:
                return 'P'
            
            discard_value = self.safe_get_card_value(discard[-1])
            
            # Analyse complète
            revealed_cards = self.collect_all_revealed_cards(grid, other_grids, discard)
            probabilities = self.calculate_card_probabilities_cached(revealed_cards)
            
            # Détection d'opportunité anti-InitialAI
            opportunity_info = self.detect_anti_initiali_opportunity(grid, other_grids, probabilities)
            
            # Contexte de décision
            context = {
                'discard_value': discard_value,
                'game_phase': len(revealed_cards) / 60.0,  # Progression approximative
            }
            
            # Score de décision spécialisé
            decision_score = self.calculate_anti_initiali_score("source", context, opportunity_info)
            
            # Seuil adaptatif selon performance
            base_threshold = 0.55
            
            # Ajustement selon performance cible
            current_estimate = self.estimate_ultra_precise_score(grid, probabilities)
            if current_estimate > self.performance_target:
                base_threshold *= 0.85  # Plus agressif si on dépasse la cible
            
            return 'D' if decision_score > base_threshold else 'P'
            
        except Exception as e:
            return 'P'
    
    def choose_keep(self, card, grid, other_grids):
        """Décision de garde optimisée anti-InitialAI"""
        try:
            card_value = self.safe_get_card_value(card)
            
            # Analyse rapide
            revealed_cards = self.collect_all_revealed_cards(grid, other_grids)
            probabilities = self.calculate_card_probabilities_cached(revealed_cards)
            
            # Opportunité
            opportunity_info = self.detect_anti_initiali_opportunity(grid, other_grids, probabilities)
            
            context = {'card_value': card_value}
            
            # Score de garde
            keep_score = self.calculate_anti_initiali_score("keep", context, opportunity_info)
            
            # Seuil adaptatif selon cible de performance
            threshold = 0.5
            current_estimate = self.estimate_ultra_precise_score(grid, probabilities)
            
            if current_estimate > self.performance_target + 1:
                threshold *= 0.8  # Plus sélectif si score élevé
            elif current_estimate < self.performance_target - 1:
                threshold *= 1.2  # Plus tolérant si score bas
            
            return keep_score > threshold
            
        except Exception as e:
            return card_value <= 4
    
    def choose_position(self, card, grid, other_grids):
        """Choix de position ultra-optimisé"""
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
                            
                            # Score de base ultra-précis
                            if is_revealed:
                                base_score = (current_value - card_value) * 2
                            else:
                                # Estimation optimiste pour cartes cachées
                                expected_value = 4.0 * self.estimation_accuracy
                                base_score = (expected_value - card_value) * 1.5
                            
                            # Bonus position optimale (anti-pattern InitialAI)
                            position_bonus = 0
                            if i == 0 or i == GRID_ROWS - 1:  # Bordures
                                position_bonus += 1.5
                            if j == 0 or j == GRID_COLS - 1:  # Côtés
                                position_bonus += 1.5
                            
                            total_score = base_score + position_bonus
                            
                            if total_score > best_score:
                                best_score = total_score
                                best_position = (i, j)
            
            return best_position
            
        except Exception as e:
            return (0, 0) if grid and len(grid) > 0 and len(grid[0]) > 0 else None
    
    def choose_reveal(self, grid):
        """Choix de révélation anti-InitialAI"""
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
            
            # Stratégie optimale (contraire à InitialAI qui évite les bordures)
            best_position = unrevealed_positions[0]
            best_score = 0
            
            for i, j in unrevealed_positions:
                score = 0
                
                # Bonus position optimale
                if i == 0 or i == GRID_ROWS - 1:
                    score += 3
                if j == 0 or j == GRID_COLS - 1:
                    score += 3
                
                # Bonus complétion colonne
                col_revealed = sum(1 for row in range(min(GRID_ROWS, len(grid)))
                                 if (row < len(grid) and j < len(grid[row]) and 
                                     grid[row][j] is not None and
                                     self.safe_is_revealed(grid[row][j])))
                
                if col_revealed == GRID_ROWS - 1:
                    score += 8  # Très favorable
                elif col_revealed >= GRID_ROWS - 2:
                    score += 5
                
                if score > best_score:
                    best_score = score
                    best_position = (i, j)
            
            return best_position
            
        except Exception as e:
            return None
    
    def update_performance_tracking(self, final_score):
        """Met à jour le suivi de performance pour optimisation continue"""
        try:
            self.games_analyzed += 1
            
            # Ajuster cible de performance si nécessaire
            if final_score < self.target_score:
                # Excellent résultat, maintenir la stratégie
                self.estimation_accuracy = min(0.95, self.estimation_accuracy + 0.01)
            elif final_score < self.initiali_baseline:
                # Bon résultat, légère amélioration
                self.estimation_accuracy = min(0.9, self.estimation_accuracy + 0.005)
            else:
                # Résultat décevant, ajustement nécessaire
                self.performance_target = max(self.target_score, self.performance_target - 0.2)
                self.estimation_accuracy = max(0.7, self.estimation_accuracy - 0.01)
            
            # Nettoyage périodique du cache
            if self.games_analyzed % 20 == 0:
                self.probability_cache.clear()
                
        except Exception as e:
            pass
    
    def get_performance_stats(self):
        """Statistiques de performance anti-InitialAI"""
        stats = {
            'games_analyzed': self.games_analyzed,
            'target_score': self.target_score,
            'performance_target': self.performance_target,
            'estimation_accuracy': self.estimation_accuracy,
            'initiali_baseline': self.initiali_baseline,
            'precision_factor': self.precision_factor,
            'cache_size': len(self.probability_cache)
        }
        return stats 