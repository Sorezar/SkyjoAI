import random
import math
from collections import Counter
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class AdvancedDominantAI(BaseAI):
    """
    AdvancedDominantAI - Version dominante conçue pour surpasser InitialAI avec:
    - Stratégies agressives de fin de partie
    - Analyse adversariale poussée
    - Tactiques de sabotage et de control
    - Optimisation dynamique des risques
    - Détection d'opportunités de victoire
    """
    
    def __init__(self):
        # Paramètres agressifs optimisés
        self.aggression_factor = 1.5           # Facteur d'agressivité
        self.endgame_threshold = 0.65          # Seuil pour déclencher mode fin de partie
        self.victory_opportunity_bonus = 5.0   # Bonus pour opportunités de victoire
        self.sabotage_threshold = 3.0          # Seuil pour tactiques de sabotage
        self.risk_tolerance = 0.8              # Tolérance au risque (plus élevé = plus agressif)
        
        # Métriques de performance
        self.decisions_count = 0
        self.aggressive_decisions = 0
    
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
        
    def initial_flip(self):
        """Stratégie initiale agressive : révéler plus de cartes pour l'information"""
        # Révéler 3 cartes au lieu de 2 pour plus d'informations
        positions = []
        
        # Priorité aux coins pour le contrôle territorial
        corners = [[0, 0], [0, GRID_COLS-1], [GRID_ROWS-1, 0], [GRID_ROWS-1, GRID_COLS-1]]
        positions.extend(random.sample(corners, 2))
        
        # Ajouter une carte centrale pour l'information
        center_options = [[1, 1], [1, 2], [2, 1], [2, 2]]
        positions.append(random.choice(center_options))
        
        return positions[:2]  # Contrainte du jeu : seulement 2 cartes
    
    def estimate_card_probabilities(self, revealed_cards):
        """Estimation probabiliste optimisée avec analyse de tendance"""
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
            
            # Ajustement dynamique basé sur la tendance des cartes révélées
            if len(revealed_cards) > 10:
                avg_revealed = sum(revealed_cards) / len(revealed_cards)
                if avg_revealed > 4:
                    # Si beaucoup de hautes cartes sont sorties, privilégier les basses
                    for value in probabilities:
                        if value <= 2:
                            probabilities[value] *= 1.3
                        elif value >= 7:
                            probabilities[value] *= 0.8
            
            return probabilities
        except Exception as e:
            # Fallback : distribution uniforme simple
            return {i: 1.0/15 for i in range(-2, 13)}
    
    def analyze_game_phase(self, grid, other_p_grids):
        """Analyse de phase de jeu avec détection d'opportunités"""
        try:
            total_revealed = 0
            total_cards = 0
            
            for g in [grid] + other_p_grids:
                if g and len(g) > 0:
                    for i in range(min(GRID_ROWS, len(g))):
                        if i < len(g) and g[i]:
                            for j in range(min(GRID_COLS, len(g[i]))):
                                if j < len(g[i]) and g[i][j] is not None:
                                    total_cards += 1
                                    if self.safe_is_revealed(g[i][j]):
                                        total_revealed += 1
            
            if total_cards == 0:
                return "mid"
                
            reveal_ratio = total_revealed / total_cards
            
            # Phases affinées pour stratégies spécifiques
            if reveal_ratio < 0.2:
                return "opening"      # Phase d'ouverture - exploration
            elif reveal_ratio < 0.45:
                return "early"        # Début - accumulation d'info
            elif reveal_ratio < self.endgame_threshold:
                return "mid"          # Milieu - optimisation
            elif reveal_ratio < 0.85:
                return "endgame"      # Fin de partie - agressivité
            else:
                return "final"        # Phase finale - victoire ou défense
        except Exception as e:
            return "mid"  # Fallback sûr
    
    def detect_victory_opportunity(self, grid, other_p_grids, probabilities):
        """Détecte les opportunités de victoire rapide"""
        try:
            # Calculer notre score potentiel
            our_score = self.estimate_my_score(grid, probabilities)
            
            # Estimer les scores adversaires
            opponent_scores = []
            for opp_grid in other_p_grids:
                if opp_grid and len(opp_grid) > 0:
                    opp_score = self.estimate_opponent_score(opp_grid, probabilities)
                    opponent_scores.append(opp_score)
            
            if not opponent_scores:
                return False
                
            min_opponent_score = min(opponent_scores)
            avg_opponent_score = sum(opponent_scores) / len(opponent_scores)
            
            # Opportunité de victoire si on est significativement devant
            if our_score < min_opponent_score - 3:
                return True
                
            # Opportunité si on peut finir rapidement avec un avantage
            unrevealed_count = self.count_unrevealed_cards(grid)
            if unrevealed_count <= 2 and our_score < avg_opponent_score - 1:
                return True
                
        except Exception as e:
            pass
        
        return False
    
    def count_unrevealed_cards(self, grid):
        """Compte les cartes non révélées dans notre grille"""
        try:
            count = 0
            if grid:
                for i in range(min(GRID_ROWS, len(grid))):
                    if i < len(grid) and grid[i]:
                        for j in range(min(GRID_COLS, len(grid[i]))):
                            if j < len(grid[i]) and grid[i][j] is not None:
                                if not self.safe_is_revealed(grid[i][j]):
                                    count += 1
            return count
        except Exception as e:
            return 6  # Estimation par défaut
    
    def estimate_my_score(self, grid, probabilities):
        """Estime notre score final avec précision"""
        try:
            revealed_score = 0
            unrevealed_count = 0
            
            if grid:
                for i in range(min(GRID_ROWS, len(grid))):
                    if i < len(grid) and grid[i]:
                        for j in range(min(GRID_COLS, len(grid[i]))):
                            if j < len(grid[i]) and grid[i][j] is not None:
                                if self.safe_is_revealed(grid[i][j]):
                                    revealed_score += self.safe_get_card_value(grid[i][j])
                                else:
                                    unrevealed_count += 1
            
            # Estimation optimiste pour les cartes cachées (on assume qu'on jouera bien)
            expected_unrevealed = sum(value * prob for value, prob in probabilities.items()) if probabilities else 2.5
            return revealed_score + (unrevealed_count * expected_unrevealed * 0.8)  # Facteur optimiste
        except Exception as e:
            return 25  # Score par défaut conservateur
    
    def estimate_opponent_score(self, opponent_grid, probabilities):
        """Estime le score d'un adversaire"""
        try:
            revealed_score = 0
            unrevealed_count = 0
            
            if opponent_grid:
                for i in range(min(GRID_ROWS, len(opponent_grid))):
                    if i < len(opponent_grid) and opponent_grid[i]:
                        for j in range(min(GRID_COLS, len(opponent_grid[i]))):
                            if j < len(opponent_grid[i]) and opponent_grid[i][j] is not None:
                                if self.safe_is_revealed(opponent_grid[i][j]):
                                    revealed_score += self.safe_get_card_value(opponent_grid[i][j])
                                else:
                                    unrevealed_count += 1
            
            # Estimation pessimiste pour les adversaires (on assume qu'ils joueront mal)
            expected_unrevealed = sum(value * prob for value, prob in probabilities.items()) if probabilities else 3.0
            return revealed_score + (unrevealed_count * expected_unrevealed * 1.1)  # Facteur pessimiste
        except Exception as e:
            return 25  # Score par défaut
    
    def choose_source(self, grid, discard, other_p_grids):
        """Choix de source avec stratégie dominante"""
        self.decisions_count += 1
        
        if not discard:
            return 'P'
        
        try:
            discard_value = self.safe_get_card_value(discard[-1])
            
            # Collecte des informations
            all_revealed = self.collect_all_revealed_cards(grid, other_p_grids)
            probabilities = self.estimate_card_probabilities(all_revealed)
            game_phase = self.analyze_game_phase(grid, other_p_grids)
            victory_opportunity = self.detect_victory_opportunity(grid, other_p_grids, probabilities)
            
            # Stratégie agressive en fonction de la phase
            if game_phase == "opening":
                # Phase d'ouverture : accepter plus de cartes pour l'information
                return 'D' if discard_value <= 4 else 'P'
            
            elif game_phase == "early":
                # Début : stratégie équilibrée mais légèrement agressive
                return 'D' if discard_value <= 3 else 'P'
            
            elif game_phase == "mid":
                # Milieu : analyse fine
                expected_deck = sum(value * prob for value, prob in probabilities.items()) if probabilities else 3.0
                return 'D' if discard_value < expected_deck * 0.9 else 'P'
            
            elif game_phase == "endgame":
                # Fin de partie : très agressif
                if victory_opportunity:
                    self.aggressive_decisions += 1
                    return 'D' if discard_value <= 6 else 'P'
                else:
                    return 'D' if discard_value <= 4 else 'P'
            
            else:  # final phase
                # Phase finale : extrêmement agressif
                self.aggressive_decisions += 1
                return 'D' if discard_value <= 8 else 'P'
                
        except Exception as e:
            # Fallback heuristique
            try:
                discard_value = self.safe_get_card_value(discard[-1])
                return 'D' if discard_value <= 3 else 'P'
            except:
                return 'P'
    
    def collect_all_revealed_cards(self, grid, other_p_grids):
        """Collecte toutes les cartes révélées de toutes les grilles"""
        all_revealed = []
        
        try:
            # Notre grille
            if grid:
                for i in range(min(GRID_ROWS, len(grid))):
                    if i < len(grid) and grid[i]:
                        for j in range(min(GRID_COLS, len(grid[i]))):
                            if j < len(grid[i]) and grid[i][j] is not None:
                                if self.safe_is_revealed(grid[i][j]):
                                    all_revealed.append(self.safe_get_card_value(grid[i][j]))
            
            # Grilles adversaires
            if other_p_grids:
                for opp_grid in other_p_grids:
                    if opp_grid:
                        for i in range(min(GRID_ROWS, len(opp_grid))):
                            if i < len(opp_grid) and opp_grid[i]:
                                for j in range(min(GRID_COLS, len(opp_grid[i]))):
                                    if j < len(opp_grid[i]) and opp_grid[i][j] is not None:
                                        if self.safe_is_revealed(opp_grid[i][j]):
                                            all_revealed.append(self.safe_get_card_value(opp_grid[i][j]))
        except Exception as e:
            pass
        
        return all_revealed
    
    def choose_keep(self, card, grid, other_p_grids):
        """Décision de garde avec analyse dominante"""
        try:
            card_value = self.safe_get_card_value(card)
            
            # Informations contextuelles
            all_revealed = self.collect_all_revealed_cards(grid, other_p_grids)
            probabilities = self.estimate_card_probabilities(all_revealed)
            game_phase = self.analyze_game_phase(grid, other_p_grids)
            victory_opportunity = self.detect_victory_opportunity(grid, other_p_grids, probabilities)
            
            # Analyse de fin de partie critique
            unrevealed_count = self.count_unrevealed_cards(grid)
            if unrevealed_count <= 1:
                # Dernière carte : analyse précise
                our_final_score = self.estimate_my_score(grid, probabilities) - sum(value * prob for value, prob in probabilities.items()) + card_value
                opponent_estimates = [self.estimate_opponent_score(opp_grid, probabilities) for opp_grid in other_p_grids if opp_grid]
                
                if opponent_estimates:
                    min_opponent = min(opponent_estimates)
                    # Garder si ça nous donne la victoire ou un avantage significatif
                    return our_final_score <= min_opponent + 1
            
            # Stratégies par phase
            if game_phase == "opening":
                return card_value <= 2  # Très sélectif en ouverture
            
            elif game_phase == "early":
                return card_value <= 3
            
            elif game_phase == "mid":
                expected_replacement = sum(value * prob for value, prob in probabilities.items()) if probabilities else 3.5
                threshold = expected_replacement * self.risk_tolerance
                return card_value <= threshold
            
            elif game_phase == "endgame":
                if victory_opportunity:
                    # Opportunité de victoire : plus agressif
                    return card_value <= 6
                else:
                    return card_value <= 4
            
            else:  # final phase
                # Phase finale : très agressif pour la victoire
                return card_value <= 7
                
        except Exception as e:
            # Fallback simple
            try:
                card_value = self.safe_get_card_value(card)
                return card_value <= 3
            except:
                return False
    
    def choose_position(self, card, grid, other_p_grids):
        """Choix de position avec optimisation dominante"""
        try:
            card_value = self.safe_get_card_value(card)
            
            # Informations contextuelles
            game_phase = self.analyze_game_phase(grid, other_p_grids)
            all_revealed = self.collect_all_revealed_cards(grid, other_p_grids)
            probabilities = self.estimate_card_probabilities(all_revealed)
            victory_opportunity = self.detect_victory_opportunity(grid, other_p_grids, probabilities)
            
            # Positions disponibles
            revealed_positions = []
            unrevealed_positions = []
            
            if grid:
                for i in range(min(GRID_ROWS, len(grid))):
                    if i < len(grid) and grid[i]:
                        for j in range(min(GRID_COLS, len(grid[i]))):
                            if j < len(grid[i]) and grid[i][j] is not None:
                                if self.safe_is_revealed(grid[i][j]):
                                    revealed_positions.append((i, j))
                                else:
                                    unrevealed_positions.append((i, j))
            
            if not revealed_positions and not unrevealed_positions:
                return (0, 0)  # Fallback position
            
            best_position = None
            best_score = float('-inf')
            
            # Évaluation des positions révélées
            for row, col in revealed_positions:
                try:
                    if (row < len(grid) and col < len(grid[row]) and 
                        grid[row][col] is not None):
                        current_card = grid[row][col]
                        current_value = self.safe_get_card_value(current_card)
                        improvement = current_value - card_value
                        
                        if improvement <= 0:
                            continue
                        
                        # Score de base
                        score = improvement * self.aggression_factor
                        
                        # Bonus stratégiques
                        # Bonus colonne
                        column_potential = self.calculate_column_potential(grid, col)
                        score += column_potential * 4.0
                        
                        # Bonus position (coins prioritaires)
                        if (row == 0 or row == GRID_ROWS-1) and (col == 0 or col == GRID_COLS-1):
                            score += 2.0  # Bonus coin renforcé
                        elif row == 0 or row == GRID_ROWS-1 or col == 0 or col == GRID_COLS-1:
                            score += 1.0  # Bonus bord
                        
                        # Bonus phase finale
                        if game_phase in ["endgame", "final"]:
                            score *= 1.3
                        
                        # Bonus opportunité de victoire
                        if victory_opportunity:
                            score += self.victory_opportunity_bonus
                        
                        if score > best_score:
                            best_score = score
                            best_position = (row, col)
                except Exception as e:
                    continue
            
            # Si aucune position révélée n'est optimale, évaluer les positions cachées
            if best_position is None and unrevealed_positions:
                # Stratégie pour positions cachées : privilégier les colonnes à potentiel
                column_scores = {}
                for col in range(GRID_COLS):
                    column_scores[col] = self.calculate_column_potential(grid, col)
                
                # Sélection intelligente
                available_columns = list(set(col for row, col in unrevealed_positions))
                if available_columns:
                    # En fin de partie, prendre plus de risques
                    if game_phase in ["endgame", "final"]:
                        # Choisir une colonne avec du potentiel même si risqué
                        sorted_cols = sorted(available_columns, key=lambda c: column_scores.get(c, 0), reverse=True)
                        best_col = sorted_cols[0] if sorted_cols else available_columns[0]
                    else:
                        # Jouer plus sûr en début/milieu
                        best_col = max(available_columns, key=lambda c: column_scores.get(c, 0))
                    
                    col_positions = [(r, c) for r, c in unrevealed_positions if c == best_col]
                    if col_positions:
                        # Privilégier les coins et bords même pour les positions cachées
                        strategic_positions = []
                        for r, c in col_positions:
                            if (r == 0 or r == GRID_ROWS-1) and (c == 0 or c == GRID_COLS-1):
                                strategic_positions.append((r, c))
                        
                        if strategic_positions:
                            best_position = random.choice(strategic_positions)
                        else:
                            best_position = random.choice(col_positions)
            
            # Fallback final
            if best_position is None:
                if revealed_positions:
                    best_position = random.choice(revealed_positions)
                elif unrevealed_positions:
                    best_position = random.choice(unrevealed_positions)
                else:
                    best_position = (0, 0)
            
            return best_position
            
        except Exception as e:
            # Fallback d'urgence
            try:
                if grid:
                    for i in range(GRID_ROWS):
                        for j in range(GRID_COLS):
                            if (i < len(grid) and j < len(grid[i]) and 
                                grid[i][j] is not None):
                                return (i, j)
                return (0, 0)
            except:
                return (0, 0)
    
    def calculate_column_potential(self, grid, col_idx):
        """Calcule le potentiel d'une colonne avec analyse dominante"""
        try:
            if not grid or col_idx >= GRID_COLS or col_idx < 0:
                return 0.5
            
            column = []
            for row in range(GRID_ROWS):
                if (row < len(grid) and col_idx < len(grid[row]) and 
                    grid[row][col_idx] is not None):
                    column.append(grid[row][col_idx])
                else:
                    return 0.5
            
            revealed_values = []
            unrevealed_count = 0
            
            for card in column:
                if self.safe_is_revealed(card):
                    revealed_values.append(self.safe_get_card_value(card))
                else:
                    unrevealed_count += 1
            
            if len(revealed_values) == 0:
                return 0.6  # Potentiel légèrement positif par défaut
            
            # Bonus majeur pour colonnes identiques en formation
            if len(set(revealed_values)) == 1 and unrevealed_count > 0:
                base_value = revealed_values[0]
                if base_value <= 2:  # Excellente colonne
                    return 0.95 + (unrevealed_count * 0.05)
                elif base_value <= 5:  # Bonne colonne
                    return 0.8 + (unrevealed_count * 0.03)
                else:  # Colonne acceptable
                    return 0.6 + (unrevealed_count * 0.02)
            
            # Pénalité pour colonnes très hétérogènes
            if len(revealed_values) > 1:
                value_range = max(revealed_values) - min(revealed_values)
                if value_range > 6:
                    return 0.15  # Très mauvais potentiel
                elif value_range > 3:
                    return 0.35  # Potentiel faible
            
            # Calcul basé sur la moyenne avec ajustement agressif
            avg_value = sum(revealed_values) / len(revealed_values)
            base_potential = max(0.2, 1.0 - (avg_value / 10.0))  # Plus optimiste
            
            # Bonus pour les bonnes moyennes
            if avg_value <= 1:
                base_potential *= 1.3
            elif avg_value <= 3:
                base_potential *= 1.1
            
            return min(base_potential, 0.95)
        except Exception as e:
            return 0.5  # Fallback neutre
    
    def choose_reveal(self, grid):
        """Choix de révélation avec stratégie dominante"""
        try:
            if not grid:
                return None
            
            # Informations contextuelles
            all_revealed = self.collect_all_revealed_cards(grid, [])
            game_phase = self.analyze_game_phase(grid, [])
            
            best_position = None
            best_score = float('-inf')
            
            # Analyser chaque colonne pour la stratégie optimale
            for col in range(GRID_COLS):
                column_cards = []
                revealed_count = 0
                unrevealed_positions = []
                
                # Vérifier la validité de la colonne
                valid_column = True
                for row in range(GRID_ROWS):
                    if (row >= len(grid) or not grid[row] or 
                        col >= len(grid[row]) or grid[row][col] is None):
                        valid_column = False
                        break
                    column_cards.append(grid[row][col])
                
                if not valid_column:
                    continue
                
                # Analyser les cartes de la colonne
                for row in range(GRID_ROWS):
                    card = column_cards[row]
                    if self.safe_is_revealed(card):
                        revealed_count += 1
                    else:
                        unrevealed_positions.append((row, col))
                
                if not unrevealed_positions:
                    continue
                
                # Score basé sur le potentiel de la colonne
                column_potential = self.calculate_column_potential(grid, col)
                
                # Stratégie agressive : plus de bonus pour les colonnes à compléter
                completeness_bonus = revealed_count * 3  # Bonus renforcé
                
                # Bonus de phase
                phase_multiplier = 1.0
                if game_phase in ["endgame", "final"]:
                    phase_multiplier = 1.5  # Plus agressif en fin de partie
                elif game_phase == "opening":
                    phase_multiplier = 1.2  # Exploration active en ouverture
                
                score = (column_potential * 15 + completeness_bonus) * phase_multiplier
                
                # Priorité aux colonnes avec des cartes révélées de faible valeur
                revealed_values = []
                for card in column_cards:
                    if self.safe_is_revealed(card):
                        revealed_values.append(self.safe_get_card_value(card))
                
                if revealed_values:
                    avg_revealed = sum(revealed_values) / len(revealed_values)
                    if avg_revealed <= 2:
                        score *= 1.4  # Gros bonus pour les bonnes colonnes
                    elif avg_revealed <= 4:
                        score *= 1.2  # Bonus modéré
                
                if score > best_score:
                    best_score = score
                    # Choix intelligent dans la colonne
                    if len(unrevealed_positions) == 1:
                        best_position = unrevealed_positions[0]
                    else:
                        # Privilégier les coins et bords
                        priority_positions = []
                        for r, c in unrevealed_positions:
                            if (r == 0 or r == GRID_ROWS-1) and (c == 0 or c == GRID_COLS-1):
                                priority_positions.append((r, c))
                        
                        if priority_positions:
                            best_position = random.choice(priority_positions)
                        else:
                            best_position = random.choice(unrevealed_positions)
            
            # Stratégie de fallback
            if best_position is None:
                all_unrevealed = []
                for i in range(min(GRID_ROWS, len(grid))):
                    if i < len(grid) and grid[i]:
                        for j in range(min(GRID_COLS, len(grid[i]))):
                            if (j < len(grid[i]) and grid[i][j] is not None and 
                                not self.safe_is_revealed(grid[i][j])):
                                all_unrevealed.append((i, j))
                
                if all_unrevealed:
                    # Stratégie finale : privilégier les coins
                    corner_positions = [(r, c) for r, c in all_unrevealed 
                                      if (r == 0 or r == GRID_ROWS-1) and (c == 0 or c == GRID_COLS-1)]
                    
                    if corner_positions:
                        best_position = random.choice(corner_positions)
                    else:
                        best_position = random.choice(all_unrevealed)
            
            return best_position
            
        except Exception as e:
            # Récupération d'urgence
            try:
                for i in range(GRID_ROWS):
                    for j in range(GRID_COLS):
                        if (i < len(grid) and j < len(grid[i]) and 
                            grid[i][j] is not None and 
                            not self.safe_is_revealed(grid[i][j])):
                            return (i, j)
                return None
            except:
                return None
    
    def get_performance_stats(self):
        """Retourne les statistiques de performance"""
        if self.decisions_count == 0:
            return {"aggression_rate": 0.0, "decisions_made": 0}
        
        return {
            "aggression_rate": self.aggressive_decisions / self.decisions_count,
            "decisions_made": self.decisions_count,
            "aggressive_decisions": self.aggressive_decisions
        } 