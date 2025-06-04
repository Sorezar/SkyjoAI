import random
import math
from collections import Counter
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class AdvancedAI(BaseAI):
    """
    AdvancedAI - Version améliorée d'InitialAI avec:
    - Calculs probabilistes avancés
    - Analyse de colonnes et stratégies adaptatives
    - Gestion dynamique des risques
    - Optimisation basée sur l'état du jeu
    """
    
    def __init__(self):
        # Paramètres ajustables
        self.conservative_threshold = 0.7  # Seuil de prudence
        self.aggressive_bonus = 2.0        # Bonus pour jeu agressif
        self.column_synergy_bonus = 3.0    # Bonus synergie colonnes
        
    def initial_flip(self):
        """Stratégie initiale optimisée : coins et centre"""
        positions = [
            [0, 0], [0, GRID_COLS-1],  # Coins supérieurs
            [GRID_ROWS-1, 0], [GRID_ROWS-1, GRID_COLS-1],  # Coins inférieurs
            [GRID_ROWS//2, GRID_COLS//2]  # Centre
        ]
        return random.sample(positions, 2)
    
    def estimate_card_probabilities(self, revealed_cards):
        """Estime les probabilités des cartes restantes dans le deck"""
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
        
        return {value: count/total_remaining for value, count in card_counts.items()}
    
    def calculate_expected_unrevealed_value(self, grid, probabilities):
        """Calcule la valeur attendue des cartes non révélées"""
        if not probabilities:
            return 2.0  # Valeur par défaut
        
        expected_value = sum(value * prob for value, prob in probabilities.items())
        
        # Ajustement basé sur le contexte de la grille
        revealed_values = [card.value for row in grid for card in row if card.revealed]
        if revealed_values:
            avg_revealed = sum(revealed_values) / len(revealed_values)
            # Si nos cartes révélées sont mauvaises, on espère mieux dans les cachées
            if avg_revealed > 4:
                expected_value *= 0.8
            elif avg_revealed < 2:
                expected_value *= 1.2
        
        return expected_value
    
    def analyze_game_phase(self, grid, other_p_grids):
        """Détermine la phase de jeu (début, milieu, fin)"""
        total_revealed = sum(
            sum(1 for row in g for card in row if card.revealed)
            for g in [grid] + other_p_grids
        )
        total_cards = len([grid] + other_p_grids) * GRID_ROWS * GRID_COLS
        reveal_ratio = total_revealed / total_cards
        
        if reveal_ratio < 0.3:
            return "early"
        elif reveal_ratio < 0.7:
            return "mid"
        else:
            return "late"
    
    def calculate_column_potential(self, grid, col_idx):
        """Calcule le potentiel d'optimisation d'une colonne"""
        try:
            # Vérification défensive
            if not grid or col_idx >= GRID_COLS or col_idx < 0:
                return 0.5
            
            column = []
            for row in range(GRID_ROWS):
                if (row < len(grid) and col_idx < len(grid[row]) and 
                    grid[row][col_idx] is not None):
                    column.append(grid[row][col_idx])
                else:
                    return 0.5  # Colonne invalide
            
            revealed_values = []
            unrevealed_count = 0
            
            for card in column:
                if hasattr(card, 'revealed') and hasattr(card, 'value'):
                    if card.revealed:
                        revealed_values.append(card.value)
                    else:
                        unrevealed_count += 1
        except (IndexError, AttributeError, TypeError):
            return 0.5
        
        if len(revealed_values) == 0:
            return 0.5  # Potentiel neutre
        
        # Bonus si on peut former une colonne identique
        if len(set(revealed_values)) == 1 and unrevealed_count > 0:
            return 0.9 + (unrevealed_count * 0.05)
        
        # Malus pour colonnes avec valeurs très différentes
        if len(revealed_values) > 1:
            value_range = max(revealed_values) - min(revealed_values)
            if value_range > 8:
                return 0.1
        
        # Calcul basé sur la moyenne des valeurs révélées
        avg_value = sum(revealed_values) / len(revealed_values)
        return max(0.1, 1.0 - (avg_value / 12.0))
    
    def estimate_opponent_score(self, opponent_grid, probabilities):
        """Estime le score probable d'un adversaire"""
        try:
            # Vérifications défensives
            if not opponent_grid or len(opponent_grid) == 0:
                return 50  # Score estimé par défaut
            
            revealed_score = 0
            unrevealed_count = 0
            
            for row_idx, row in enumerate(opponent_grid):
                if row_idx >= GRID_ROWS:
                    break
                if not row or len(row) == 0:
                    continue
                    
                for col_idx, card in enumerate(row):
                    if col_idx >= GRID_COLS:
                        break
                    if card is not None and hasattr(card, 'revealed') and hasattr(card, 'value'):
                        if card.revealed:
                            revealed_score += card.value
                        else:
                            unrevealed_count += 1
            
        except (IndexError, AttributeError, TypeError):
            return 50  # Score estimé par défaut
        
        expected_unrevealed = self.calculate_expected_unrevealed_value(opponent_grid, probabilities)
        estimated_total = revealed_score + (unrevealed_count * expected_unrevealed)
        
        # Ajustement pessimiste (on suppose qu'ils jouent bien)
        return estimated_total * 0.9
    
    def choose_source(self, grid, discard, other_p_grids):
        """Choix de source avec analyse probabiliste avancée"""
        if not discard:
            return 'P'
        
        discard_value = discard[-1].value if discard and len(discard) > 0 else 5
        
        # Collecte des cartes révélées pour estimation probabiliste
        all_revealed = []
        try:
            # Notre grille
            if grid:
                for row_idx, row in enumerate(grid):
                    if row_idx >= GRID_ROWS or not row:
                        continue
                    for col_idx, card in enumerate(row):
                        if col_idx >= GRID_COLS:
                            break
                        if (card is not None and hasattr(card, 'revealed') and 
                            hasattr(card, 'value') and card.revealed):
                            all_revealed.append(card.value)
            
            # Grilles adversaires
            if other_p_grids:
                for opp_grid in other_p_grids:
                    if not opp_grid or len(opp_grid) == 0:
                        continue
                    for row_idx, row in enumerate(opp_grid):
                        if row_idx >= GRID_ROWS or not row:
                            continue
                        for col_idx, card in enumerate(row):
                            if col_idx >= GRID_COLS:
                                break
                            if (card is not None and hasattr(card, 'revealed') and 
                                hasattr(card, 'value') and card.revealed):
                                all_revealed.append(card.value)
            
            # Défausse
            if discard:
                for card in discard:
                    if card is not None and hasattr(card, 'value'):
                        all_revealed.append(card.value)
                        
        except (IndexError, AttributeError, TypeError):
            # En cas d'erreur, continuer avec les cartes collectées
            pass
        
        probabilities = self.estimate_card_probabilities(all_revealed)
        game_phase = self.analyze_game_phase(grid, other_p_grids)
        
        # Logique adaptée à la phase de jeu
        if game_phase == "early":
            # En début de jeu, être plus sélectif
            return 'D' if discard_value <= 2 else 'P'
        elif game_phase == "mid":
            # En milieu de jeu, équilibrer opportunités et sécurité
            expected_draw = sum(value * prob for value, prob in probabilities.items())
            threshold = 4 if discard_value < expected_draw else 6
            return 'D' if discard_value <= threshold else 'P'
        else:
            # En fin de jeu, être plus agressif
            our_score_estimate = self.estimate_my_final_score(grid, probabilities)
            opponent_estimates = []
            
            if other_p_grids:
                for opp_grid in other_p_grids:
                    if opp_grid and len(opp_grid) > 0:
                        opponent_estimates.append(self.estimate_opponent_score(opp_grid, probabilities))
            
            if not opponent_estimates:
                opponent_estimates = [50]  # Score par défaut
            
            if our_score_estimate > min(opponent_estimates):
                # On est en retard, prendre plus de risques
                return 'D' if discard_value <= 7 else 'P'
            else:
                # On mène, être conservateur
                return 'D' if discard_value <= 3 else 'P'
    
    def estimate_my_final_score(self, grid, probabilities):
        """Estime notre score final probable"""
        try:
            if not grid or len(grid) == 0:
                return 50  # Score par défaut
            
            revealed_score = 0
            unrevealed_count = 0
            
            for row_idx, row in enumerate(grid):
                if row_idx >= GRID_ROWS or not row:
                    continue
                for col_idx, card in enumerate(row):
                    if col_idx >= GRID_COLS:
                        break
                    if card is not None and hasattr(card, 'revealed') and hasattr(card, 'value'):
                        if card.revealed:
                            revealed_score += card.value
                        else:
                            unrevealed_count += 1
            
        except (IndexError, AttributeError, TypeError):
            return 50  # Score par défaut
        
        expected_unrevealed = self.calculate_expected_unrevealed_value(grid, probabilities)
        
        return revealed_score + (unrevealed_count * expected_unrevealed)
    
    def choose_keep(self, card, grid, other_p_grids):
        """Décision de garde avec analyse contextuelle"""
        card_value = card.value
        
        # Collecte des informations pour l'analyse
        all_revealed = []
        try:
            # Notre grille
            if grid:
                for row_idx, row in enumerate(grid):
                    if row_idx >= GRID_ROWS or not row:
                        continue
                    for col_idx, c in enumerate(row):
                        if col_idx >= GRID_COLS:
                            break
                        if (c is not None and hasattr(c, 'revealed') and 
                            hasattr(c, 'value') and c.revealed):
                            all_revealed.append(c.value)
            
            # Grilles adversaires
            if other_p_grids:
                for opp_grid in other_p_grids:
                    if not opp_grid or len(opp_grid) == 0:
                        continue
                    for row_idx, row in enumerate(opp_grid):
                        if row_idx >= GRID_ROWS or not row:
                            continue
                        for col_idx, c in enumerate(row):
                            if col_idx >= GRID_COLS:
                                break
                            if (c is not None and hasattr(c, 'revealed') and 
                                hasattr(c, 'value') and c.revealed):
                                all_revealed.append(c.value)
        except (IndexError, AttributeError, TypeError):
            pass
        
        probabilities = self.estimate_card_probabilities(all_revealed)
        game_phase = self.analyze_game_phase(grid, other_p_grids)
        
        # Cas spécial : dernière carte
        try:
            unrevealed_count = 0
            if grid:
                for row_idx, row in enumerate(grid):
                    if row_idx >= GRID_ROWS or not row:
                        continue
                    for col_idx, c in enumerate(row):
                        if col_idx >= GRID_COLS:
                            break
                        if (c is not None and hasattr(c, 'revealed') and 
                            not c.revealed):
                            unrevealed_count += 1
            
            if unrevealed_count <= 1:
                our_final_score = 0
                if grid:
                    for row_idx, row in enumerate(grid):
                        if row_idx >= GRID_ROWS or not row:
                            continue
                        for col_idx, c in enumerate(row):
                            if col_idx >= GRID_COLS:
                                break
                            if (c is not None and hasattr(c, 'revealed') and 
                                hasattr(c, 'value') and c.revealed):
                                our_final_score += c.value
                
                our_final_score += card_value
                
                opponent_estimates = []
                if other_p_grids:
                    for opp_grid in other_p_grids:
                        if opp_grid and len(opp_grid) > 0:
                            opponent_estimates.append(self.estimate_opponent_score(opp_grid, probabilities))
                
                if opponent_estimates:
                    return our_final_score <= min(opponent_estimates) * 1.05  # Marge de 5%
                    
        except (IndexError, AttributeError, TypeError):
            pass
        
        # Logique adaptée à la phase
        if game_phase == "early":
            return card_value <= 3
        elif game_phase == "mid":
            expected_replacement = sum(value * prob for value, prob in probabilities.items())
            return card_value < expected_replacement
        else:
            # En fin de jeu, plus agressif
            return card_value <= 5
    
    def choose_position(self, card, grid, other_p_grids):
        """Choix de position avec optimisation multi-critères"""
        card_value = card.value
        best_position = None
        best_score = float('-inf')
        
        # Collecte des positions disponibles avec vérifications défensives
        revealed_positions = []
        unrevealed_positions = []
        
        try:
            if grid:
                for i in range(min(len(grid), GRID_ROWS)):
                    if not grid[i] or len(grid[i]) == 0:
                        continue
                    for j in range(min(len(grid[i]), GRID_COLS)):
                        if grid[i][j] is not None and hasattr(grid[i][j], 'revealed'):
                            if grid[i][j].revealed:
                                revealed_positions.append((i, j))
                            else:
                                unrevealed_positions.append((i, j))
        except (IndexError, AttributeError, TypeError):
            # En cas d'erreur, retour sécurisé
            return (0, 0) if grid and len(grid) > 0 and len(grid[0]) > 0 else None
        
        # Évaluer les positions révélées
        for row, col in revealed_positions:
            try:
                if (row < len(grid) and col < len(grid[row]) and 
                    grid[row][col] is not None and hasattr(grid[row][col], 'value')):
                    
                    current_value = grid[row][col].value
                    improvement = current_value - card_value
                    
                    if improvement <= 0:
                        continue
                    
                    score = improvement
                    
                    # Bonus synergie colonne
                    column_potential = self.calculate_column_potential(grid, col)
                    score += column_potential * self.column_synergy_bonus
                    
                    # Bonus position stratégique (coins et bords)
                    if (row == 0 or row == GRID_ROWS-1) and (col == 0 or col == GRID_COLS-1):
                        score += 1.5  # Bonus coin
                    elif row == 0 or row == GRID_ROWS-1 or col == 0 or col == GRID_COLS-1:
                        score += 0.5  # Bonus bord
                    
                    if score > best_score:
                        best_score = score
                        best_position = (row, col)
            except (IndexError, AttributeError, TypeError):
                continue
        
        # Si aucune position révélée n'est bonne, évaluer les positions cachées
        if best_position is None and unrevealed_positions:
            try:
                # Privilégier les colonnes avec le meilleur potentiel
                column_scores = {}
                for col in range(GRID_COLS):
                    column_scores[col] = self.calculate_column_potential(grid, col)
                
                # Choisir dans la meilleure colonne disponible
                available_columns = list(set(col for row, col in unrevealed_positions))
                if available_columns:
                    best_col = max(available_columns, key=lambda c: column_scores[c])
                    col_positions = [(r, c) for r, c in unrevealed_positions if c == best_col]
                    if col_positions:
                        best_position = random.choice(col_positions)
            except (IndexError, AttributeError, TypeError, ValueError):
                if unrevealed_positions:
                    best_position = random.choice(unrevealed_positions)
        
        return best_position or (random.choice(unrevealed_positions) if unrevealed_positions else None)
    
    def choose_reveal(self, grid):
        """Choix de révélation avec stratégie de colonnes"""
        best_position = None
        best_score = float('-inf')
        
        try:
            if not grid or len(grid) == 0:
                return None
                
            # Analyser chaque colonne
            for col in range(GRID_COLS):
                column = []
                revealed_count = 0
                unrevealed_positions = []
                
                # Vérifier que la colonne existe dans toutes les lignes
                valid_column = True
                for row in range(GRID_ROWS):
                    if (row >= len(grid) or not grid[row] or 
                        col >= len(grid[row]) or grid[row][col] is None):
                        valid_column = False
                        break
                    column.append(grid[row][col])
                
                if not valid_column:
                    continue
                
                # Compter les cartes révélées et positions non révélées
                for row in range(GRID_ROWS):
                    card = column[row]
                    if hasattr(card, 'revealed'):
                        if card.revealed:
                            revealed_count += 1
                        else:
                            unrevealed_positions.append((row, col))
                
                if not unrevealed_positions:
                    continue
                
                # Score basé sur le potentiel de la colonne
                column_potential = self.calculate_column_potential(grid, col)
                
                # Privilégier les colonnes avec plus de cartes révélées
                completeness_bonus = revealed_count * 2
                
                score = column_potential * 10 + completeness_bonus
                
                if score > best_score:
                    best_score = score
                    best_position = random.choice(unrevealed_positions)
            
            # Si aucune stratégie claire, révéler au hasard
            if best_position is None:
                all_unrevealed = []
                for i in range(min(len(grid), GRID_ROWS)):
                    if not grid[i] or len(grid[i]) == 0:
                        continue
                    for j in range(min(len(grid[i]), GRID_COLS)):
                        if (grid[i][j] is not None and hasattr(grid[i][j], 'revealed') and 
                            not grid[i][j].revealed):
                            all_unrevealed.append((i, j))
                
                best_position = random.choice(all_unrevealed) if all_unrevealed else None
                
        except (IndexError, AttributeError, TypeError):
            # En dernier recours, essayer de retourner une position sécurisée
            try:
                if grid and len(grid) > 0 and len(grid[0]) > 0:
                    return (0, 0)
            except:
                return None
        
        return best_position 