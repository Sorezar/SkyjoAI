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
                    if hasattr(card, 'value'):
                        all_revealed.append(card.value)
                    else:
                        all_revealed.append(card)
                        
        except (IndexError, AttributeError, TypeError):
            pass  # Continue avec les cartes collectées
        
        # Estimation probabiliste
        probabilities = self.estimate_card_probabilities(all_revealed)
        expected_deck_value = sum(value * prob for value, prob in probabilities.items()) if probabilities else 3.0
        
        # Phase de jeu
        game_phase = self.analyze_game_phase(grid, other_p_grids)
        
        # Facteurs de décision
        discard_threshold = 5.0
        if game_phase == "early":
            discard_threshold = 6.0  # Plus sélectif en début
        elif game_phase == "late":
            discard_threshold = 3.0  # Plus agressif en fin
        
        # Analyse de nos cartes révélées
        our_revealed_values = []
        try:
            if grid:
                for row in grid:
                    if not row:
                        continue
                    for card in row:
                        if (card is not None and hasattr(card, 'revealed') and 
                            hasattr(card, 'value') and card.revealed):
                            our_revealed_values.append(card.value)
        except (IndexError, AttributeError, TypeError):
            pass
        
        our_avg = sum(our_revealed_values) / len(our_revealed_values) if our_revealed_values else 4.0
        
        # Ajustement du seuil basé sur notre performance
        if our_avg > 5:
            discard_threshold -= 1.0  # Plus agressif si on a de mauvaises cartes
        elif our_avg < 2:
            discard_threshold += 1.0  # Plus conservateur si on va bien
        
        # Décision finale
        if discard_value <= discard_threshold:
            return 'D'
        else:
            return 'P'
    
    def estimate_my_final_score(self, grid, probabilities):
        """Estime notre score final probable"""
        try:
            if not grid:
                return 50
            
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
            return 50
        
        expected_unrevealed = self.calculate_expected_unrevealed_value(grid, probabilities)
        return revealed_score + (unrevealed_count * expected_unrevealed)
    
    def choose_keep(self, card, grid, other_p_grids):
        """Décision de garder une carte avec analyse avancée"""
        if not hasattr(card, 'value'):
            return False
        
        card_value = card.value
        
        # Collecte des cartes révélées
        all_revealed = []
        try:
            # Notre grille
            if grid:
                for row in grid:
                    if not row:
                        continue
                    for c in row:
                        if (c is not None and hasattr(c, 'revealed') and 
                            hasattr(c, 'value') and c.revealed):
                            all_revealed.append(c.value)
            
            # Grilles adversaires
            if other_p_grids:
                for opp_grid in other_p_grids:
                    if not opp_grid:
                        continue
                    for row in opp_grid:
                        if not row:
                            continue
                        for c in row:
                            if (c is not None and hasattr(c, 'revealed') and 
                                hasattr(c, 'value') and c.revealed):
                                all_revealed.append(c.value)
                                
        except (IndexError, AttributeError, TypeError):
            pass
        
        # Estimation probabiliste
        probabilities = self.estimate_card_probabilities(all_revealed)
        
        # Phase de jeu
        game_phase = self.analyze_game_phase(grid, other_p_grids)
        
        # Seuils adaptatifs
        if game_phase == "early":
            keep_threshold = 3.0
        elif game_phase == "mid":
            keep_threshold = 4.0
        else:  # late
            keep_threshold = 5.0
        
        # Analyse de notre performance actuelle
        our_revealed_values = []
        try:
            if grid:
                for row in grid:
                    if not row:
                        continue
                    for c in row:
                        if (c is not None and hasattr(c, 'revealed') and 
                            hasattr(c, 'value') and c.revealed):
                            our_revealed_values.append(c.value)
        except (IndexError, AttributeError, TypeError):
            pass
        
        our_avg = sum(our_revealed_values) / len(our_revealed_values) if our_revealed_values else 4.0
        
        # Ajustement du seuil
        if our_avg > 5:
            keep_threshold += 1.0  # Plus sélectif si on a de mauvaises cartes
        elif our_avg < 2:
            keep_threshold -= 0.5  # Moins sélectif si on va bien
        
        # Estimation des scores adversaires
        opponent_estimates = []
        if other_p_grids:
            for opp_grid in other_p_grids:
                opp_score = self.estimate_opponent_score(opp_grid, probabilities)
                opponent_estimates.append(opp_score)
        
        best_opponent = min(opponent_estimates) if opponent_estimates else 25
        my_estimate = self.estimate_my_final_score(grid, probabilities)
        
        # Si on est en retard, être plus agressif
        if my_estimate > best_opponent + 5:
            keep_threshold += 1.0
        elif my_estimate < best_opponent - 5:
            keep_threshold -= 1.0
        
        return card_value <= keep_threshold
    
    def choose_position(self, card, grid, other_p_grids):
        """Choix de position avec optimisation avancée"""
        if not hasattr(card, 'value') or not grid:
            return (0, 0)
        
        card_value = card.value
        best_position = (0, 0)
        best_score = float('-inf')
        
        # Collecte des cartes révélées pour estimation probabiliste
        all_revealed = []
        try:
            for row in grid:
                if not row:
                    continue
                for c in row:
                    if (c is not None and hasattr(c, 'revealed') and 
                        hasattr(c, 'value') and c.revealed):
                        all_revealed.append(c.value)
            
            if other_p_grids:
                for opp_grid in other_p_grids:
                    if not opp_grid:
                        continue
                    for row in opp_grid:
                        if not row:
                            continue
                        for c in row:
                            if (c is not None and hasattr(c, 'revealed') and 
                                hasattr(c, 'value') and c.revealed):
                                all_revealed.append(c.value)
                                
        except (IndexError, AttributeError, TypeError):
            pass
        
        probabilities = self.estimate_card_probabilities(all_revealed)
        
        # Évaluer chaque position possible
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                try:
                    if (i >= len(grid) or j >= len(grid[i]) or 
                        grid[i][j] is None or not hasattr(grid[i][j], 'value')):
                        continue
                    
                    current_card = grid[i][j]
                    improvement = current_card.value - card_value
                    
                    # Score de base = amélioration directe
                    position_score = improvement
                    
                    # Bonus pour les colonnes
                    column_potential = self.calculate_column_potential(grid, j)
                    position_score += column_potential * self.column_synergy_bonus
                    
                    # Bonus pour retirer des cartes très mauvaises
                    if current_card.value >= 8:
                        position_score += self.aggressive_bonus
                    
                    # Bonus pour placer des cartes très bonnes
                    if card_value <= 0:
                        position_score += 2.0
                    
                    # Pénalité pour remplacer des cartes cachées utiles
                    if not current_card.revealed:
                        expected_hidden = self.calculate_expected_unrevealed_value(grid, probabilities)
                        hidden_penalty = max(0, expected_hidden - card_value)
                        position_score -= hidden_penalty * 0.5
                    
                    if position_score > best_score:
                        best_score = position_score
                        best_position = (i, j)
                        
                except (IndexError, AttributeError, TypeError):
                    continue
        
        return best_position
    
    def choose_reveal(self, grid):
        """Choix de révélation avec stratégie optimisée"""
        if not grid:
            return (0, 0)
        
        best_position = (0, 0)
        best_score = float('-inf')
        
        # Analyser chaque position non révélée
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                try:
                    if (i >= len(grid) or j >= len(grid[i]) or 
                        grid[i][j] is None or not hasattr(grid[i][j], 'revealed')):
                        continue
                    
                    card = grid[i][j]
                    if card.revealed:
                        continue  # Déjà révélée
                    
                    # Score de base : position stratégique
                    position_score = 0
                    
                    # Bonus pour les coins (positions stratégiques)
                    if (i == 0 or i == GRID_ROWS-1) and (j == 0 or j == GRID_COLS-1):
                        position_score += 2.0
                    
                    # Bonus pour le centre
                    if i == GRID_ROWS//2 and j == GRID_COLS//2:
                        position_score += 1.5
                    
                    # Bonus pour compléter l'analyse d'une colonne
                    column_revealed = 0
                    for row_idx in range(GRID_ROWS):
                        if (row_idx < len(grid) and j < len(grid[row_idx]) and 
                            grid[row_idx][j] is not None and 
                            hasattr(grid[row_idx][j], 'revealed') and 
                            grid[row_idx][j].revealed):
                            column_revealed += 1
                    
                    if column_revealed >= GRID_ROWS - 1:  # Dernière carte de la colonne
                        position_score += 3.0
                    elif column_revealed >= GRID_ROWS // 2:  # Majorité révélée
                        position_score += 1.0
                    
                    # Préférer révéler près des cartes déjà révélées
                    adjacent_revealed = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if (0 <= ni < GRID_ROWS and 0 <= nj < GRID_COLS and 
                                ni < len(grid) and nj < len(grid[ni]) and 
                                grid[ni][nj] is not None and 
                                hasattr(grid[ni][nj], 'revealed') and 
                                grid[ni][nj].revealed):
                                adjacent_revealed += 1
                    
                    position_score += adjacent_revealed * 0.5
                    
                    if position_score > best_score:
                        best_score = position_score
                        best_position = (i, j)
                        
                except (IndexError, AttributeError, TypeError):
                    continue
        
        return best_position 