import random
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from ai.base import BaseAI
from config.config import GRID_ROWS, GRID_COLS

class MachineLearningAI(BaseAI):
    """
    IA basée sur Random Forest qui apprend à partir de données de parties.
    Utilise des features ingénierie pour capturer l'état du jeu et 
    prédit les meilleures actions à partir d'exemples d'entraînement.
    """
    
    def __init__(self, model_path="ml_models/skyjo_rf_model.pkl"):
        self.model_path = model_path
        self.models = {
            'source': None,
            'keep': None,
            'position': None,
            'reveal': None
        }
        self.training_data = {
            'source': {'X': [], 'y': []},
            'keep': {'X': [], 'y': []},
            'position': {'X': [], 'y': []},
            'reveal': {'X': [], 'y': []}
        }
        self.load_models()
        
    def save_models(self):
        """Sauvegarde les modèles entraînés"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.models, f)
    
    def load_models(self):
        """Charge les modèles pré-entraînés s'ils existent"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.models = pickle.load(f)
            except:
                print("Erreur lors du chargement des modèles ML, utilisation de modèles vides")
    
    def extract_game_features(self, grid, discard, other_p_grids):
        """Extrait les features de l'état du jeu pour l'apprentissage"""
        features = []
        
        # Vérification défensive de la grille principale
        if not grid or len(grid) == 0:
            # Retourner des features par défaut si la grille n'est pas initialisée
            return np.zeros(101)  # Dimension attendue des features (24+6+60+1+2+8)
        
        # Vérifier que la grille a les bonnes dimensions
        for i in range(len(grid)):
            if not grid[i] or len(grid[i]) == 0:
                return np.zeros(101)
        
        # Features de notre grille (24 features : 3*4*2)
        for i in range(GRID_ROWS):
            for j in range(GRID_COLS):
                try:
                    if (i < len(grid) and j < len(grid[i]) and 
                        grid[i][j] is not None):
                        if hasattr(grid[i][j], 'revealed') and grid[i][j].revealed:
                            features.extend([1, grid[i][j].value])  # [is_revealed, value]
                        else:
                            features.extend([0, 0])  # [is_revealed, placeholder_value]
                    else:
                        features.extend([0, 0])  # Position non initialisée
                except (IndexError, AttributeError):
                    features.extend([0, 0])  # Gestion sécurisée des erreurs
        
        # Features de la défausse (6 features)
        try:
            if discard and len(discard) > 0:
                # Prendre les 5 dernières cartes de la défausse
                recent_cards = discard[-5:] if len(discard) >= 5 else discard
                discard_values = []
                
                for card in recent_cards:
                    if card is not None and hasattr(card, 'value'):
                        discard_values.append(card.value)
                    else:
                        discard_values.append(0)
                
                # Padding pour avoir exactement 5 valeurs
                while len(discard_values) < 5:
                    discard_values.append(0)
                
                features.extend(discard_values[:5])  # Exactement 5 valeurs
                features.append(len(discard))  # Taille de la défausse
            else:
                features.extend([0] * 6)  # Pas de défausse : 5 valeurs + taille
        except (IndexError, AttributeError):
            features.extend([0] * 6)
        
        # Features des adversaires (60 features : 3 adversaires * 20 features chacun)
        processed_opponents = 0
        
        if other_p_grids:
            for opp_idx, opp_grid in enumerate(other_p_grids):
                if processed_opponents >= 3:  # Maximum 3 adversaires
                    break
                    
                try:
                    # Vérifier que la grille adversaire est valide
                    if not opp_grid or len(opp_grid) == 0:
                        features.extend([0] * 20)
                        processed_opponents += 1
                        continue
                    
                    # Vérifier les dimensions de la grille adversaire
                    valid_grid = True
                    for row in opp_grid:
                        if not row or len(row) == 0:
                            valid_grid = False
                            break
                    
                    if not valid_grid:
                        features.extend([0] * 20)
                        processed_opponents += 1
                        continue
                    
                    # Score révélé de l'adversaire
                    opp_revealed_score = 0
                    opp_revealed_count = 0
                    
                    for row_idx, row in enumerate(opp_grid):
                        if row_idx >= GRID_ROWS:  # Limiter aux dimensions attendues
                            break
                        for col_idx, card in enumerate(row):
                            if col_idx >= GRID_COLS:  # Limiter aux dimensions attendues
                                break
                            try:
                                if (card is not None and hasattr(card, 'revealed') and 
                                    hasattr(card, 'value') and card.revealed):
                                    opp_revealed_score += card.value
                                    opp_revealed_count += 1
                            except (AttributeError, TypeError):
                                continue
                    
                    features.append(opp_revealed_score)
                    features.append(opp_revealed_count)
                    
                    # Analyse par colonne de l'adversaire (16 features : 4 colonnes * 4 features)
                    for col in range(GRID_COLS):
                        col_revealed = 0
                        col_sum = 0
                        
                        try:
                            # Vérifier que la colonne existe
                            for row_idx in range(min(len(opp_grid), GRID_ROWS)):
                                if (col < len(opp_grid[row_idx]) and 
                                    opp_grid[row_idx][col] is not None):
                                    card = opp_grid[row_idx][col]
                                    if (hasattr(card, 'revealed') and hasattr(card, 'value') and 
                                        card.revealed):
                                        col_revealed += 1
                                        col_sum += card.value
                        except (IndexError, AttributeError, TypeError):
                            pass
                        
                        features.extend([col_revealed, col_sum])
                    
                    # Features statistiques adversaire (2 features)
                    if opp_revealed_count > 0:
                        avg_score = opp_revealed_score / opp_revealed_count
                        reveal_ratio = opp_revealed_count / (GRID_ROWS * GRID_COLS)
                        features.extend([avg_score, reveal_ratio])
                    else:
                        features.extend([0, 0])
                    
                    processed_opponents += 1
                    
                except (IndexError, AttributeError, TypeError):
                    # En cas d'erreur, ajouter des features par défaut
                    features.extend([0] * 20)
                    processed_opponents += 1
        
        # Padding si moins de 3 adversaires
        while processed_opponents < 3:
            features.extend([0] * 20)
            processed_opponents += 1
        
        # Features globales du jeu (10 features)
        try:
            total_revealed = 0
            total_cards = 0
            all_grids = [grid]
            
            if other_p_grids:
                all_grids.extend(other_p_grids[:3])  # Maximum 3 adversaires
            
            for g in all_grids:
                if g and len(g) > 0:
                    for row_idx, row in enumerate(g):
                        if row_idx >= GRID_ROWS:
                            break
                        if row and len(row) > 0:
                            for col_idx, card in enumerate(row):
                                if col_idx >= GRID_COLS:
                                    break
                                total_cards += 1
                                try:
                                    if (card is not None and hasattr(card, 'revealed') and 
                                        card.revealed):
                                        total_revealed += 1
                                except (AttributeError, TypeError):
                                    pass
            
            game_progress = total_revealed / total_cards if total_cards > 0 else 0
            features.append(game_progress)
            
        except (IndexError, AttributeError, TypeError):
            features.append(0.0)
        
        # Notre score actuel (2 features)
        try:
            our_score = 0
            our_revealed = 0
            
            for i in range(min(len(grid), GRID_ROWS)):
                for j in range(min(len(grid[i]), GRID_COLS)):
                    try:
                        card = grid[i][j]
                        if (card is not None and hasattr(card, 'revealed') and 
                            hasattr(card, 'value')):
                            if card.revealed:
                                our_score += card.value
                                our_revealed += 1
                    except (IndexError, AttributeError, TypeError):
                        pass
            
            features.extend([our_score, our_revealed])
            
        except (IndexError, AttributeError, TypeError):
            features.extend([0, 0])
        
        # Analyse de nos colonnes (8 features : 4 colonnes * 2 features)
        try:
            for col in range(GRID_COLS):
                col_revealed = 0
                col_sum = 0
                
                for row in range(min(len(grid), GRID_ROWS)):
                    try:
                        if (col < len(grid[row]) and grid[row][col] is not None):
                            card = grid[row][col]
                            if (hasattr(card, 'revealed') and hasattr(card, 'value') and 
                                card.revealed):
                                col_revealed += 1
                                col_sum += card.value
                    except (IndexError, AttributeError, TypeError):
                        pass
                
                features.extend([col_revealed, col_sum])
                
        except (IndexError, AttributeError, TypeError):
            # En cas d'erreur, ajouter des features par défaut
            features.extend([0] * 8)
        
        # S'assurer que nous avons exactement le bon nombre de features
        expected_features = 24 + 6 + 60 + 1 + 2 + 8  # = 101 features
        
        while len(features) < expected_features:
            features.append(0.0)
        
        if len(features) > expected_features:
            features = features[:expected_features]
        
        return np.array(features, dtype=np.float32)
    
    def add_training_example(self, action_type, features, action):
        """Ajoute un exemple d'entraînement"""
        self.training_data[action_type]['X'].append(features)
        self.training_data[action_type]['y'].append(action)
    
    def train_models(self, min_samples=100):
        """Entraîne les modèles Random Forest"""
        print("🎯 Entraînement des modèles Machine Learning...")
        
        for action_type in self.models.keys():
            if len(self.training_data[action_type]['X']) >= min_samples:
                X = np.array(self.training_data[action_type]['X'])
                y = np.array(self.training_data[action_type]['y'])
                
                # Division train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Entraînement Random Forest
                self.models[action_type] = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
                
                self.models[action_type].fit(X_train, y_train)
                
                # Évaluation
                y_pred = self.models[action_type].predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"✅ Modèle {action_type}: {len(X)} échantillons, précision: {accuracy:.3f}")
            else:
                print(f"❌ Pas assez d'échantillons pour {action_type}: {len(self.training_data[action_type]['X'])}")
        
        self.save_models()
    
    def initial_flip(self):
        """Stratégie initiale : coins optimaux"""
        return [[0, 0], [2, 3]]  # Coins opposés
    
    def choose_source(self, grid, discard, other_p_grids):
        """Choix de source avec prédiction ML"""
        if not discard:
            return 'P'
        
        features = self.extract_game_features(grid, discard, other_p_grids)
        
        if self.models['source'] is not None:
            try:
                prediction = self.models['source'].predict([features])[0]
                return 'D' if prediction == 1 else 'P'
            except:
                pass
        
        # Fallback : heuristique simple
        if discard and len(discard) > 0 and discard[-1] is not None:
            return 'D' if discard[-1].value <= 4 else 'P'
        else:
            return 'P'  # Si pas de défausse, piocher
    
    def choose_keep(self, card, grid, other_p_grids):
        """Décision de garde avec prédiction ML"""
        features = self.extract_game_features(grid, [], other_p_grids)
        # Ajouter la valeur de la carte piochée
        features = np.append(features, card.value)
        
        if self.models['keep'] is not None:
            try:
                prediction = self.models['keep'].predict([features])[0]
                return prediction == 1
            except:
                pass
        
        # Fallback : heuristique simple
        return card.value <= 4
    
    def choose_position(self, card, grid, other_p_grids):
        """Choix de position avec prédiction ML"""
        try:
            features = self.extract_game_features(grid, [], other_p_grids)
            features = np.append(features, card.value)
            
            available_positions = []
            
            # Collecte sécurisée des positions disponibles
            if grid:
                for i in range(min(len(grid), GRID_ROWS)):
                    if not grid[i] or len(grid[i]) == 0:
                        continue
                    for j in range(min(len(grid[i]), GRID_COLS)):
                        if grid[i][j] is not None and hasattr(grid[i][j], 'revealed'):
                            available_positions.append((i, j))
            
            if self.models['position'] is not None and available_positions:
                try:
                    # Encoder les positions comme indices uniques
                    position_indices = [i * GRID_COLS + j for i, j in available_positions]
                    best_pos_idx = 0
                    best_score = float('-inf')
                    
                    # Évaluer chaque position possible
                    for idx, (i, j) in enumerate(available_positions):
                        pos_features = np.append(features, [i, j])
                        try:
                            score = self.models['position'].predict_proba([pos_features])[0]
                            if len(score) > 1 and score[1] > best_score:  # Classe "bonne position"
                                best_score = score[1]
                                best_pos_idx = idx
                        except:
                            continue
                    
                    return available_positions[best_pos_idx]
                except:
                    pass
            
            # Fallback sécurisé : choisir la meilleure position révélée ou une au hasard
            revealed_positions = []
            unrevealed_positions = []
            
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
            
            if revealed_positions:
                try:
                    best_pos = revealed_positions[0]
                    best_improvement = float('-inf')
                    for pos in revealed_positions:
                        i, j = pos
                        if (i < len(grid) and j < len(grid[i]) and 
                            grid[i][j] is not None and hasattr(grid[i][j], 'value')):
                            improvement = grid[i][j].value - card.value
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_pos = pos
                    return best_pos
                except (IndexError, AttributeError, TypeError):
                    pass
            
            return random.choice(unrevealed_positions) if unrevealed_positions else (0, 0)
            
        except (IndexError, AttributeError, TypeError):
            # En cas d'erreur majeure, retour sécurisé
            return (0, 0)
    
    def choose_reveal(self, grid):
        """Choix de révélation avec prédiction ML"""
        try:
            features = self.extract_game_features(grid, [], [])
            
            unrevealed_positions = []
            
            # Collecte sécurisée des positions non révélées
            if grid:
                for i in range(min(len(grid), GRID_ROWS)):
                    if not grid[i] or len(grid[i]) == 0:
                        continue
                    for j in range(min(len(grid[i]), GRID_COLS)):
                        if (grid[i][j] is not None and hasattr(grid[i][j], 'revealed') and 
                            not grid[i][j].revealed):
                            unrevealed_positions.append((i, j))
            
            if self.models['reveal'] is not None and unrevealed_positions:
                try:
                    best_pos = unrevealed_positions[0]
                    best_score = float('-inf')
                    
                    for i, j in unrevealed_positions:
                        pos_features = np.append(features, [i, j])
                        try:
                            score = self.models['reveal'].predict_proba([pos_features])[0]
                            if len(score) > 1 and score[1] > best_score:
                                best_score = score[1]
                                best_pos = (i, j)
                        except:
                            continue
                    
                    return best_pos
                except:
                    pass
            
            # Fallback sécurisé : révéler dans la colonne la plus avancée
            column_progress = []
            for col in range(GRID_COLS):
                revealed_count = 0
                if grid:
                    for row in range(min(len(grid), GRID_ROWS)):
                        if (col < len(grid[row]) and grid[row][col] is not None and 
                            hasattr(grid[row][col], 'revealed') and grid[row][col].revealed):
                            revealed_count += 1
                column_progress.append((col, revealed_count))
            
            # Trier par progression décroissante
            column_progress.sort(key=lambda x: x[1], reverse=True)
            
            for col, _ in column_progress:
                col_unrevealed = []
                if grid:
                    for row in range(min(len(grid), GRID_ROWS)):
                        if (col < len(grid[row]) and grid[row][col] is not None and 
                            hasattr(grid[row][col], 'revealed') and not grid[row][col].revealed):
                            col_unrevealed.append((row, col))
                if col_unrevealed:
                    return random.choice(col_unrevealed)
            
            return random.choice(unrevealed_positions) if unrevealed_positions else (0, 0)
            
        except (IndexError, AttributeError, TypeError):
            # En cas d'erreur majeure, retour sécurisé
            return (0, 0)


def collect_training_data_from_initial_ai():
    """Collecte des données d'entraînement en observant InitialAI jouer"""
    from ai.initial import InitialAI
    from core.game import SkyjoGame, Scoreboard
    from core.player import Player
    
    print("🔄 Collecte de données d'entraînement à partir d'InitialAI...")
    
    ml_ai = MachineLearningAI()
    
    # Créer des joueurs pour la collecte
    players = [Player(i, f"InitialAI_{i}", InitialAI()) for i in range(4)]
    scoreboard = Scoreboard(players)
    
    successful_games = 0
    
    # Simuler des parties pour collecter des données
    for game_num in range(50):  # 50 parties d'entraînement
        try:
            game = SkyjoGame(players, scoreboard)
            
            game_steps = 0
            while not game.finished and game_steps < 1000:  # Limite de sécurité
                if not game.round_over:
                    current_player = game.players[game.current_player_index]
                    
                    # Vérifier que le joueur et sa grille sont valides
                    if (current_player and hasattr(current_player, 'grid') and 
                        current_player.grid and len(current_player.grid) > 0):
                        
                        grid = current_player.grid
                        other_grids = []
                        
                        # Collecter les grilles des autres joueurs de manière sécurisée
                        for i, p in enumerate(game.players):
                            if i != game.current_player_index:
                                if (p and hasattr(p, 'grid') and p.grid and 
                                    len(p.grid) > 0):
                                    other_grids.append(p.grid)
                        
                        # Extraire les features de manière sécurisée
                        try:
                            features = ml_ai.extract_game_features(grid, game.discard, other_grids)
                            
                            # Vérifier que les features sont valides
                            if features is not None and len(features) > 0:
                                # Observer les décisions d'InitialAI pour la source
                                if game.discard and len(game.discard) > 0:
                                    try:
                                        source_choice = current_player.ai.choose_source(grid, game.discard, other_grids)
                                        if source_choice in ['D', 'P']:
                                            ml_ai.add_training_example('source', features, 1 if source_choice == 'D' else 0)
                                    except Exception as e:
                                        # En cas d'erreur, ignorer cet échantillon
                                        pass
                        except Exception as e:
                            # En cas d'erreur dans l'extraction des features, continuer
                            pass
                
                try:
                    game.step()
                    game_steps += 1
                except Exception as e:
                    # En cas d'erreur dans le step, arrêter cette partie
                    break
            
            if game.finished:
                successful_games += 1
            
            if (game_num + 1) % 10 == 0:
                print(f"📊 Parties traitées: {game_num + 1}/50, Réussies: {successful_games}")
                
        except Exception as e:
            print(f"⚠️ Erreur dans la partie {game_num + 1}: {e}")
            continue
    
    print(f"✅ Collecte terminée! {successful_games} parties réussies sur 50")
    
    # Entraîner les modèles avec les données collectées si on a assez d'échantillons
    min_samples = max(10, successful_games // 5)  # Au moins 10 échantillons ou 1/5 des parties réussies
    ml_ai.train_models(min_samples=min_samples)
    
    print("✅ Collecte et entraînement terminés!")
    return ml_ai 