"""
Script de test pour évaluer tous les modèles d'IA Skyjo contre InitialAI
Teste toutes les approches contre 3 InitialAI sur plusieurs parties
"""

import json
import numpy as np
from datetime import datetime
from collections import defaultdict
import argparse
import traceback

# Imports de base (toujours disponibles)
from ai.initial import InitialAI
from ai.advanced import AdvancedAI
from ai.ml_ai import MachineLearningAI
from ai.deep_ai import UnsupervisedDeepAI
from ai.advanced_dominant import AdvancedDominantAI

# Nouveaux modèles enhanced
from ai.unsupervised_pattern_ai import UnsupervisedPatternAI
from ai.hybrid_elite_ai import HybridEliteAI
from ai.adaptive_ml_ai import AdaptiveMLAI
from ai.champion_elite_ai import ChampionEliteAI

# Imports conditionnels pour les modèles avec dépendances spéciales
optional_models = {}

try:
    from ai.ml_xgboost import XGBoostSkyjoAI
    optional_models["XGBoostSkyjoAI"] = XGBoostSkyjoAI
except ImportError as e:
    print(f"⚠️ XGBoostSkyjoAI non disponible: {e}")

try:
    from ai.ml_xgboost_enhanced import XGBoostEnhancedAI
    optional_models["XGBoostEnhancedAI"] = XGBoostEnhancedAI
except ImportError as e:
    print(f"⚠️ XGBoostEnhancedAI non disponible: {e}")


from core.game import SkyjoGame, Scoreboard
from core.player import Player

class AIBenchmark:
    """Classe pour benchmarker les différentes IA"""
    
    def __init__(self, num_games=100, verbose=True):
        self.num_games = num_games
        self.verbose = verbose
        self.results = {}
        
        # Initialiser les IA de base
        print("🔧 Initialisation des modèles d'IA...")
        self.ai_models = {
            "InitialAI": InitialAI(),
            "AdvancedAI": AdvancedAI(),
            "MachineLearningAI": MachineLearningAI(),
            "UnsupervisedDeepAI": UnsupervisedDeepAI(),
            "AdvancedDominantAI": AdvancedDominantAI(),
            "UnsupervisedPatternAI": UnsupervisedPatternAI(),
            "HybridEliteAI": HybridEliteAI(),
            "AdaptiveMLAI": AdaptiveMLAI(),
            "ChampionEliteAI": ChampionEliteAI()
        }
        
        # Ajouter les modèles optionnels disponibles
        for model_name, model_class in optional_models.items():
            try:
                self.ai_models[model_name] = model_class()
            except Exception as e:
                print(f"⚠️ Erreur initialisation {model_name}: {e}")
        
        # Vérifier quels modèles sont prêts
        self.validate_models()
        print(f"✅ {len(self.ai_models)} modèles initialisés!")
    
    def validate_models(self):
        """Valide que tous les modèles peuvent être instanciés correctement"""
        models_to_remove = []
        
        for ai_name, ai_instance in self.ai_models.items():
            try:
                # Test basique d'instanciation
                if ai_instance is None:
                    print(f"⚠️ {ai_name}: Instance None")
                    models_to_remove.append(ai_name)
                    continue
                
                # Vérifier les méthodes essentielles
                required_methods = ['initial_flip', 'choose_source', 'choose_keep', 'choose_position', 'choose_reveal']
                for method in required_methods:
                    if not hasattr(ai_instance, method):
                        print(f"⚠️ {ai_name}: Méthode {method} manquante")
                        models_to_remove.append(ai_name)
                        break
                
            except Exception as e:
                print(f"❌ {ai_name}: Erreur d'initialisation - {e}")
                models_to_remove.append(ai_name)
        
        # Supprimer les modèles défaillants
        for ai_name in models_to_remove:
            del self.ai_models[ai_name]
        
        if models_to_remove:
            print(f"🗑️ Modèles retirés du test: {', '.join(models_to_remove)}")
    
    def run_single_test(self, ai_name, ai_instance):
        """Teste une IA spécifique contre 3 InitialAI"""
        print(f"\n🎯 Test de {ai_name} contre 3 InitialAI...")
        print("-" * 50)
        
        scores = []
        wins = 0
        game_details = []
        errors = []
        
        for game_num in range(self.num_games):
            try:
                # Créer les joueurs avec gestion d'erreur améliorée
                try:
                    players = [
                        Player(0, ai_name, ai_instance),
                        Player(1, "InitialAI_1", InitialAI()),
                        Player(2, "InitialAI_2", InitialAI()),
                        Player(3, "InitialAI_3", InitialAI())
                    ]
                except Exception as e:
                    errors.append(f"Partie {game_num + 1}: Erreur création joueurs - {e}")
                    continue
                
                # Créer et jouer la partie
                try:
                    scoreboard = Scoreboard(players)
                    game = SkyjoGame(players, scoreboard)
                except Exception as e:
                    errors.append(f"Partie {game_num + 1}: Erreur création jeu - {e}")
                    continue
                
                # Jouer jusqu'à la fin avec timeout de sécurité
                max_steps = 1000  # Limite pour éviter les boucles infinies
                steps = 0
                
                try:
                    while not game.finished and steps < max_steps:
                        game.step()
                        steps += 1
                    
                    if steps >= max_steps:
                        errors.append(f"Partie {game_num + 1}: Timeout - trop d'étapes")
                        continue
                        
                except Exception as e:
                    errors.append(f"Partie {game_num + 1}: Erreur pendant le jeu - {e}")
                    continue
                
                # Récupérer les scores finaux
                try:
                    final_scores = [player.round_score() for player in game.players]
                    ai_score = final_scores[0]  # L'IA testée est en première position
                    opponent_scores = final_scores[1:]
                    
                    # Validation des scores
                    if not isinstance(ai_score, (int, float)) or ai_score < -50 or ai_score > 200:
                        errors.append(f"Partie {game_num + 1}: Score invalide {ai_score}")
                        continue
                    
                    scores.append(ai_score)
                    
                    # Vérifier si l'IA a gagné
                    if ai_score == min(final_scores):
                        wins += 1
                    
                    # Enregistrer les détails de la partie
                    game_details.append({
                        'game_num': game_num + 1,
                        'ai_score': ai_score,
                        'opponent_scores': opponent_scores,
                        'won': ai_score == min(final_scores),
                        'margin': ai_score - min(opponent_scores),
                        'steps': steps
                    })
                    
                except Exception as e:
                    errors.append(f"Partie {game_num + 1}: Erreur calcul scores - {e}")
                    continue
                
                # Affichage périodique
                if self.verbose and (game_num + 1) % 25 == 0:
                    if scores:
                        avg_score = np.mean(scores)
                        win_rate = wins / len(scores) * 100
                        print(f"   Partie {game_num + 1}/{self.num_games}: "
                              f"Score moyen: {avg_score:.2f}, "
                              f"Taux de victoire: {win_rate:.1f}%, "
                              f"Erreurs: {len(errors)}")
                
            except Exception as e:
                errors.append(f"Partie {game_num + 1}: Erreur générale - {e}")
                if self.verbose:
                    print(f"⚠️ Erreur dans la partie {game_num + 1}: {e}")
                continue
        
        # Afficher les erreurs si il y en a
        if errors:
            print(f"\n⚠️ {len(errors)} erreurs détectées pour {ai_name}:")
            for error in errors[:5]:  # Afficher seulement les 5 premières
                print(f"   - {error}")
            if len(errors) > 5:
                print(f"   ... et {len(errors) - 5} autres erreurs")
        
        # Calculer les statistiques finales
        if scores:
            stats = self.calculate_statistics(scores, wins, game_details)
            stats['errors'] = len(errors)
            stats['success_rate'] = len(scores) / self.num_games * 100
            self.results[ai_name] = stats
            self.print_ai_results(ai_name, stats)
        else:
            print(f"❌ Aucune partie valide pour {ai_name} ({len(errors)} erreurs)")
            self.results[ai_name] = None
    
    def calculate_statistics(self, scores, wins, game_details):
        """Calcule les statistiques détaillées"""
        stats = {
            'games_played': len(scores),
            'wins': wins,
            'win_rate': wins / len(scores) * 100,
            'average_score': np.mean(scores),
            'median_score': np.median(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'scores_under_20': sum(1 for s in scores if s < 20),
            'scores_under_25': sum(1 for s in scores if s < 25),
            'game_details': game_details
        }
        
        # Calculs additionnels
        stats['consistency'] = 100 - (stats['std_score'] / stats['average_score'] * 100)
        stats['performance_vs_initial'] = 20.9 - stats['average_score']  # InitialAI baseline: 20.9
        
        return stats
    
    def print_ai_results(self, ai_name, stats):
        """Affiche les résultats détaillés d'une IA"""
        if stats is None:
            print(f"❌ {ai_name}: Aucun résultat disponible")
            return
        
        print(f"\n📊 RÉSULTATS POUR {ai_name}")
        print("=" * 60)
        print(f"🎮 Parties jouées: {stats['games_played']} / {self.num_games}")
        print(f"✅ Taux de réussite: {stats['success_rate']:.1f}%")
        if stats.get('errors', 0) > 0:
            print(f"⚠️ Erreurs: {stats['errors']}")
        print(f"🏆 Victoires: {stats['wins']} ({stats['win_rate']:.1f}%)")
        print(f"📈 Score moyen: {stats['average_score']:.2f}")
        print(f"📊 Score médian: {stats['median_score']:.2f}")
        print(f"📏 Écart-type: {stats['std_score']:.2f}")
        print(f"⬇️ Score minimum: {stats['min_score']}")
        print(f"⬆️ Score maximum: {stats['max_score']}")
        print(f"🎯 Scores < 20: {stats['scores_under_20']} ({stats['scores_under_20']/stats['games_played']*100:.1f}%)")
        print(f"🎯 Scores < 25: {stats['scores_under_25']} ({stats['scores_under_25']/stats['games_played']*100:.1f}%)")
        print(f"⚖️ Consistance: {stats['consistency']:.1f}%")
        print(f"🔥 Performance vs InitialAI: {stats['performance_vs_initial']:.2f} points")
        
        # Évaluation qualitative avec plus de nuances
        if stats['average_score'] < 18:
            print("🌟 EXCELLENT - Largement supérieur à InitialAI!")
        elif stats['average_score'] < 20:
            print("🔥 TRÈS BON - Nettement supérieur à InitialAI!")
        elif stats['average_score'] < 20.9:
            print("✅ BON - Meilleur que InitialAI!")
        elif stats['average_score'] < 22:
            print("🟡 CORRECT - Proche d'InitialAI")
        elif stats['average_score'] < 25:
            print("🟠 FAIBLE - En retard sur InitialAI")
        else:
            print("🔴 TRÈS FAIBLE - Très en retard sur InitialAI")
        
        # Ajouter une évaluation de la stabilité
        if stats['consistency'] > 85:
            print("💎 Très stable dans ses performances")
        elif stats['consistency'] > 70:
            print("⚖️ Assez stable")
        else:
            print("📉 Performance irrégulière")
    
    def run_all_tests(self):
        """Lance tous les tests"""
        print("🚀 BENCHMARK COMPLET DES IA SKYJO")
        print("=" * 60)
        print(f"📊 {self.num_games} parties par IA")
        print(f"🎯 Objectif: Battre InitialAI (≈20.9 points)")
        print(f"🕐 Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Tester chaque IA (sauf InitialAI qui sert de référence)
        ai_to_test = [name for name in self.ai_models.keys() if name != "InitialAI"]
        
        for ai_name in ai_to_test:
            try:
                self.run_single_test(ai_name, self.ai_models[ai_name])
            except Exception as e:
                print(f"❌ Erreur lors du test de {ai_name}: {e}")
                self.results[ai_name] = None
        
        # Afficher le classement final
        self.print_final_ranking()
        
        # Sauvegarder les résultats
        self.save_results()
    
    def print_final_ranking(self):
        """Affiche le classement final"""
        print("\n" + "=" * 80)
        print("🏆 CLASSEMENT FINAL")
        print("=" * 80)
        
        # Filtrer les résultats valides et trier par score moyen
        valid_results = [(name, stats) for name, stats in self.results.items() if stats is not None]
        valid_results.sort(key=lambda x: x[1]['average_score'])
        
        print(f"{'Rang':<5} {'IA':<20} {'Score Moy':<12} {'Victoires':<10} {'vs InitialAI':<12}")
        print("-" * 80)
        
        for i, (ai_name, stats) in enumerate(valid_results, 1):
            performance_icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            vs_initial = f"{stats['performance_vs_initial']:+.2f}"
            
            print(f"{performance_icon}{i:<4} {ai_name:<20} {stats['average_score']:<12.2f} "
                  f"{stats['win_rate']:<10.1f}% {vs_initial:<12}")
        
        # Ligne de référence InitialAI
        print("-" * 80)
        print(f"{'REF':<5} {'InitialAI (baseline)':<20} {'20.90':<12} {'≈20.0%':<10} {'±0.00':<12}")
        
        # Analyser les performances
        print("\n📈 ANALYSE DES PERFORMANCES")
        print("-" * 40)
        
        best_performers = [name for name, stats in valid_results if stats['average_score'] < 20.9]
        if best_performers:
            print(f"✅ IA battant InitialAI: {', '.join(best_performers)}")
            best_ai = valid_results[0]
            improvement = 20.9 - best_ai[1]['average_score']
            print(f"🏆 Meilleure amélioration: {best_ai[0]} (-{improvement:.2f} points)")
        else:
            print("❌ Aucune IA ne bat InitialAI de manière consistante")
        
        # Recommandations
        print("\n💡 RECOMMANDATIONS")
        print("-" * 40)
        
        if best_performers:
            print(f"🎯 Utiliser {best_performers[0]} pour les meilleures performances")
        
        # Identifier les points d'amélioration
        worst_ai = valid_results[-1] if valid_results else None
        if worst_ai and worst_ai[1]['average_score'] > 25:
            print(f"🔧 {worst_ai[0]} nécessite un réentraînement (score: {worst_ai[1]['average_score']:.2f})")
    
    def save_results(self):
        """Sauvegarde les résultats dans un fichier JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        
        # Préparer les données pour la sérialisation
        serializable_results = {}
        for ai_name, stats in self.results.items():
            if stats is not None:
                # Convertir les numpy arrays en listes
                serializable_stats = stats.copy()
                for key, value in serializable_stats.items():
                    if isinstance(value, np.ndarray):
                        serializable_stats[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        serializable_stats[key] = float(value)
                
                serializable_results[ai_name] = serializable_stats
        
        # Ajouter les métadonnées
        full_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_games': self.num_games,
                'baseline_reference': 'InitialAI (≈20.9 points)'
            },
            'results': serializable_results
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(full_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Résultats sauvegardés dans: {filename}")
        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test de performance des IA Skyjo")
    parser.add_argument("--games", type=int, default=100,
                       help="Nombre de parties par IA (défaut: 100)")
    parser.add_argument("--quick", action="store_true",
                       help="Test rapide avec 25 parties")
    parser.add_argument("--ai", type=str, 
                       choices=["advanced", "ml", "deep", "dominant", "pattern", "hybrid", "adaptive", "champion",
                               "xgboost", "xgboost_enhanced", "transformer"],
                       help="Tester une seule IA spécifique")
    parser.add_argument("--quiet", action="store_true",
                       help="Mode silencieux")
    parser.add_argument("--validate-only", action="store_true",
                       help="Valider seulement les modèles sans les tester")
    
    args = parser.parse_args()
    
    # Ajuster le nombre de parties
    num_games = 25 if args.quick else args.games
    verbose = not args.quiet
    
    # Créer le benchmark
    benchmark = AIBenchmark(num_games=num_games, verbose=verbose)
    
    if args.validate_only:
        print("✅ Validation des modèles terminée!")
        return
    
    if args.ai:
        # Tester une seule IA
        ai_mapping = {
            "advanced": "AdvancedAI",
            "ml": "MachineLearningAI", 
            "deep": "UnsupervisedDeepAI",
            "dominant": "AdvancedDominantAI",
            "pattern": "UnsupervisedPatternAI",
            "hybrid": "HybridEliteAI",
            "adaptive": "AdaptiveMLAI",
            "champion": "ChampionEliteAI",
            "xgboost": "XGBoostSkyjoAI",
            "xgboost_enhanced": "XGBoostEnhancedAI",
            "transformer": "TransformerDeepAI"
        }
        ai_name = ai_mapping[args.ai]
        if ai_name in benchmark.ai_models:
            print(f"🎯 Test spécifique de {ai_name}")
            benchmark.run_single_test(ai_name, benchmark.ai_models[ai_name])
        else:
            print(f"❌ {ai_name} n'est pas disponible (échec de validation)")
    else:
        # Tester toutes les IA
        benchmark.run_all_tests()
    
    print(f"\n🕐 Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎉 Benchmark terminé!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Test interrompu par l'utilisateur.")
    except Exception as e:
        print(f"\n💥 Erreur fatale: {e}")
        traceback.print_exc() 