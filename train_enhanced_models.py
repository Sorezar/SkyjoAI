"""
Script d'entraînement pour tous les modèles améliorés de Skyjo AI
- AdvancedDominantAI
- XGBoostSkyjoAI
"""

import os
import sys
from datetime import datetime
import argparse

def train_advanced_dominant():
    """Entraîne AdvancedDominantAI (pas d'entraînement nécessaire)"""
    print("🎯 ADVANCED DOMINANT AI")
    print("=" * 50)
    print("✅ AdvancedDominantAI est prêt à l'emploi!")
    print("   Aucun entraînement nécessaire - modèle heuristique optimisé")
    print()
    
    from ai.advanced_dominant import AdvancedDominantAI
    ai = AdvancedDominantAI()
    return ai


def train_xgboost(data_games=2000):
    """Entraîne le modèle XGBoost"""
    print("🚀 XGBOOST ML AI")
    print("=" * 50)
    
    try:
        from ai.ml_xgboost import XGBoostSkyjoAI
        
        # Créer le modèle
        ai = XGBoostSkyjoAI()
        
        # Essayer de charger des modèles existants
        if ai.load_models():
            print("📂 Modèles XGBoost existants chargés")
            print("   Voulez-vous réentraîner ? (recommandé avec nouvelles données)")
        
        # Collecte de données d'entraînement
        print(f"📊 Collecte de données d'entraînement sur {data_games} parties...")
        ai.collect_training_data(num_games=data_games)
        
        # Entraînement des modèles
        ai.train_xgboost_models()
        
        print("✅ XGBoost AI entraîné avec succès!")
        print()
        return ai
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement XGBoost: {e}")
        return None

def train_neural_supervised(episodes=2000):
    """Entraîne le Neural Network supervisé (Approche B pour ML)"""
    print("🧠 NEURAL NETWORK SUPERVISED AI")
    print("=" * 50)
    print("⚠️ Ce modèle sera implémenté dans la phase suivante")
    print("   Concentration sur les 5 modèles actuels d'abord")
    print()
    return None

def main():
    parser = argparse.ArgumentParser(description="Entraînement des modèles Skyjo AI améliorés")
    parser.add_argument("--models", type=str, default="all", 
                       help="Modèles à entraîner: all, dominant, actor_critic, vae, transformer, xgboost")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Nombre d'épisodes pour les modèles deep learning")
    parser.add_argument("--data_games", type=int, default=2000,
                       help="Nombre de parties pour collecter les données XGBoost")
    parser.add_argument("--quick", action="store_true",
                       help="Entraînement rapide (moins d'épisodes)")
    
    args = parser.parse_args()
    
    # Ajuster pour entraînement rapide
    if args.quick:
        args.episodes = 500
        args.data_games = 1000
    
    print("🚀 ENTRAÎNEMENT DES MODÈLES SKYJO AI AMÉLIORÉS")
    print("=" * 60)
    print(f"🕐 Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Modèles: {args.models}")
    print(f"📊 Episodes DL: {args.episodes}")
    print(f"📈 Parties ML: {args.data_games}")
    print("=" * 60)
    print()
    
    # Créer les répertoires nécessaires
    os.makedirs("deep_models", exist_ok=True)
    os.makedirs("ml_models", exist_ok=True)
    
    trained_models = {}
    
    # Entraîner les modèles sélectionnés
    if args.models == "all" or "dominant" in args.models:
        trained_models["AdvancedDominantAI"] = train_advanced_dominant()
    
    if args.models == "all" or "xgboost" in args.models:
        trained_models["XGBoostSkyjoAI"] = train_xgboost(args.data_games)
    
    # Résumé final
    print("🏁 RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("=" * 60)
    print(f"🕐 Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    successful_models = [name for name, model in trained_models.items() if model is not None]
    failed_models = [name for name, model in trained_models.items() if model is None]
    
    print(f"✅ Modèles entraînés avec succès: {len(successful_models)}")
    for model_name in successful_models:
        print(f"   • {model_name}")
    
    if failed_models:
        print(f"❌ Modèles échoués: {len(failed_models)}")
        for model_name in failed_models:
            print(f"   • {model_name}")
    
    print()
    print("🎯 PROCHAINES ÉTAPES:")
    print("1. Tester les modèles avec test_all_ai_models.py")
    print("2. Comparer les performances vs InitialAI baseline (~20.9 points)")
    print("3. Identifier les meilleurs modèles pour hybridation")
    print()
    print("💡 COMMANDES UTILES:")
    print("   # Test rapide de tous les modèles")
    print("   python test_all_ai_models.py --quick")
    print()
    print("   # Benchmark complet")
    print("   python test_all_ai_models.py --games 200")
    print()
    print("   # Test spécifique")
    print("   python test_all_ai_models.py --ai advanced_dominant --games 100")
    
    print("=" * 60)
    
    return len(successful_models), len(failed_models)

if __name__ == "__main__":
    try:
        successful, failed = main()
        if failed > 0:
            sys.exit(1)  # Code d'erreur si des modèles ont échoué
    except KeyboardInterrupt:
        print("\n🛑 Entraînement interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erreur fatale: {e}")
        sys.exit(1) 