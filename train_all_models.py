"""
Script d'entraînement pour tous les modèles d'IA Skyjo
- Machine Learning (Random Forest)
- Deep Learning (Auto-encodeur non supervisé)
- Reinforcement Learning (DQN)
"""

import os
import sys
import argparse
from datetime import datetime

def train_machine_learning_model():
    """Entraîne le modèle Machine Learning"""
    print("🔧 Début de l'entraînement du modèle Machine Learning...")
    
    try:
        from ai.ml_ai import collect_training_data_from_initial_ai
        ml_ai = collect_training_data_from_initial_ai()
        print("✅ Modèle Machine Learning entraîné avec succès!")
        return ml_ai
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement ML: {e}")
        import traceback
        traceback.print_exc()  # Afficher la stack trace complète
        return None

def train_deep_learning_model():
    """Entraîne le modèle Deep Learning non supervisé"""
    print("🧠 Début de l'entraînement du modèle Deep Learning...")
    
    try:
        from ai.deep_ai import train_unsupervised_model
        deep_ai = train_unsupervised_model()
        print("✅ Modèle Deep Learning entraîné avec succès!")
        return deep_ai
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement Deep Learning: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Entraînement des modèles d'IA Skyjo")
    parser.add_argument("--models", nargs="+", 
                       choices=["ml", "deep", "rl", "all"],
                       default=["all"],
                       help="Modèles à entraîner: ml, deep, rl, ou all")
    parser.add_argument("--quick", action="store_true",
                       help="Entraînement rapide avec moins d'épisodes")
    
    args = parser.parse_args()
    
    print("🚀 Script d'entraînement des modèles d'IA Skyjo")
    print("=" * 60)
    print(f"🕐 Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Modèles sélectionnés: {args.models}")
    print("=" * 60)
    
    # Créer les répertoires de modèles
    os.makedirs("ml_models", exist_ok=True)
    os.makedirs("deep_models", exist_ok=True)
    os.makedirs("rl_models", exist_ok=True)
    
    results = {}
    
    # Déterminer quels modèles entraîner
    models_to_train = []
    if "all" in args.models:
        models_to_train = ["ml", "deep"]  # Exclu RL par défaut car performances insuffisantes
    else:
        models_to_train = args.models
    
    # Entraîner les modèles sélectionnés
    if "ml" in models_to_train:
        print("\n" + "="*50)
        results["ml"] = train_machine_learning_model()
    
    if "deep" in models_to_train:
        print("\n" + "="*50)
        results["deep"] = train_deep_learning_model()
    
    # Résumé final
    print("\n" + "="*60)
    print("📋 RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("="*60)
    
    for model_name, model_instance in results.items():
        status = "✅ Succès" if model_instance is not None else "❌ Échec"
        print(f"{model_name.upper()}: {status}")
    
    successful_models = [name for name, instance in results.items() if instance is not None]
    failed_models = [name for name, instance in results.items() if instance is None]
    
    print(f"\n✅ Modèles entraînés avec succès: {len(successful_models)}")
    print(f"❌ Modèles ayant échoué: {len(failed_models)}")
    
    if failed_models:
        print(f"⚠️ Modèles à réentraîner: {', '.join(failed_models)}")
    
    print(f"\n🕐 Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return successful_models

if __name__ == "__main__":
    try:
        successful_models = main()
        if len(successful_models) > 0:
            print("\n🎉 Entraînement terminé! Vous pouvez maintenant tester les modèles.")
            print("📝 Utilisez 'python test_all_ai_models.py' pour évaluer les performances.")
        else:
            print("\n💥 Aucun modèle n'a été entraîné avec succès.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Entraînement interrompu par l'utilisateur.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erreur fatale: {e}")
        sys.exit(1) 