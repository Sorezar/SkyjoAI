#!/usr/bin/env python3
import os
import sys
from datetime import datetime
import argparse
import traceback


def create_model_directories():
    """Crée les répertoires nécessaires pour les modèles"""
    dirs = ["ml_models", "deep_models", "rl_models"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

def train_xgboost_models():
    """Entraîne les modèles XGBoost de manière sécurisée"""
    print("\n🔒 ENTRAÎNEMENT SÉCURISÉ - XGBOOST")
    print("=" * 60)
    
    try:
        from ai.ml_xgboost import XGBoostSkyjoAI
        
        print("🚀 Initialisation XGBoostSkyjoAI...")
        ai = XGBoostSkyjoAI()
        
        print("📊 Collecte de données d'entraînement SÉCURISÉES...")
        ai.collect_training_data(num_games=1000)
        
        print("🧠 Entraînement des modèles XGBoost...")
        ai.train_xgboost_models()
        
        print("✅ XGBoostSkyjoAI entraîné avec succès (mode sécurisé)")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement XGBoost: {e}")
        traceback.print_exc()
        return False


def train_enhanced_models():
    """Entraîne les modèles Enhanced de manière sécurisée"""
    print("\n🔒 ENTRAÎNEMENT SÉCURISÉ - MODÈLES ENHANCED")
    print("=" * 60)
    
    models_results = {}
    
    # Liste des modèles enhanced à entraîner
    enhanced_models = [
        ("UnsupervisedPatternAI", "ai.unsupervised_pattern_ai", "UnsupervisedPatternAI"),
        ("HybridEliteAI", "ai.hybrid_elite_ai", "HybridEliteAI"),
        ("AdaptiveMLAI", "ai.adaptive_ml_ai", "AdaptiveMLAI"),
        ("ChampionEliteAI", "ai.champion_elite_ai", "ChampionEliteAI")
    ]
    
    for model_name, module_path, class_name in enhanced_models:
        try:
            print(f"\n🎯 Entraînement {model_name}...")
            
            # Import dynamique
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            
            # Instanciation et entraînement si la méthode existe
            ai = model_class()
            if hasattr(ai, 'train_models'):
                ai.train_models()
                print(f"✅ {model_name} entraîné avec succès")
                models_results[model_name] = True
            else:
                print(f"⚠️ {model_name} ne nécessite pas d'entraînement")
                models_results[model_name] = True
                
        except ImportError:
            print(f"⚠️ {model_name} non disponible (dépendances manquantes)")
            models_results[model_name] = False
        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement {model_name}: {e}")
            models_results[model_name] = False
    
    successful = len([r for r in models_results.values() if r])
    total = len(models_results)
    print(f"\n📊 Modèles Enhanced: {successful}/{total} réussis")
    
    return successful > 0


def main():
    parser = argparse.ArgumentParser(description="Entraînement sécurisé unifié des modèles d'IA")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Ignorer les tests de validation")
    
    args = parser.parse_args()
    
    print("🔒 ENTRAÎNEMENT SÉCURISÉ UNIFIÉ - SKYJO AI")
    print("=" * 70)
    print(f"🕐 Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🛡️ OBJECTIF: Entraîner tous les modèles")
    print("=" * 70)
    
    # Étape 1: Préparation
    create_model_directories()
    
    # Étape 2: Entraînement sécurisé
    models_to_train = args.models if "all" not in args.models else ["xgboost", "enhanced"]
    successful_models = []
    failed_models = []
    
    
    if "xgboost" in models_to_train:
        if train_xgboost_models():
            successful_models.append("XGBoostSkyjoAI")
        else:
            failed_models.append("XGBoostSkyjoAI")
    
    if "enhanced" in models_to_train:
        if train_enhanced_models():
            successful_models.append("Enhanced Models")
        else:
            failed_models.append("Enhanced Models")
    
    # Résumé final
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ DE L'ENTRAÎNEMENT SÉCURISÉ")
    print("=" * 70)
    print(f"🕐 Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Groupes de modèles réussis: {len(successful_models)}")
    for model in successful_models:
        print(f"   • {model}")
    
    if failed_models:
        print(f"❌ Groupes de modèles échoués: {len(failed_models)}")
        for model in failed_models:
            print(f"   • {model}")
    print("=" * 70)
    
    return len(successful_models), len(failed_models)

if __name__ == "__main__":
    try:
        successful, failed = main()
        if failed > 0:
            print(f"\n⚠️ {failed} groupe(s) de modèles ont échoué lors de l'entraînement")
            sys.exit(1)
        else:
            print("\n🎉 Entraînement sécurisé terminé avec succès!")
    except KeyboardInterrupt:
        print("\n🛑 Entraînement interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erreur fatale: {e}")
        traceback.print_exc()
        sys.exit(1) 