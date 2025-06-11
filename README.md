# 🎮 Skyjo AI - Projet d'Intelligence Artificielle

Système complet d'IA pour le jeu de cartes Skyjo avec interface web React et backend Python FastAPI.

## 🏗️ Architecture du projet

```
SkyjoAI/
├── 🎯 ai/                    # Modèles d'intelligence artificielle
├── 🎮 api/                   # Backend FastAPI  
├── 📊 benchmark/             # Tests et évaluation des IA
├── 🏗️ core/                 # Logique de jeu principale
├── 🎨 ui/                    # Interface React
├── 📁 ml_models/             # Modèles entraînés
├── 🔧 config/                # Configuration
└── 📋 train_secure_models.py # Entraînement unifié
```

## 🚀 Démarrage rapide

### 1. Installation des dépendances
```bash
npm run install:all
```

### 2. Lancement de l'application
```bash
npm start
```
- Backend API : http://localhost:8000
- Interface React : http://localhost:3000

### 3. Entraînement des modèles IA
```bash
python train_secure_models.py
```

### 4. Test des performances
```bash
python benchmark/test_all_ai_models.py --quick
```

## 🧠 Modèles d'IA disponibles

### IA de base
- **InitialAI** : IA de référence (baseline ≈20.9 points)
- **AdvancedAI** : Version améliorée avec stratégies avancées
- **AdvancedDominantAI** : Variant dominant plus agressif

### IA Machine Learning
- **MachineLearningAI** : Random Forest avec features avancées
- **XGBoostSkyjoAI** : Modèle XGBoost optimisé
- **XGBoostEnhancedAI** : Version enhanced avec plus de features

### IA Enhanced
- **UnsupervisedPatternAI** : Détection de patterns non supervisée
- **HybridEliteAI** : Combinaison de plusieurs approches
- **AdaptiveMLAI** : Apprentissage adaptatif en temps réel
- **ChampionEliteAI** : Modèle champion avec toutes les optimisations

## 🛡️ Système anti-triche

**SÉCURITÉ GARANTIE** : Tous les modèles sont entraînés et testés avec le système anti-triche intégré :
- ✅ Les IA ne voient que les cartes révélées
- ✅ Aucun accès aux cartes cachées des adversaires  
- ✅ Performances basées sur des décisions légitimes
- ✅ Validation automatique de sécurité

## 📊 Benchmark et performances

Le dossier `benchmark/` contient tous les outils d'évaluation :
- Test automatisé contre InitialAI (baseline)
- Métriques détaillées (score moyen, taux de victoire, consistance)
- Historique des résultats avec métadonnées
- Classification des performances (Excellent/Bon/Faible)

## 🔧 Scripts principaux

### Entraînement
```bash
python train_secure_models.py
```

### Benchmark
```bash
python benchmark/test_all_ai_models.py           # Test complet
python benchmark/test_all_ai_models.py --quick   # Test rapide
python benchmark/test_all_ai_models.py --ai advanced  # IA spécifique
```

### Interface
```bash
npm start                   # Interface complète (API + React)
npm run start:api           # Backend seulement
npm run start:react         # Frontend seulement
```

## 🎯 Objectifs de performance

- **Baseline** : InitialAI ≈ 20.9 points
- **Objectif** : Score moyen < 18 points
- **Excellence** : Taux de victoire > 30% avec consistance élevée

## 🔄 Historique des améliorations

### Phase 1 : Interface et base
- ✅ Interface React fonctionnelle
- ✅ Backend FastAPI avec API complète
- ✅ Correction des bugs d'affichage et scoring

### Phase 2 : Sécurité critique
- ✅ Découverte et correction du système de triche
- ✅ Implémentation du système anti-triche
- ✅ Réentraînement sécurisé de tous les modèles

### Phase 3 : Optimisation (en cours)
- 🔄 Ménage et réorganisation du code
- 🔄 Modèles enhanced et approches hybrides
- 🔄 Interface de logs détaillés

## 📋 TODO

- [ ] Logs détaillés en temps réel dans l'interface
- [ ] Correction du système de suppression de lignes/colonnes  
- [ ] Optimisation des modèles enhanced
- [ ] Interface d'administration des IA

## 🤝 Contribution

Le projet suit une architecture modulaire avec validation de sécurité intégrée. Toute nouvelle IA doit :
1. Respecter le système anti-triche
2. Être testée via le benchmark
3. Documenter ses performances

---

**🛡️ Projet 100% sécurisé contre la triche - Performances légitimes garanties** 