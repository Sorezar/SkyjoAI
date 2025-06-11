# 📊 Benchmark Skyjo AI

Ce dossier contient tous les outils de test et d'évaluation des performances des modèles d'IA Skyjo.

## 🎯 Contenu

### Scripts de test
- **`test_all_ai_models.py`** : Script principal de benchmark pour évaluer toutes les IA contre InitialAI

### Résultats historiques
- **`benchmark_results_*.json`** : Résultats sauvegardés des tests précédents avec métadonnées

## 🚀 Utilisation

### Test complet de toutes les IA
```bash
python benchmark/test_all_ai_models.py
```

### Test rapide (25 parties par IA)
```bash
python benchmark/test_all_ai_models.py --quick
```

### Test d'une IA spécifique
```bash
python benchmark/test_all_ai_models.py --ai advanced
python benchmark/test_all_ai_models.py --ai xgboost
```

### Options disponibles
- `--games N` : Nombre de parties par IA (défaut: 100)
- `--quick` : Test rapide avec 25 parties
- `--ai MODEL` : Tester une seule IA spécifique
- `--quiet` : Mode silencieux
- `--validate-only` : Validation des modèles sans les tester

## 📈 Interprétation des résultats

### Baseline de référence
- **InitialAI** : ≈20.9 points (référence)
- Objectif : Obtenir un score moyen < 20.9 pour battre InitialAI

### Indicateurs de performance
- **Score moyen** : Plus bas = meilleur
- **Taux de victoire** : Pourcentage de parties gagnées
- **Consistance** : Stabilité des performances (moins de variabilité)
- **Performance vs InitialAI** : Différence par rapport à la baseline

### Qualité des performances
- 🌟 **EXCELLENT** : < 18 points
- 🔥 **TRÈS BON** : 18-20 points  
- ✅ **BON** : 20-20.9 points
- 🟡 **CORRECT** : 20.9-22 points
- 🟠 **FAIBLE** : 22-25 points
- 🔴 **TRÈS FAIBLE** : > 25 points

## 🛡️ Sécurité

Tous les tests sont effectués avec le système anti-triche :
- Les IA ne voient que les cartes révélées
- Aucun accès aux cartes cachées des adversaires
- Performances basées sur des décisions légitimes

## 📁 Structure des résultats

Les fichiers JSON contiennent :
```json
{
  "metadata": {
    "timestamp": "2024-XX-XX",
    "num_games": 100,
    "baseline_reference": "InitialAI (≈20.9 points)"
  },
  "results":  {
    "AIName": {
      "games_played": 100,
      "wins": 25,
      "win_rate": 25.0,
      "average_score": 19.5,
      "performance_vs_initial": 1.4
    }
  }
}
``` 