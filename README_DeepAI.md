# DeepAI pour Skyjo

## Description

DeepAI est une intelligence artificielle basée sur le Deep Reinforcement Learning (DRL) pour jouer au Skyjo. Elle utilise des réseaux de neurones profonds (DQN) pour apprendre à prendre des décisions optimales.

## Architecture

DeepAI utilise 4 réseaux de neurones distincts pour chaque type de décision :
- **Source Network** : Choisit entre piocher ou prendre la défausse
- **Keep Network** : Décide de garder ou non la carte piochée
- **Position Network** : Choisit où placer la carte
- **Reveal Network** : Choisit quelle carte révéler

## Entraînement

Pour entraîner DeepAI :

```bash
python train_deepai.py
```

Par défaut, l'entraînement se fait sur 1000 épisodes. Vous pouvez modifier ce nombre dans le code.

### Paramètres d'entraînement
- **Learning rate** : 0.001
- **Gamma** : 0.99
- **Epsilon initial** : 1.0
- **Epsilon decay** : 0.995
- **Epsilon minimum** : 0.01

### Récompenses
- **Positives** :
  - Réduction du score : +2 points par point de score réduit
  - Suppression de colonne : +20 points par colonne
  - Révélation de carte : +1 point
  
- **Négatives** :
  - Augmentation du score : -1 point par point ajouté
  - Finir premier sans le meilleur score : -30 points

## Utilisation

### Avec un modèle pré-entraîné

```bash
python main_deepai.py
```

Le programme chargera automatiquement `deepai_model.pth` s'il existe.

### Dans votre propre code

```python
from ai.deepai import DeepAI

# Créer l'IA
deep_ai = DeepAI()

# Charger un modèle pré-entraîné
deep_ai.load_model('deepai_model.pth')

# Utiliser dans le jeu
player = Player(0, "DeepAI", deep_ai)
```

## Métriques de suivi

L'entraînement génère :
- Un fichier `deepai_model.pth` contenant les poids du modèle
- Un graphique `training_metrics.png` avec :
  - L'évolution des récompenses
  - L'évolution des scores
  - L'évolution du taux de victoire

## Test de performance

Après l'entraînement, DeepAI est automatiquement testée contre :
- 3 RandomAI
- 3 InitialAI
- Un mélange des deux

Les résultats affichent le taux de victoire et le score moyen.

## Améliorations possibles

1. **Double DQN** : Utiliser deux réseaux pour réduire la surestimation des Q-values
2. **Prioritized Experience Replay** : Donner plus d'importance aux expériences rares
3. **Dueling DQN** : Séparer l'estimation de la valeur d'état et de l'avantage
4. **Augmenter la complexité du réseau** : Plus de couches ou de neurones
5. **Ajuster les récompenses** : Expérimenter avec différents systèmes de récompense 