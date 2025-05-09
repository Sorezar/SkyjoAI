import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Charger les données depuis le fichier JSON
with open("simulation_results.json", "r") as file:
    data = json.load(file)

# Appliquer un style seaborn
sns.set_theme(style="whitegrid")

# Créer une figure avec une grille personnalisée
fig = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 5, figure=fig)  # Diviser en 5 colonnes (1 pour 20%, 4 pour 80%)

# Sous-graphe 1 : Barplot des valeurs de wins_by_ai
ax1 = fig.add_subplot(gs[0, 0])  # Première colonne (20%)
wins_by_ai = data.get("wins_by_ai", {})
keys = list(wins_by_ai.keys())
values = list(wins_by_ai.values())

sns.barplot(x=keys, y=values, palette="Blues_d", ax=ax1)
ax1.set_title("Wins by AI", fontsize=10)
ax1.set_xlabel("AI", fontsize=8)
ax1.set_ylabel("Wins", fontsize=8)
ax1.tick_params(axis='x', rotation=45, labelsize=8)
ax1.tick_params(axis='y', labelsize=8)

# Sous-graphe 2 : Courbe des points initiaux
ax2 = fig.add_subplot(gs[0, 1:])  # Colonnes 2 à 5 (80%)
initial_points = data.get("wins_by_initial_cards", {})
keys_sorted = sorted(initial_points.keys(), key=int)
values_sorted = [initial_points[k] for k in keys_sorted]

sns.lineplot(x=keys_sorted, y=values_sorted, marker='o', color='orange', ax=ax2)
ax2.set_title("Somme des cartes initiales du vainqueur", fontsize=10)
ax2.set_xlabel("Somme des cartes initiales", fontsize=8)
ax2.set_ylabel("Nombre de victoires", fontsize=8)
ax2.tick_params(axis='x', rotation=45, labelsize=8)
ax2.set_xticks(range(0, len(keys_sorted), 2))
ax2.set_xticklabels(keys_sorted[::2])

# Ajuster l'espacement
plt.tight_layout()
plt.show()
