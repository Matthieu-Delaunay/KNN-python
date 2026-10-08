# Classification par k plus proches voisins (k-NN) en Python

Implémentation de l'algorithme des **k plus proches voisins** (k-NN) en Python, en deux versions :

1. une version **« à la main »**, avec des boucles et des listes Python ;
2. une version **vectorisée**, avec NumPy et SciPy, sans boucle explicite.

Le projet permet de comparer les deux approches et de comprendre comment la vectorisation simplifie et accélère le code.

## Principe de l'algorithme

Pour classer un nouvel individu `x` :

1. on calcule la distance euclidienne entre `x` et chaque individu du jeu d'entraînement ;
2. on retient les `k` individus les plus proches ;
3. on affecte à `x` la classe la plus représentée parmi ces `k` voisins (en cas d'égalité, la première classe rencontrée est retenue).

## Structure du projet

```
.
├── knn.py          # Fonctions k-NN et code principal
└── README.md
```

> **Le module `P01_utils` n'est pas inclus dans ce dépôt.** Il a été fourni dans le cadre du cours et n'est donc pas redistribué ici (voir la section suivante).

## Le module `P01_utils` (non fourni)

Le code principal (section `__main__`) s'appuie sur un module externe `P01_utils`, **absent de ce dépôt**. Il fournit trois fonctions :

| Fonction | Rôle |
|---|---|
| `lire_donnees(n)` | Lit `n` individus et renvoie `X, y` sous forme de listes |
| `lire_donnees_numpy(n)` | Même chose, mais renvoie des tableaux NumPy |
| `visualiser_donnees(X_train, y_train, X_test, fichier)` | Trace les données et enregistre la figure (PDF) |

Conséquences :

- le script `knn.py` **ne peut pas être exécuté tel quel** sans ce module, car il l'importe au début du fichier ;
- pour tester le projet, il faut soit récupérer `P01_utils.py` auprès de la source d'origine et le placer à côté de `knn.py`, soit le remplacer par ses propres fonctions de chargement de données, soit supprimer l'import et la section `__main__` pour n'utiliser que les fonctions k-NN.

Les fonctions k-NN elles-mêmes (`dist`, `knn_indiv`, `classe_plus_repr`, `k_plus_proches_voisins_liste`, `k_plus_proches_voisins_numpy`) n'utilisent pas `P01_utils` et fonctionnent avec n'importe quel jeu de données au bon format.

## Prérequis

- Python 3.8 ou supérieur
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)

Installation des dépendances :

```bash
pip install numpy scipy matplotlib
```

## Utilisation

### Avec `P01_utils`

Une fois `P01_utils.py` placé dans le même dossier que `knn.py` :

```bash
python knn.py
```

Le script :

1. charge un jeu d'entraînement (100 individus) et un jeu de test (10 individus) ;
2. génère la visualisation des données dans `dataset.pdf` ;
3. affiche les prédictions de la version avec listes (`k = 5`) ;
4. affiche les prédictions de la version NumPy (`k = 5`).

### Avec ses propres données

Après avoir retiré l'import de `P01_utils` et la section `__main__` :

```python
import numpy as np
from knn import k_plus_proches_voisins_numpy

X_train = np.array([[0, 0], [1, 1], [5, 5], [6, 5]])
y_train = np.array([0, 0, 1, 1])
X_test = np.array([[0.5, 0.5], [5.5, 5.0]])

print(k_plus_proches_voisins_numpy(X_train, y_train, X_test, k=3))
# [0 1]
```

## Fonctions principales

| Fonction | Rôle |
|---|---|
| `dist(X_i, X_j)` | Distance euclidienne entre deux vecteurs |
| `knn_indiv(X_indiv, liste_indiv, nb_voisins)` | Indices des `nb_voisins` individus les plus proches de `X_indiv` |
| `classe_plus_repr(liste_classes)` | Classe(s) la (les) plus représentée(s) dans une liste |
| `k_plus_proches_voisins_liste(X_train, y_train, X_test, k)` | Prédictions k-NN avec boucles et listes Python |
| `k_plus_proches_voisins_numpy(X_train, y_train, X_test, k)` | Prédictions k-NN vectorisées avec NumPy |

## Version vectorisée

La version NumPy repose sur quatre étapes :

```python
# 1. matrice des distances (n_test, n_train)
distance = cdist(X_test, X_train)

# 2. indices des k plus proches voisins de chaque point de test
indices = np.argsort(distance, axis=1)[:, :k]

# 3. classes de ces voisins : forme (n_test, k)
voisins = y_train[indices]

# 4. vote : on compte les voisins de chaque classe, puis on prend la classe majoritaire
classes = np.unique(y_train)
votes = np.sum(voisins[:, :, None] == classes[None, None, :], axis=1)
predictions = classes[np.argmax(votes, axis=1)]
```

L'étape 4 utilise le *broadcasting* : chaque voisin est comparé à toutes les classes possibles, puis `np.sum` additionne les correspondances sur l'axe des voisins pour obtenir le nombre de votes par classe.

## Limites et remarques

- En cas d'égalité entre plusieurs classes, la version avec listes retient la première classe rencontrée, tandis que la version NumPy retient la première dans l'ordre de `np.unique` (la plus petite valeur). Les résultats peuvent donc différer dans ce cas précis.
- La version vectorisée construit un tableau intermédiaire de taille `n_test × k × n_classes`, ce qui peut être coûteux en mémoire pour de très grands jeux de données.
- La version NumPy renvoie un tableau NumPy, la version avec listes renvoie une liste Python.

## Auteur

**Matthieu Delaunay**
