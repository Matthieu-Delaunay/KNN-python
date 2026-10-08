# Matthieu Delaunay

# Section 1 : les imports
import numpy as np
import matplotlib.pyplot as plt
import P01_utils as ut
from scipy.spatial.distance import cdist

# Section 2 : les fonctions
def dist(X_i, X_j):
    #La fonction calcule la distance entre 2 vecteurs
    distance = 0
    for i in range(len(X_i)):
        distance += (X_i[i]-X_j[i])**2
    return distance**(1/2)

def knn_indiv(X_indiv, liste_indiv, nb_voisins):
    # La fonction renvoie la liste de n individus de la liste_indiv les plus proches de X_indiv
    # On initialise les listes qu'on utilise
    liste_dist = []
    liste_indice = []

    # On fait une liste des distances 
    for indiv in liste_indiv:
        liste_dist.append(dist(X_indiv, indiv))

    # On trie la liste en ordre croissant et on renvoie une liste de leurs indices originaux
    l_dist_trie = np.argsort(liste_dist)

    # On renvoie les n premiers éléments des indices triés
    for i in range(nb_voisins):
        liste_indice.append(l_dist_trie[i])
    return liste_indice

def classe_plus_repr(liste_classes):
    # renvoie les classes les plus représentés dans une liste
    dict_classe = {}

    # créé un dictionnaire avec pour chaque classe en indice, son nombre de représentants
    for classe in liste_classes:
        if classe not in dict_classe.keys():
            dict_classe[classe] = 1
        else : 
            dict_classe[classe] += 1

    # renvoie l'indice (et donc la classe) de la valeur maximum
    max_val = max(dict_classe.values())
    return [classe for classe in dict_classe.keys() if dict_classe[classe] == max_val]

def k_plus_proches_voisins_liste(X_train, y_train, X_test, k=1):
    # renvoie la liste des prédictions de x_test
    y_test = []

    # Pour chaque vecteur de X_test on calcule sa distance avec tous les vecteurs de X_train et on garde les indice des k plus proches
    for vect in X_test:
        liste_indice = knn_indiv(vect, X_train, k)
        liste_y_train = []

        # Avec ces indices, on vient récupérer une liste des classe des k plus proche voisins
        for indice in liste_indice:
            liste_y_train.append(y_train[indice])

        # On ajoute la valeur max de cette liste de classe dans la liste y_test, en cas d'égalité on prend le premier
        y_test.append(classe_plus_repr(liste_y_train)[0])
    return y_test

def k_plus_proches_voisins_numpy(X_train, y_train, X_test, k=1):
    # matrice des distances de chaque X_test avec chaque X_train
    distance = cdist(X_test, X_train)
    # indices des k plus proches voisins pour chaque X_test
    indices = np.argsort(distance, axis=1)[:, :k]
    # classes des k voisins : shape (n_test, k)
    voisins = y_train[indices]
    # liste des classes possibles : shape (n_classes,)
    classes = np.unique(y_train)
    # votes[i, c] = nombre de voisins de X_test[i] appartenant à la classe c
    votes = np.sum(voisins[:, :, None] == classes[None, None, :], axis=1)
    # classe la plus représentée pour chaque ligne
    return classes[np.argmax(votes, axis=1)]

#Section 3 : main code
if __name__ == "__main__":

    #3.2
    X_train, y_train = ut.lire_donnees(100)
    X_test, y_test = ut.lire_donnees(10)
    ut.visualiser_donnees(X_train, y_train, X_test, "dataset.pdf")

    #3.3
    print(k_plus_proches_voisins_liste(X_train, y_train, X_test, 5))

    # 3.4
    X_train, y_train = ut.lire_donnees_numpy(100)
    X_test, y_test = ut.lire_donnees_numpy(10)
    print(k_plus_proches_voisins_numpy(X_train, y_train, X_test, 5))
