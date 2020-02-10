# Projet-Deep Learning Ganrator
Etude du [dataset MNIST Fashion de Zalando]((https://github.com/zalandoresearch/fashion-mnist)) par du deep Learning
Autre possibilité : Etude du dataset [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
 
## Installation

Deux dépendances sont `matplotlib` et `ipython`

```
pip install matplotlib ipython
```

Il faut aussi installer PyTorch. Toutes les informations sont tirées du [site officiel de PyTorch](https://pytorch.org/get-started/locally/)

### Avec pip

Si CUDA est disponible (regarder sur internet si la GPU a CUDA):

```
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Sinon

```
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### Verification

Lancer `python` et entrer ces lignes

```
from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
```

Pour vérifier que CUDA (gpu) est disponible:

```
import torch
torch.cuda.is_available()
```

## Abstract

Les membres de la communauté des data scientists commencent souvent à apprendre avec le dataset MNIST, composé de chiffres manuscrits, voire l'utilisent comme benchmark pour valider leurs modèles. 

Ce dataset est aujourd'hui sur-utilisé, trop facile et peu représentatif des tâches de CV modernes. L’objectif de cette étude est donc de comparer les approches de vision par ordinateur classiques avec des approches dites de deep learning dans le cadre de la classification d’objets issus du dataset MNIST Fashion.

On présentera en particulier les techniques d’augmentation de données pour l’utilisation de réseaux profonds, les méthodes de deep learning utilisées grâce à cela, les méthodes de vision par ordinateur à l’aide de descripteurs. Les résultats obtenus seront présentés et comparés.


Autre possibilité : Reconnaissance de facial feature du dataset CelebA

## Introduction
 
Enjeu économique ou pratique : blablabla

## Etat de l'art

Tout le monde l'a déjà fait on essaye de comprendre comment ca marche et d'avoir des bons résultats

## Approche

Choix du modele, des poids, du dataset, sorties attendues, metrique, score

## Experimentations

Ajouter de plus en plus d'optimisations, graphes et tableaux

## Conclusions
 
On a bien progressé et taffé, on est plus fort avec ce modèle parce que ca, on pourrait améliorer avec tel truc, des bonnes données c'est bien, et on a pas fait ca c'est notre limite
 
## References

Un max