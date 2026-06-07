# A-IAT Research

Projet local pour explorer et visualiser les datasets CIFAR presents dans `data/`.

## Structure actuelle

- `tools/visualize_data.py`: outil de visualisation CIFAR-10 / CIFAR-100
- `data/CIFAR-10/`: fichiers batch CIFAR-10
- `data/CIFAR-100/`: fichiers batch CIFAR-100
- `requirements.txt`: dependances Python necessaires au projet

## Installation

Creer un environnement virtuel :

```powershell
python -m venv .venv
```

Activer l'environnement :

```powershell
.\.venv\Scripts\Activate.ps1
```

Installer les dependances :

```powershell
python -m pip install -r requirements.txt
```

Les dependances actuelles sont volontairement limitees aux bibliotheques utilisees par le code :

- `numpy`: manipulation des batchs CIFAR et selection aleatoire
- `matplotlib`: affichage des grilles d'images

## Visualiser les donnees CIFAR

L'outil `tools/visualize_data.py` lit les fichiers locaux dans `data/CIFAR-10` ou `data/CIFAR-100`, reconstruit les images 32x32, puis les affiche avec Matplotlib.

Aucune image n'est enregistree sur disque. L'affichage se fait uniquement dans une fenetre Matplotlib.

### Mode interactif

Lancer le script sans argument ouvre un menu dans le terminal :

```powershell
python .\tools\visualize_data.py
```

Le menu demande successivement :

- le dataset disponible a utiliser (`CIFAR-10` ou `CIFAR-100`)
- le split a afficher (`train`, `test` ou `all`)
- toutes les classes ou une classe precise
- le nombre d'images par classe
- le nombre maximum de classes a afficher, utile pour `CIFAR-100`
- le nombre de colonnes de la grille
- la seed aleatoire

### Mode avec arguments

Afficher CIFAR-10 :

```powershell
python .\tools\visualize_data.py --dataset cifar10 --samples-per-class 6
```

Afficher une classe precise :

```powershell
python .\tools\visualize_data.py --dataset cifar10 --class-name cat --samples-per-class 12
```

Afficher une partie de CIFAR-100 :

```powershell
python .\tools\visualize_data.py --dataset cifar100 --samples-per-class 2 --max-classes 12 --columns 6
```

Options principales :

- `--dataset`: `cifar10` ou `cifar100`
- `--split`: `train`, `test` ou `all`
- `--class-name`: nom ou numero de classe a afficher
- `--samples-per-class`: nombre d'images par classe
- `--max-classes`: limite le nombre de classes affichees
- `--columns`: nombre de colonnes dans la grille
- `--seed`: seed utilisee pour choisir les images aleatoirement

## Verification rapide

Depuis le venv actif :

```powershell
python -c "import numpy, matplotlib; print(numpy.__version__); print(matplotlib.__version__)"
```

Puis lancer le visualiseur :

```powershell
python .\tools\visualize_data.py --dataset cifar10 --class-name cat --samples-per-class 4 --columns 4
```
