# A-IAT Research

Projet local pour explorer CIFAR-10 / CIFAR-100 / GTSRB et iterer proprement sur des experiences d'entrainement CNN.

## Structure actuelle

- `configs/simple_cnn.yaml`: configuration d'experience par defaut
- `models/simple_cnn.py`: architecture CNN simple pour CIFAR-10
- `models/registry.py`: registre des architectures et creation des checkpoints enrichis
- `models/loading.py`: chargement automatique d'un checkpoint enrichi en modele PyTorch
- `training/train.py`: point d'entree de l'entrainement
- `training/utils.py`: dataset PyTorch local CIFAR-10, seed, device
- `training/experiment.py`: creation de train runs, logs CSV, configs et summaries
- `tools/visualize_data.py`: outil utilisateur pour visualiser CIFAR-10 / CIFAR-100 / GTSRB
- `tools/compare_runs.py`: outil utilisateur pour comparer les experiences terminees
- `tools/promote_model.py`: outil utilisateur pour promouvoir un checkpoint de train run vers `trained_models/`
- `tools/generate_training_report.py`: outil utilisateur pour generer un rapport Markdown/PDF d'un train run
- `data/`: datasets locaux, non versionnes
- `train_runs/`: traces d'entrainement et checkpoints enrichis, non versionnes
- `attack_runs/`: traces d'attaques adversariales, non versionnees
- `trained_models/`: modeles selectionnes pour inference/attaques, non versionnes
- `reports/`: rapports generes, non versionnes

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Dependances principales : `numpy`, `matplotlib`, `Pillow`, `torch`, `torchvision`, `tqdm`, `PyYAML`.

## Visualiser les donnees

Mode interactif :

```powershell
python .\tools\visualize_data.py
```

Exemples avec arguments :

```powershell
python .\tools\visualize_data.py --dataset cifar10 --samples-per-class 6
python .\tools\visualize_data.py --dataset cifar10 --class-name cat --samples-per-class 12
python .\tools\visualize_data.py --dataset cifar100 --samples-per-class 2 --max-classes 12 --columns 6
python .\tools\visualize_data.py --dataset gtsrb --split train --samples-per-class 4 --max-classes 12 --columns 6
python .\tools\visualize_data.py --dataset gtsrb --split test --class-name 14 --samples-per-class 8 --columns 4
```

Aucune image n'est enregistree sur disque. L'affichage se fait dans une fenetre Matplotlib. Pour GTSRB, les classes sont numerotees de `class_00000` a `class_00042`; tu peux aussi utiliser directement un numero comme `14`. L'alias `gtrsb` est accepte.

## Entrainer une experience

La configuration par defaut est dans `configs/simple_cnn.yaml`.

```powershell
python -m training.train --config .\configs\simple_cnn.yaml
```

Override de parametres sans modifier la config :

```powershell
python -m training.train --experiment-name simplecnn_lr0005 --lr 0.0005 --epochs 20 --batch-size 128 --early-stopping-patience 8
```

Options disponibles :

- `--config`: chemin vers la config YAML
- `--experiment-name`: nom de l'experience
- `--epochs`: nombre maximum d'epochs
- `--batch-size`: taille de batch
- `--lr`: learning rate
- `--seed`: seed aleatoire
- `--early-stopping-patience`: nombre d'epochs sans amelioration de `test_acc` avant arret anticipe, `0` pour desactiver

Chaque entrainement cree un dossier dans `train_runs/`. Le nom suit le format `DD-MM-YYYY_HH-MM-SS_nom_experience` :

```txt
train_runs/
  07-06-2026_16-30-04_simplecnn_cifar10/
    config.yaml
    metrics.csv
    summary.json
    checkpoint.pth
```

Contenu utile :

- `config.yaml`: configuration exacte utilisee pour le run
- `metrics.csv`: loss, accuracy et temps par epoch
- `summary.json`: resume final du run
- `checkpoint.pth`: checkpoint enrichi du meilleur modele selon `test_acc`

## Checkpoint enrichi

Le checkpoint enrichi contient les poids et les metadonnees necessaires pour reconstruire le modele :

- `model_state_dict`: poids PyTorch du modele
- `model_name`: architecture a reconstruire, ex. `simple_cnn`
- `dataset`: dataset utilise, ex. `cifar10`
- `num_classes`: nombre de classes
- `class_names`: noms ou ids des classes
- `input_shape`: forme attendue en entree
- `normalization`: moyenne et ecart-type utilises
- `config`: config complete du run
- `metrics`: meilleure epoch, meilleure accuracy et loss minimale connue

Charger un modele pour inference ou attaque :

```python
from models.loading import load_trained_model

model, metadata = load_trained_model("train_runs/<train_run_id>/checkpoint.pth")
```

Le loader reconstruit automatiquement l'architecture, charge les poids et met le modele en `eval()`.

## Promouvoir un modele

`train_runs/` garde toutes les experiences d'entrainement. `trained_models/` sert aux modeles selectionnes pour etre reutilises facilement en inference, attaques adversariales ou comparaisons.

Mode interactif :

```powershell
python .\tools\promote_model.py
```

Promouvoir un checkpoint precis :

```powershell
python .\tools\promote_model.py --checkpoint .\train_runs\<train_run_id>\checkpoint.pth
```

La promotion cree un artifact enrichi dans `trained_models/` sans supprimer le checkpoint du run. Le lien vers le run source est conserve dans l'artifact promu.

## Early stopping

`early_stopping_patience` peut etre defini dans la config YAML ou via `--early-stopping-patience`.

- Si la valeur est vide ou `0`, l'arret anticipe est desactive.
- Si la valeur est `N`, l'entrainement s'arrete apres `N` epochs consecutives sans amelioration de `test_acc`.
- Le meilleur checkpoint reste celui qui a obtenu la meilleure `test_acc`, meme si l'entrainement s'arrete plus tard.

Exemple :

```yaml
early_stopping_patience: 15
```

## Comparer les runs

```powershell
python .\tools\compare_runs.py
```

Le tableau est trie par meilleure accuracy test.

- `best_acc`: meilleure accuracy test atteinte pendant le run
- `min_loss`: plus petite loss test atteinte pendant le run
- `checkpoint`: `ok` si le checkpoint enrichi existe, `missing` sinon

`best_acc` et `min_loss` ne classent pas toujours les runs de la meme facon : l'accuracy mesure les predictions correctes, alors que la loss mesure aussi la confiance du modele.

## Generer un rapport de run

Dernier run disponible :

```powershell
python .\tools\generate_training_report.py
```

Run precis :

```powershell
python .\tools\generate_training_report.py --run-dir .\train_runs\<train_run_id>
```

Le rapport est genere dans `reports/` en Markdown et en PDF.

## Verification rapide

```powershell
python -c "import numpy, matplotlib, PIL, torch, torchvision, tqdm, yaml; print('deps ok')"
python .\tools\compare_runs.py
```
