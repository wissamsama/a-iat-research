# A-IAT Research

Projet local pour explorer CIFAR-10 / CIFAR-100 / GTSRB et iterer proprement sur des experiences d'entrainement CNN.

## Structure actuelle

- `configs/simple_cnn.yaml`: configuration unique de `SimpleCNN`; le dataset se choisit dans ce fichier
- `models/simple_cnn.py`: architecture CNN simple compatible avec une taille d'entree configurable
- `models/registry.py`: registre des architectures et creation des checkpoints enrichis
- `models/loading.py`: chargement automatique d'un checkpoint enrichi en modele PyTorch
- `tools/train.py`: outil utilisateur pour lancer un entrainement
- `tools/datasets.py`: outil et registry unique des datasets, transforms, normalisation et loaders
- `training/utils.py`: seed, device et utilitaires generaux
- `training/experiment.py`: creation de train runs, logs CSV, configs et summaries
- `attacks/fgsm.py`: implementation de l'attaque FGSM
- `evaluation/attack_evaluator.py`: boucle d'evaluation clean/adversarial
- `tools/visualize_data.py`: outil utilisateur pour visualiser CIFAR-10 / CIFAR-100 / GTSRB / STURM-Flood / Sen1Floods11
- `tools/compare_train_runs.py`: outil utilisateur pour comparer les entrainements termines
- `tools/promote_model.py`: outil utilisateur pour promouvoir un checkpoint de train run vers `trained_models/`
- `configs/fgsm.yaml`: configuration d'attaque FGSM par defaut
- `tools/run_attack.py`: outil utilisateur pour lancer une attaque adversariale
- `tools/compare_attack_runs.py`: outil utilisateur pour comparer les attaques lancees
- `data/`: datasets locaux, non versionnes
- `train_runs/`: traces d'entrainement et checkpoints enrichis, non versionnes
- `attack_runs/`: traces d'attaques adversariales, non versionnees
- `trained_models/`: modeles selectionnes pour inference/attaques, non versionnes

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Dependances principales : `numpy`, `matplotlib`, `Pillow`, `torch`, `torchvision`, `tqdm`, `PyYAML`, `rasterio`.

## Visualiser les donnees

Mode interactif :

```powershell
python .\tools\visualize_data.py
```

Le menu s'adapte au dataset choisi : CIFAR/GTSRB proposent split et classes, STURM-Flood propose Sentinel-1/Sentinel-2, et Sen1Floods11 propose ses sources hand/weak/permanent-water.

Exemples avec arguments :

```powershell
python .\tools\visualize_data.py --dataset cifar10 --samples-per-class 6
python .\tools\visualize_data.py --dataset cifar10 --class-name cat --samples-per-class 12
python .\tools\visualize_data.py --dataset cifar100 --samples-per-class 2 --max-classes 12 --columns 6
python .\tools\visualize_data.py --dataset gtsrb --split train --samples-per-class 4 --max-classes 12 --columns 6
python .\tools\visualize_data.py --dataset gtsrb --split test --class-name 14 --samples-per-class 8 --columns 4
python .\tools\visualize_data.py --dataset sturm_flood --split sentinel1 --samples-per-class 4 --columns 4
python .\tools\visualize_data.py --dataset sturm_flood --split sentinel2 --samples-per-class 4 --columns 4
python .\tools\visualize_data.py --dataset sen1floods11 --split hand_s1 --samples-per-class 4 --columns 4
python .\tools\visualize_data.py --dataset sen1floods11 --split weak_s1 --samples-per-class 4 --columns 4
python .\tools\visualize_data.py --dataset sen1floods11 --split weak_s2 --samples-per-class 4 --columns 4
python .\tools\visualize_data.py --dataset sen1floods11 --split perm_water_s1 --samples-per-class 4 --columns 4
```

Aucune image n'est enregistree sur disque. L'affichage se fait dans une fenetre Matplotlib. Pour GTSRB, les classes sont numerotees de `class_00000` a `class_00042`; tu peux aussi utiliser directement un numero comme `14`. L'alias `gtrsb` est accepte. Pour STURM-Flood, `--split sentinel1` affiche Sentinel-1, `--split sentinel2` affiche Sentinel-2, et chaque exemple montre l'image source a gauche et le masque d'inondation a droite. Pour Sen1Floods11, `--split hand_s1`, `hand_s2`, `weak_s1`, `weak_s2` ou `perm_water_s1` affiche les paires GeoTIFF source + masque disponibles localement; les fichiers temporaires du telechargement ne sont pas utilises.

## Entrainer une experience

La configuration par defaut est dans `configs/simple_cnn.yaml`, qui pointe vers CIFAR-10. Le script d'entrainement reste le meme pour tous les datasets; on change seulement le fichier YAML.

```powershell
python .\tools\train.py --config .\configs\simple_cnn.yaml
```

Switcher de dataset sans changer de script : modifier `dataset`, `data_dir` et `experiment_name` dans `configs/simple_cnn.yaml`, puis relancer la meme commande.

Exemples de valeurs :

```yaml
dataset: cifar10
data_dir: data/CIFAR-10
experiment_name: simplecnn_cifar10
```

```yaml
dataset: cifar100
data_dir: data/CIFAR-100
experiment_name: simplecnn_cifar100
```

```yaml
dataset: gtsrb
data_dir: data/GTSRB
experiment_name: simplecnn_gtsrb
```

Override de parametres sans modifier la config :

```powershell
python .\tools\train.py --experiment-name simplecnn_lr0005 --lr 0.0005 --epochs 20 --batch-size 128 --early-stopping-patience 8
```

Options disponibles :

- `--config`: chemin vers la config YAML
- `--experiment-name`: nom de l'experience
- `--epochs`: nombre maximum d'epochs
- `--batch-size`: taille de batch
- `--lr`: learning rate
- `--seed`: seed aleatoire
- `--early-stopping-patience`: nombre d'epochs sans amelioration de `test_acc` avant arret anticipe, `0` pour desactiver
- `--training-mode`: `classic` ou `adversarial`
- `--clean-loss-lambda`: poids de la loss clean en adversarial training
- `--adv-training-epsilon`: epsilon FGSM utilise pendant l'adversarial training

Parametres dataset importants dans les configs :

- `dataset`: `cifar10`, `cifar100` ou `gtsrb`
- `data_dir`: dossier local du dataset
- `input_size`: taille carree des images donnees au modele, par exemple `32` ou `64`
- `num_classes`: optionnel, inferre automatiquement depuis le dataset si absent
- `normalization`: optionnel, les valeurs par defaut viennent de la registry dataset

La registry dataset est la source de verite pour `num_classes`, `input_shape`, `normalization`, les transforms et les loaders. `input_size` est un parametre experimental: CIFAR peut etre redimensionne depuis `32x32`, et GTSRB est toujours redimensionne vers cette taille car ses images natives ont des dimensions variables.

Parametres training importants :

- `training_mode`: `classic` pour l'entrainement standard, `adversarial` pour la loss mixte clean + FGSM
- `clean_loss_lambda`: poids de la loss clean dans `lambda * L(clean) + (1 - lambda) * L(adversarial)`; ignore en mode `classic`
- `adv_training_epsilon`: intensite FGSM en espace pixel `[0, 1]`; ignore en mode `classic`
- `adv_training_attack`: attaque utilisee pour l'adversarial training, actuellement `fgsm`

Exemple adversarial training :

```yaml
training_mode: adversarial
clean_loss_lambda: 0.5
adv_training_epsilon: 0.03
adv_training_attack: fgsm
```

Afficher l'etat des datasets locaux :

```powershell
python .\tools\datasets.py
python .\tools\datasets.py --dataset gtsrb
```

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
- `input_shape`: forme attendue en entree, derivee de `input_size`
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

## Lancer une attaque

La premiere attaque supportee est FGSM. Elle reutilise le dataset, la normalisation et le chemin `data_dir` stockes dans le checkpoint enrichi du modele attaque. L'epsilon est exprime dans l'espace pixel `[0, 1]`, avant normalisation. Le modele charge reste normalise avec les statistiques stockees dans le checkpoint.

Lancer depuis la config par defaut :

```powershell
python .\tools\run_attack.py --config .\configs\fgsm.yaml
```

Attaquer un modele promu en surchargeant la config :

```powershell
python .\tools\run_attack.py --config .\configs\fgsm.yaml --model .\trained_models\<model>.pth --epsilon 0.03
```

Par defaut, `max_samples: null` attaque tout le split choisi. Pour un test rapide, surcharge `--max-samples 128`.

Test rapide sur un sous-ensemble :

```powershell
python .\tools\run_attack.py --config .\configs\fgsm.yaml --max-samples 128
```

Chaque attaque cree un dossier dans `attack_runs/` :

```txt
attack_runs/
  09-06-2026_12-00-00_fgsm_simplecnn_cifar10/
    config.yaml
    metrics.csv
    summary.json
```

Comparer les attaques :

```powershell
python .\tools\compare_attack_runs.py
```

Contenu utile :

- `clean_acc`: accuracy sans attaque
- `adversarial_acc`: accuracy apres attaque
- `accuracy_drop`: difference entre clean et adversarial
- `success_rate_on_clean_correct`: taux de succes sur les exemples correctement classes au depart

## Early stopping

`early_stopping_patience` peut etre defini dans la config YAML ou via `--early-stopping-patience`.

- Si la valeur est vide ou `0`, l'arret anticipe est desactive.
- Si la valeur est `N`, l'entrainement s'arrete apres `N` epochs consecutives sans amelioration de `test_acc`.
- Le meilleur checkpoint reste celui qui a obtenu la meilleure `test_acc`, meme si l'entrainement s'arrete plus tard.

Exemple :

```yaml
early_stopping_patience: 15
```

## Comparer les entrainements

```powershell
python .\tools\compare_train_runs.py
```

Le tableau est trie par meilleure accuracy test.

- `best_acc`: meilleure accuracy test atteinte pendant le run
- `min_loss`: plus petite loss test atteinte pendant le run
- `checkpoint`: `ok` si le checkpoint enrichi existe, `missing` sinon

`best_acc` et `min_loss` ne classent pas toujours les runs de la meme facon : l'accuracy mesure les predictions correctes, alors que la loss mesure aussi la confiance du modele.



## Profilage factuel de FloodCastBench

Les scripts suivants inspectent les fichiers locaux reels dans `data/FloodCastBench/` et generent un profil exploitable avant d'implementer les modeles spatio-temporels.

```powershell
python .\scripts\inspect_floodcastbench_structure.py --data_dir data\FloodCastBench --output_dir outputs\dataset_profile
python .\scripts\build_floodcastbench_manifest.py --data_dir data\FloodCastBench --output_dir outputs\dataset_profile
python .\scripts\profile_floodcastbench_rasters.py --data_dir data\FloodCastBench --output_dir outputs\dataset_profile --max_files_per_group 5
python .\scripts\check_floodcastbench_temporal_windows.py --data_dir data\FloodCastBench --output_dir outputs\dataset_profile --input_window 5 --horizons 20 72 144
python .\scripts\visualize_floodcastbench_samples.py --data_dir data\FloodCastBench --output_dir outputs\dataset_profile --threshold 0.01
```

Sorties principales :

- `outputs/dataset_profile/floodcastbench_structure.md`
- `outputs/dataset_profile/floodcastbench_manifest.csv`
- `outputs/dataset_profile/events_summary.csv`
- `outputs/dataset_profile/variables_summary.csv`
- `outputs/dataset_profile/raster_statistics.csv`
- `outputs/dataset_profile/temporal_index_summary.csv`
- `outputs/dataset_profile/supervised_learning_samples_preview.csv`
- `outputs/dataset_profile/dataset_questions_answered.md`
- `outputs/dataset_profile/figures/`

Le rapport separe les faits verifies depuis les fichiers locaux des points encore inconnus ou a verifier dans la documentation officielle.

## Verification rapide

```powershell
python -c "import numpy, matplotlib, PIL, torch, torchvision, tqdm, yaml; print('deps ok')"
python .\tools\compare_train_runs.py
```



