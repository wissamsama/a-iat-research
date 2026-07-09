# DIFF-SPARSE v1 & v2 — Spécification technique exhaustive

De la donnée brute à la sortie, chaque transformation, chaque couche, chaque
(méta)paramètre, exactement tels qu'implémentés dans ce repo. Puis une partie
interprétation : gain potentiel de chaque modification v2 et pourquoi.

Fichiers de référence :
- v1 : `models/diff_sparse_v1.py`, `datasets/floodcastbench_diff_sparse_v1_dataset.py`,
  `tools/train_floodcastbench_diff_sparse_v1.py`, `tools/evaluate_floodcastbench_diff_sparse_v1.py`,
  `configs/floodcastbench_diff_sparse_v1_highfid_60m.yaml`
- v2 : `models/diff_sparse_v2.py`, `datasets/floodcastbench_diff_sparse_v2_dataset.py`,
  `tools/train_floodcastbench_diff_sparse_v2.py`, `tools/evaluate_floodcastbench_diff_sparse_v2.py`,
  `configs/floodcastbench_diff_sparse_v2_highfid_60m.yaml`

---

# PARTIE 0 — La donnée brute (partagée v1/v2)

## 0.1 Ce qui existe sur disque

Racine : `/home/wissam/utem-workspace/data/FloodCastBench`, événement Australia,
haute fidélité, 60 m.

- **Eau (hauteur d'inondation)** : `High-fidelity flood forecasting/60m/Australia/<timestamp>.tif`.
  2881 fichiers GeoTIFF, un par pas de temps. Nom du fichier = timestamp entier
  en secondes. Grille **536×536**, valeurs en **mètres** de hauteur d'eau.
  Pas de temps : **300 s** (5 min) entre frames consécutives (vérifié à l'exécution
  par `_validate_uniform_timestamps`).
- **DEM (élévation)** : `Relevant data/DEM/Australia_DEM.tif`. Une seule frame
  statique, mètres.
- **Pluie** : `Relevant data/Rainfall/Australia flood/*.tif`. Une frame toutes
  les **1800 s** (30 min) — donc 6× plus grossier temporellement que l'eau.

## 0.2 Lecture brute (`_read_raster`)

Pour chaque `.tif` : `rasterio.read(1)` → tableau float32. Les `nodata` sont
remplacés par 0.0, puis `nan_to_num(nan=0, posinf=0, neginf=0)`. Aucune autre
transformation à la lecture.

## 0.3 Alignement DEM et pluie sur la grille eau

- DEM : lu une fois, redimensionné à 536×536 par interpolation **bilinéaire**
  (`_resize_array`, `align_corners=False`).
- Pluie : pour un timestamp d'eau `t`, on prend l'index pluie
  `min(t // 1800, len-1)` (association au pas de pluie le plus proche par le bas),
  lu et redimensionné bilinéairement à 536×536, puis **mis en cache** par index.

## 0.4 Découpage en splits

`split_frame_ranges` dérive les bornes à partir des fenêtres canoniques de 20
frames × comptes 116/14/14 :
- **train** : frames [0, 2320)
- **val** : frames [2320, 2600)
- **test** : frames [2600, 2881)

Ces bornes sont fixes et **indépendantes** de context/prediction_length, ce qui
garantit la comparabilité avec FNO+/persistence quel que soit le réglage.

## 0.5 Statistiques de normalisation (train uniquement)

`compute_v1_normalization_stats` parcourt uniquement les frames [0, 2320) et
calcule, en float64, moyenne/std/min/max par canal :

| Canal | moyenne | std | min | max |
|---|---:|---:|---:|---:|
| eau | 0.10608 | 0.29114 | 2.17e-5 | 15.0139 |
| DEM | (séparé) | (séparé) | | |
| pluie | (séparé) | (séparé) | | |

`std` est planchée à `min_std=1e-6`. Une **seule statistique eau partagée** sert
au contexte ET aux cibles (pas de re-ciblage de la persistence entre espaces).
Précalculé dans `outputs/floodcastbench_normalization/diff_sparse_v1_water_dem_rainfall_train_stats.json`.

Constante dérivée cruciale : la **valeur normalisée de 0 m d'eau** =
`(0 − 0.10608)/0.29114 = −0.36437`. C'est le plancher physique (profondeur ≥ 0).

---

# PARTIE 1 — DIFF-SPARSE v1

## 1.1 Construction d'un échantillon (`FloodCastBenchDiffSparseV1Dataset.__getitem__`)

Paramètres v1 : `context_length=12`, `prediction_length=12`, donc
`window_length=24`. `patch_size=64`. Train : stride 1 (2297 fenêtres). Val/test :
stride 20 (13 fenêtres chacun, celles qui rentrent entièrement).

Pour un index de fenêtre :
1. On prend les 24 frames consécutives `frames[start : start+24]`.
2. **Mode patch** :
   - train (`patch_mode="random"`) : origine aléatoire `(y0, x0)` uniforme,
     crop 64×64.
   - val/test (`patch_mode="full"`) : champ entier 536×536 (l'évaluateur le
     tuile lui-même).
3. On empile l'eau des 24 frames → tenseur `[24, ph, pw]`. Idem pluie via
   `_rainfall_for_timestamp` par timestamp → `[24, ph, pw]`. DEM cropé →
   `[1, ph, pw]`. Timestamps → `[24]`.
4. **Normalisation** (si stats fournies) : chaque canal standardisé
   `(x − mean)/std` avec sa propre stat (eau partagée contexte/cible).
5. Découpe temporelle :
   - `context_true = water[:12]` → `[12, ph, pw]` (historique vrai, non masqué)
   - `target = water[12:]` → `[12, ph, pw]` (cibles futures)

### Masque capteur (`generate_sensor_mask`)

Statique par échantillon, binaire `[1, ph, pw]`. `missing_rate ∈ {0, 0.5, 0.95}`.
Nombre exact de capteurs = `round((1−missing_rate)·H·W)`, positions tirées par
`randperm`. À l'éval, banque de 10 masques fixes (`eval_mask_bank`), appliqués en
round-robin par index de fenêtre, graine `eval_mask_seed=1234` (protocole du
papier).

### Masquage par bruit (`apply_observation_masking`, Algorithme 1 du papier)

Sur le contexte uniquement :
```
context_masked = context_true · M + (1 − M) · fill
```
où `fill = bruit gaussien standard` (mode `noise`, défaut papier) ou `0`
(mode `zeros`, ablation). Aux cellules **sans** capteur, la vraie valeur est
remplacée par du bruit — signal explicite « ignore ici ». Si `missing_rate=0`
(masque tout-à-1), `context_masked == context_true`.

### Contrat du sample renvoyé
```
context_water_masked  [12, ph, pw]   historique masqué normalisé
context_water_true    [12, ph, pw]   historique vrai (persistence/diagnostics)
sensor_mask           [1, ph, pw]
dem                   [1, ph, pw]
rainfall              [24, ph, pw]    contexte + futur (pluie exogène dense)
timestamps            [24]            secondes
target                [12, ph, pw]
```

## 1.2 Contrat d'entraînement (une-frame, `prepare_model_batch`)

Le papier s'entraîne à prédire **une seule frame suivante**. Donc du sample on
extrait :
```
context_water_masked  [B, 12, 64, 64]
sensor_mask           [B, 1, 64, 64]
dem                   [B, 1, 64, 64]
rainfall_context      [B, 12, 64, 64]   = rainfall[:, :12]
timestamps_context    [B, 12]
target                [B, 1, 64, 64]     = target[:, 0:1]  (uniquement la 1re cible)
```
Batch 32.

## 1.3 Le modèle (`DiffSparseV1Model`) — 1 437 570 paramètres

Deux composants : un encodeur de contexte temporel (73 505 params) et un UNet
conditionnel diffusers (1 364 065 params).

### 1.3.1 TemporalContextEncoder (= HiddenStateNet du repo de référence)

But : réduire les 12 frames de contexte à **12 tokens** (un par pas de temps),
fournis en `encoder_hidden_states` à la cross-attention du UNet. **Ce n'est PAS
une carte spatiale pixel-alignée** — c'est une séquence temporelle, exactement
comme des tokens de texte dans Stable Diffusion.

Empilement d'entrée `[B, C, T=12, H=64, W=64]`, C = 4 canaux :
`[eau masquée, DEM (broadcast sur T), masque capteur (broadcast sur T), pluie contexte]`.

3 blocs `TemporalDownBlock` (canaux [16, 32, 64], `groups=8`), chacun :
- `Conv3d(kernel=(1,3,3), sans padding)` → GroupNorm → SiLU
- `Conv3d(kernel=(1,3,3), sans padding)` → GroupNorm → SiLU
- `AvgPool3d((1,2,2))`

Le noyau `(1,3,3)` ne convolue **que spatialement**, jamais sur l'axe temporel :
chaque frame est réduite indépendamment. Arithmétique spatiale (non paddée) à
partir de 64 :
`64 → (−2−2) 60 → pool 30 → 26 → 13 → 9 → pool 4`. Donc chaque frame finit en 4×4.
`Conv3d(64→1, kernel 1)` → un scalaire par (t, position 4×4) → aplati **16** par token.

Covariables temporelles (`encode_covariates`) : pour chaque timestamp,
7 features brutes = `[t/864000, sin/cos(2π·f·hod) pour f∈{1,2,4}]` où
`hod = (t mod 86400)/86400` (fraction d'heure-du-jour). Concaténées à chaque
token → **16 + 7 = 23** par token.

Projection linéaire partagée `Linear(23 → 32)` → séquence `[B, 12, 32]`.
`context_embedding_dim = 32`.

> Détail load-bearing : `token_linear` est **enregistrée à la construction** (taille
> calculée analytiquement), pas paresseusement au premier forward — sinon elle
> serait exclue de l'optimiseur créé avant ce forward (bug corrigé, test de
> non-régression présent).

### 1.3.2 UNet conditionnel (`diffusers.UNet2DConditionModel`)

Config exacte : `in_channels=1` (juste x_noisy), `out_channels=1`,
`block_out_channels=(16,32,32,64)`, `layers_per_block=2`, `norm_num_groups=16`,
`cross_attention_dim=32`, `dropout=0`,
`down_block_types=(DownBlock2D, CrossAttnDownBlock2D, CrossAttnDownBlock2D, DownBlock2D)`,
mirroir en up. `cross_attention_blocks=2` → attention aux **2 niveaux du milieu**.

Structure interne (patch 64×64) :
- `conv_in` : Conv2d(1 → 16, 3×3, pad 1).
- Embedding de timestep : sinusoïdal (dim 16) → MLP → time_embed (dim 64).
- **Down** :
  - Bloc 0 `DownBlock2D` @ 64×64 : 2× ResnetBlock2D, downsample → 32×32.
  - Bloc 1 `CrossAttnDownBlock2D` @ 32×32 : 2× (ResnetBlock2D + Transformer2DModel),
    downsample → 16×16.
  - Bloc 2 `CrossAttnDownBlock2D` @ 16×16 : 2× (Resnet + Transformer2D),
    downsample → 8×8.
  - Bloc 3 `DownBlock2D` @ 8×8 : 2× Resnet, pas de downsample.
- **Mid** `UNetMidBlock2DCrossAttn` @ 8×8 : Resnet + attention + Resnet.
- **Up** : miroir avec skip-connections, up-sampling.
- `conv_norm_out` (GroupNorm 16) → SiLU → `conv_out` Conv2d(16 → 1).

Chaque `Transformer2DModel` : GroupNorm → proj_in → BasicTransformerBlock
(self-attention spatiale + **cross-attention vers les 12 tokens de contexte** +
feed-forward) → proj_out. `only_cross_attention=False` (défaut diffusers) → la
self-attention spatiale est **incluse**. La cross-attention n'opère qu'aux
résolutions 32×32 et 16×16.

### 1.3.3 Schedule de diffusion (exact, `steps=20`, `x0`)

`betas = linspace(1e-4, 1.0, 20)` en float64. `alphas = 1−betas`,
`alpha_cumprod = cumprod(alphas)`. Valeurs clés :
- `ᾱ[0] = 0.9999`, `ᾱ[10] = 0.02587`, **`ᾱ[19] = 0.0` exactement**.

Ce dernier point (β_end=1.0, Table 2 du papier) est le **fix SNR-terminal** : au
pas terminal le forward est du bruit **pur**, donc échantillonner depuis N(0,I)
est exactement dans la distribution d'entraînement. Buffers enregistrés :
`sqrt_alpha_cumprod`, `sqrt_one_minus_alpha_cumprod`, et coefficients du
postérieur DDPM :
```
posterior_coef_x0[n] = β[n]·√ᾱ[n−1] / (1−ᾱ[n])
posterior_coef_xt[n] = √α[n]·(1−ᾱ[n−1]) / (1−ᾱ[n])
posterior_variance[n] = β[n]          # "Option 1" du repo de référence (β_t brut,
                                        # pas la β-tilde plus serrée)
```
`posterior_coef_xt[19] = 0` → le premier pas inverse **jette entièrement** le
bruit initial.

## 1.4 Un pas d'entraînement (`training_step_loss`, Algorithme 1)

1. `context_embedding = encode_context(batch)` → `[B, 12, 32]`.
2. `t ~ Uniform{0..19}` par échantillon → `[B]`.
3. `noise ~ N(0, I)`, forme de la cible.
4. **q_sample** : `x_noisy = √ᾱ[t]·target + √(1−ᾱ[t])·noise`.
5. `prediction = denoise(x_noisy, t, context_embedding)` (le UNet prédit x0).
6. `loss = MSE(prediction, target)` (moyenne sur tous les pixels/batch).

Optimiseur : **AdamW**, lr=1e-3, weight_decay=0. Scheduler
**ReduceLROnPlateau** (mode min sur val_loss, factor 0.5, **patience 15**).
Grad clip global à 1.0. `epochs=300`. Graine 42/7/123.

> Deux méta-choix v1 non-papier, documentés : (a) `epochs=300` (papier 40) —
> convergence lente réelle de la cross-attention pure diagnostiquée
> empiriquement (fixed-t=0 converge net, random-t est lent) ; (b)
> `lr_patience=15` (papier 3) — la métrique de val ici est un proxy 1-pas moins
> monotone que le rollout génératif complet du repo de référence, donc plus de
> patience nécessaire pour éviter que le LR s'effondre avant la percée.

### Sélection de checkpoint
`val_loss` = même perte 1-pas mais RNG fixé (`val_seed=1234`, état sauvé/restauré).
`checkpoint_best.pth` = argmin val_loss.

## 1.5 Échantillonnage (`sample`, Algorithme 2, DDPM ancestral)

Depuis `x_T ~ N(0, I)`, pour n de 19 à 0 :
1. `x0_hat = denoise(x_t, n, context_embedding)`.
2. `mean = posterior_coef_x0[n]·x0_hat + posterior_coef_xt[n]·x_t`.
3. si n>0 : `x_t = mean + √(posterior_variance[n])·z`, `z~N(0,I)` ; sinon `x_t=mean`.
20 passes UNet par frame échantillonnée. (v1 : pas de clamp.)

## 1.6 Évaluation (`evaluate_floodcastbench_diff_sparse_v1.py`)

- Champ 536×536 tuilé en 64×64, **stride 48** (recouvrement), fenêtre de mélange
  **Hann** centrée-cellule ; les prédictions qui se recouvrent sont moyennées
  pondérées.
- **Rollout autorégressif** sur `prediction_length` pas : à chaque pas on
  échantillonne la frame suivante (20 passes), on la **rajoute au contexte**
  (glissant), et on avance la pluie/timestamps. `rollout_remask=false` (défaut
  v1 après investigation) : la prédiction dense ré-entre telle quelle.
- **num_scenarios** : 2 (val) / 8 (test) — M échantillons par tuile, moyennés
  pour le point-forecast.
- Métriques : NRMSE (éq. 15, dénom = max−min des obs), NACRPS (éq. 16, CRPS
  empirique), RMSE/MAE normalisés et physiques, path IoU / propagation IoU
  (mais **v1 ne calcule que l'horizon final**), comparaison persistence
  oracle/sparse.

## 1.7 Sortie
Cartes de hauteur d'eau prédites (12 frames), moyenne des scénarios ;
inverse-transformées `pred·std + mean` pour les métriques physiques.

---

# PARTIE 2 — DIFF-SPARSE v2 : chaque différence exacte

v2 garde **l'identité DIFF-SPARSE** (DDPM conditionnel masqué x0, β∈[1e-4,1.0]
SNR-terminal 0, masquage bruit, tokens temporels + cross-attention diffusers,
entraînement 1-pas + rollout). Tout le reste devient un levier. Modèle :
**5 538 546 paramètres** (UNet 5 428 929 + context 74 273 + spatial 35 344).

## 2.0 Réglages changés (config)
| | v1 | v2 |
|---|---:|---:|
| context_length | 12 | **24** (2 h d'historique ; fenêtre 36) |
| prediction_length | 12 | 12 |
| diffusion steps | 20 | **40** |
| context_embedding_dim | 32 | **64** |
| unet_channels | [16,32,32,64] | **[32,64,64,128]** |
| espace cible | absolu | **delta** |
| batch/lr/patience | 32 / 1e-3 / 15 | idem |
| epochs | 300 | 300 (early stop 60) |

## 2.1 Prédiction en DELTA (le levier n°1)

**Statistique mesurée** (`compute_..._delta_stats.py`, train uniquement) : std des
différences frame-à-frame = **0.0006965 m**, vs std eau 0.29114 m → le signal
réel par pas est **~418× plus petit** que le champ absolu. (C'est aussi pourquoi
la persistence oracle est un mur : RMSE ~0.004 normalisé.)

`DeltaSpec` définit l'échelle normalisée `scale = 0.0006965/0.29114 = 0.0023925`.

- **base** = dernière frame de contexte telle qu'**observée** :
  `context_water_true[-1] · sensor_mask` (vrai aux capteurs, remplissage
  moyenne=0 ailleurs). En dense c'est exactement la vraie dernière frame ; dès le
  pas 2 du rollout c'est la propre prédiction dense du modèle.
- **cible d'entraînement** = `(next_frame − base)/scale` (donc ~variance unité).
- **reconstruction** = `absolute = base + scale·prediction`. **On récupère une
  vraie carte de profondeur en sortie**, toutes les métriques se calculent
  comme en absolu.
- **clamp** : plancher physique par-pixel dans le sampler
  `(−0.36437 − base)/scale` (profondeur ≥ 0), + plafond de sécurité numérique
  `(2·max−mean)/std = 102.78` **uniquement** sur les reconstructions single-shot
  (pushforward), pour éviter l'explosion qui avait déstabilisé un pilote.

## 2.2 Conditionnement hybride (SpatialContextEncoder, +35 344 params)

v1 : conditionnement **uniquement** par tokens temporels (aucun accès pixel du
UNet au contexte). v2 **ajoute** un chemin spatial pixel-aligné, concaténé à
`x_noisy` à l'entrée du UNet (donc `UNet in_channels = 1 + 16 = 17`).

Entrée `[B, C, 64, 64]`, C = 12 (eau masquée) + 1 (masque) + 1 (DEM) + 12 (pluie
contexte) + 1 (**pluie du pas prédit**) + 11 (**deltas de contexte** =
`context[1:] − context[:−1]`) = **38 canaux** (pour context_length 24 : 24+1+1+24+1+23).
Stack :
- Conv2d(→32, 3×3, pad 1) → GroupNorm → SiLU
- Conv2d(32→32, 3×3, pad 1) → GroupNorm → SiLU
- Conv2d(32→16, 3×3, pad 1)
→ 16 cartes pleine résolution concaténées à x_noisy.

Les tokens temporels (chemin v1) restent **inchangés** à côté.

## 2.3 Forçage pluie du pas prédit
`rainfall_target = rainfall[:, context_length]` = la pluie qui tombe **pendant**
l'intervalle prédit. Driver causal direct des pixels nouvellement inondés ;
v1 prédisait la frame suivante sans connaître la pluie qui la produit.

## 2.4 Sélection de checkpoint par ROLLOUT génératif réel (`RolloutValidator`)
v1 sélectionnait sur un proxy 1-pas. v2 : rollout autorégressif complet (12 pas,
échantillonnage ancestral réel, poids EMA) sur un petit jeu **fixe** de tuiles
(2 fenêtres × 4 tuiles = 8), RNG fixé, toutes les 5 epochs. `checkpoint_best` =
argmin de ce **rollout_val_rmse** (espace absolu). Early stopping : 60 epochs
sans amélioration.

## 2.5 Réduction du biais d'exposition (pushforward, 25% des batches)
Sur 25% des batches : un forward **no-grad** approxime la prédiction du modèle au
pas 1 (x0 depuis bruit pur au pas terminal — exactement le 1er pas du sampler),
on la **remplace** dans le contexte, et le pas de gradient entraîne le pas 2
contre la vraie frame, cible exprimée **relativement à cette base imparfaite**.
Le modèle apprend à corriger sa propre dérive — que l'entraînement 1-pas
teacher-forcé n'exerce jamais.

## 2.6 Loss pondérée par changement d'état
`change_weight=3.0` : les pixels dont l'état sec/inondé (seuil 0.001 m) **change**
entre base et cible reçoivent poids `1+3 = 4`, normalisé pour garder l'échelle.
C'est exactement la population du propagation-path IoU, sinon une fraction
négligeable de la MSE. `snr_gamma=null` (pondération Min-SNR dispo mais off).

## 2.7 EMA, bf16, augmentation
- **EMA** decay 0.999 des poids ; rollout-val et éval finale utilisent l'EMA.
- **bf16 autocast** sur le forward d'entraînement (sampling/val restent fp32).
- **Augmentation diédrale** (8 flips/rotations, dataset v2) : tous les canaux
  spatiaux transformés ensemble (cohérent physiquement, la gravité passe par le
  DEM transformé aussi). Dataset mono-événement → régularisation clé.
- Option `missing_rate_range` (off) : un seul modèle robuste à toutes les
  sparsités.

## 2.8 Évaluation v2 (différences)
- **stride tuiles 32** (v1 : 48) : recouvrement Hann plus fort, moins de coutures.
- Rollout en delta : reconstruit l'absolu par tuile (base = observé au pas 1,
  puis propre prédiction), clamp physique.
- **MultiHorizonPathAccumulator** : path IoU + propagation IoU à **chaque** pas
  (v1 : horizon final seulement), fusionnés dans le CSV per-step.
- **Masques par vote majoritaire** (colonnes `_median`) : masque via la **médiane
  par-pixel** des scénarios — pour tout seuil γ, `médiane > γ ⇔ majorité des
  scénarios > γ` — règle de décision optimale sous incertitude, vs seuiller la
  moyenne qui étale le front.

---

# PARTIE 3 — Interprétation : pourquoi chaque modif, gain attendu

## 3.1 Delta (le gros levier) — pourquoi ça marche
En absolu, « battre la persistence » = prédire mieux que « copie de la dernière
frame », un mur à ~0.004 normalisé à 5 min d'espacement, parce que 99.x% de
chaque frame est identique à la précédente. Le modèle absolu gaspille sa capacité
à **ré-encoder** ce qui est déjà connu ; le vrai signal (le changement, ~418× plus
petit) est noyé dans la MSE. En delta, la cible **est** le changement (variance
unité), donc « battre la persistence » devient « prédire mieux que zéro » — et le
changement, ce sont précisément les pixels nouvellement inondés que le
propagation-path IoU mesure. **Gain observé au pilote** : RMSE normalisé pooled
~0.0027 (val, dense) vs ~0.065 pour v1 — première fois qu'une variante bat la
persistence en dense. C'est le levier qui explique l'essentiel du saut.

> Réserve : ce gain se mesure ici à **contexte 24 (v2) vs 12 (v1) vs 1 (FNO+)** ;
> la comparaison n'est pas à budget d'information égal. À réévaluer sur test,
> multi-seed, et sous sparsité, avant toute conclusion de supériorité.

## 3.2 Conditionnement hybride — pourquoi
La cross-attention temporelle pure ne donne au UNet **aucun** accès pixel-à-pixel
au contexte : elle dit « ce qui s'est passé, par pas de temps, en résumé
grossier », pas « quelle est la profondeur à CE pixel ». Or la localisation du
front d'inondation (ce que path/propagation IoU mesure) est un problème
per-pixel. Preuve interne : le variant concat convergeait à val_loss 1-pas
~0.005 là où l'attention-seule stagnait ~3.7 pendant 70+ epochs (écart ~700× sur
l'objectif). Gain attendu : sur **toutes** les métriques, surtout les masques.

## 3.3 Pluie du pas prédit — pourquoi
FloodCastBench est piloté par la pluie. v1 prédisait la frame `t+1` sans
connaître la pluie tombant entre `t` et `t+1` — le driver causal direct de la
nouvelle inondation. La donner en forçage exogène (même statut que les entrées
pluie de FNO+) cible directement le propagation-path IoU.

## 3.4 Clamp physique — pourquoi (le fix propagation IoU)
La profondeur ne peut pas être négative, or ~24-29% des prédictions v1/FNO+
l'étaient. Pire pour les masques : le bruit d'échantillonnage DDPM oscillant
autour de γ=0.001 m crée d'énormes comptages « nouvellement inondé » parasites à
chaque pas (v1 : propagation IoU ~1e-4, ~414k faux positifs vs 38 vrais à h216).
Clamper x0 ≥ 0 à chaque pas inverse : un pixel au plancher est à exactement 0 m et
ne peut franchir γ que si le modèle prédit vraiment de l'eau. Gain ciblé
directement sur la métrique headline.

## 3.5 Sélection par rollout génératif — pourquoi
Le proxy 1-pas mesure le débruitage, pas la génération, et a déjà induit le
scheduler en erreur (plateau de v1). Sélectionner sur le vrai rollout RMSE = la
famille de métrique que l'éval finale rapporte, donc le checkpoint choisi est
optimal **pour ce qui compte**, pas pour un proxy corrélé.

## 3.6 Pushforward — pourquoi
L'entraînement 1-pas ne voit jamais les erreurs que le rollout autorégressif
compose — cause structurelle de la dégradation en horizon. FNO+ doit une partie
de sa robustesse au fait d'être entraîné directement sur 19 pas. Entraîner le
modèle à corriger sa **propre** base imparfaite réduit ce biais d'exposition, gain
attendu surtout aux horizons tardifs.

## 3.7 Loss pondérée changement — pourquoi
Distribution ultra-déséquilibrée (quasi tout sec/statique). Les pixels de front
peu profonds — ceux que γ=0.001 m mesure — pèsent presque rien dans une MSE
uniforme. Peser ×4 les pixels qui changent d'état remonte mécaniquement le
gradient là où le propagation IoU se joue, sans casser le mapping linéaire
normalisé↔physique.

## 3.8 EMA / bf16 / augmentation / contexte 24 / steps 40 — pourquoi
- **EMA** : gain de qualité d'échantillonnage quasi systématique en diffusion,
  gratuit.
- **bf16** : ~1.5-2× débit → 2× d'epochs à coût égal.
- **Augmentation diédrale** : un seul événement, une seule région → la
  régularisation spatiale est le principal rempart au sur-apprentissage.
- **Contexte 24** : 2 h d'historique ; à fenêtre de prédiction 12, le contexte
  reste 2× plus long que la portée, donc même au dernier pas évalué la moitié du
  contexte est encore de la vraie donnée (ancrage réel fort — mais **aggrave le
  biais de comparaison** vs FNO+ à contexte 1).
- **Steps 40** : processus inverse plus fin, SNR terminal reste 0 (identité
  préservée), coût inférence ~2×.

## 3.9 Vote majoritaire (éval seule) — pourquoi
Seuiller la moyenne des scénarios étale le front (valeurs intermédiaires
ambiguës autour de γ). La médiane par-pixel encode le vote majoritaire pour tous
les γ simultanément — la décision optimale pour une métrique de masque sous
l'incertitude du modèle probabiliste. Zéro réentraînement, ciblé path/propagation
IoU.

---

# PARTIE 4 — Statut & réserves

- v1 : queue complète 9 runs (3 seeds × 3 sparsités, 300 epochs) en cours,
  fidèle au papier, résultats de référence.
- v2 : **pilotes seulement** (60 puis 90 epochs, 1 seed, val, dense). Résultat
  très prometteur (bat la persistence oracle en dense, ce que v1 ne fait jamais)
  mais **pas** encore : multi-seed, test set, sparsité 0.5/0.95, budget
  d'information contrôlé vs FNO+. Toute affirmation de supériorité est prématurée
  jusqu'au protocole complet.
- Comparaison honnête FNO+ : v2 (contexte 24) vs FNO+ (contexte 1, autorégressif
  19-pas) n'est **pas** à budget d'information égal — à présenter comme
  « stratégie contexte-long vs contexte-court », pas comme preuve intrinsèque.
