# Master Plan — Papier "Prévision de champs de crue sous observation éparse"

**Document de référence unique du projet jusqu'à la soumission.** Toute nouvelle
tâche, tout changement de scope, toute décision doit être reflétée ici (voir
§10 Gouvernance). Dernière mise à jour : 2026-07-10.

---

## 1. Purpose, thèse et positionnement

**Purpose.** Prévoir l'évolution spatiotemporelle d'une inondation (champ de
profondeur d'eau complet) à partir d'une observation **partielle** de l'état
courant (réseau de capteurs épars) — le scénario réaliste de déploiement.

**Thèse centrale (à démontrer, pas à affirmer).** La diffusion conditionnelle
n'est pas justifiée par une stochasticité physique (le simulateur générateur
est déterministe) mais par l'ambiguïté de reconstruction créée par
l'observation partielle : un modèle déterministe est le bon outil quand l'état
est connu (dense) ; un modèle probabiliste ne se justifie que lorsque
plusieurs états complets sont compatibles avec la même observation éparse.
La preuve exige la comparaison contrôlée déterministe-vs-diffusion à
architecture et contexte appariés, à travers les niveaux de sparsité — ce
qu'aucun travail existant n'a fait (voir §2).

**Contributions visées (dans l'ordre de force) :**
1. Première étude contrôlée "quand le génératif se justifie-t-il" en prévision
   **autorégressive** de champs physiques sous sparsité (jumeau déterministe
   apparié — WP1), sur les crues (créneau vide vérifié — §2).
2. Mécanisme d'échec quantifié : diffuser le champ absolu à résolution
   temporelle fine (signal/champ ≈ 1/400 ici) impose un plancher de bruit
   d'échantillonnage supérieur au signal → sous la persistence triviale
   (mesuré : ×560, 3 seeds). Correction : cible delta avec échelle par régime
   d'observabilité (per-pixel visibility scale). Généalogie complète dans
   `reports/diff_sparse_v2_design.md` ("Incident 2026-07-09").
3. Restauration d'une entrée causale du benchmark : le simulateur officiel
   dérive la friction de Manning du LULC (vérifié dans
   `external_repos/FloodCastBench_official` : `main.py`,
   `hydraulics/saint_venant.py`, README) ; aucun baseline publié ne
   l'exploite (WP5).
4. Protocole d'évaluation rigoureux réutilisable : multi-seed, persistence
   oracle+sparse, masque médian (vote majoritaire) pour les métriques
   binaires, calibration probabiliste mesurée.

**Formulations interdites** (leçons de session, à respecter dans le papier) :
- ~~"DIFF-SPARSE est cassé"~~ → "la paramétrisation champ-absolu suppose
  signal ≈ échelle du champ ; cela peut tenir dans le domaine tidal d'origine
  et échoue à Δt fin".
- ~~"V2 bat FNO+"~~ tant que le confondant de contexte n'est pas traité
  (ctx 24 vs 1) — c'est le jumeau qui porte la comparaison contrôlée.
- ~~"V2 bat V1 de 800x"~~ → le ×780 dense mesure la pathologie de V1, pas le
  gain d'architecture ; le chiffre défendable est le ~×2 en sparse (encore
  confondu par ctx24/12 → WP2).

---

## 2. Paysage littérature (vérifié 2026-07-10, refaire une passe avant soumission)

| Travail | Ce qu'il fait | Ce qui nous en distingue |
|---|---|---|
| [DIFF-SPARSE / AAAI](https://arxiv.org/abs/2505.05381) | Même purpose, inondation côtière (Virginie, marées), sparsité 0/50/95%, "jusqu'à 62% vs existing methods" | Domaine crue pluviale ; jumeau déterministe (absent chez eux) ; mécanisme delta ; calibration |
| [Spatially-Aware Diffusion (arXiv 2409.00230)](https://arxiv.org/pdf/2409.00230) | Reconstruction statique de PDEs sous obs éparses, conditionnement hybride proche de V2 ; conclut "déterministe gagne sans bruit, diffusion avec bruit" | Prévision **autorégressive** (rollout), domaine réel, mécanisme, benchmark public avec baselines |
| Generative DA : [stations météo km-scale (JAMES 2025)](https://arxiv.org/abs/2406.16947), [océan (JAMES 2025)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025MS005063), [PhyDA](https://arxiv.org/html/2505.12882) | Diffusion conditionnée sur obs éparses, météo/océan — champ actif | Aucun contrôle déterministe apparié dans cette ligne ; pas de crues |
| [SF2Bench](https://arxiv.org/html/2506.04281v1) | Jauges réelles Floride, séries temporelles | Pas de champs spatiaux — tâche différente ; complément possible, pas substitut |
| [FloodCastBench (Nature Sci Data 2025)](https://www.nature.com/articles/s41597-025-04725-2) | Notre benchmark ; seul public avec champs + forçages + baselines | — |

**Choix dataset : FloodCastBench, assumé avec 3 limites à écrire** :
(a) vérité terrain = simulation Saint-Venant, une réalisation par événement →
la calibration mesurable est celle de l'ambiguïté de reconstruction ;
(b) sparsité synthétique (masques aléatoires) — mitigation : WP7 masques
structurés ; (c) 4 événements — mitigation : WP8 deuxième événement.

**Risque timing** : le créneau est vide aujourd'hui ; viser soumission rapide
(§9) plutôt que l'exhaustivité.

---

## 3. État acquis (ne pas refaire)

Toutes les valeurs : test split, Australie 60m high-fidelity, protocole natif
12 étapes (1h) sauf mention.

- **FNO+** (3 seeds, protocole officiel t2..t20) : relRMSE 0.006550 ± 0.000135
  (publié : 0.003941 — écart de reproduction 1.66x documenté,
  `reports/fno_plus_multiseed_results.md`). Long-horizon h216 par étape dans
  le dashboard.
- **V1** (9 runs, 3 seeds × 3 sparsités, 300 epochs) : dense ×560 pire que la
  persistence oracle (nrmse 0.041 vs 7.3e-5) ; m50/m95 : bat la persistence
  sparse (~-36% / ~-11% RMSE). Agrégats :
  `reports/diff_sparse_v1_rewrite_full_eval_aggregate*.csv`. Évals h216 dense
  3 seeds au dashboard.
- **V2** (9 runs propres post-correction, commit `2a6de87`) :
  - rollout_val_rmse (val, best) : dense 0.0054/0.0039/0.0078 (seeds
    42/7/123) ; m50 1.331/1.319/**0.745** ; m95 1.864/1.737/**1.189**.
    ⚠ seed123 sparse n'a jamais early-stoppé (300 epochs, encore en
    amélioration) → WP6.
  - Éval test rapide (4/13 fenêtres, stride 48, seeds 42+7) : dense relRMSE
    ~0.001 (bat persistence oracle +25%, premier modèle diffusif du projet à
    le faire) ; m50 ~0.39 (V1 : 0.78→1.06) ; m95 ~0.56 (V1 : 0.86→1.05).
    Comportement plat sur 12 étapes vs dégradation V1 (2 seeds).
  - Design final : delta + échelle par régime (per-pixel visibilité à
    l'étape observée, scalaire delta base=prédiction, loss pushforward
    restreinte aux pixels observés) — 3 pilotes documentés.
- **Incident 2026-07-09** : divergence sparse (défaut d'échelle) + NaN dense
  seed42 — corrigé, testé (24/24 smoke), documenté. Leçon codifiée §8-R3.
- **Infra** : P7 (RTX 6000 Ada 49GB) = machine principale ; Dell (A4000) =
  secondaire ; `experiments/checkpoints/logs` partagés NFS depuis P7 ;
  `data/` locale identique ; code sync par git uniquement (§8-R5).
- **En cours (WP0)** : évals h216 dense V2, 3 seeds (~2h/fenêtre, 3 fenêtres
  test éligibles chacune).

---

## 4. Work packages expérimentaux

Priorité stricte : WP1 > WP2 > WP3 > WP4 ≈ WP6 > WP5 ≈ WP7 > WP8.
WP0 en cours. Chaque WP a des critères de décision AVANT lancement (pré-enregistrement informel).

### WP0 — Long-horizon h216 dense (EN COURS)
- **Runs** : éval h216 (204 étapes) des checkpoints dense V2, 3 seeds,
  test split, stride 48, 8 scénarios. seed42 en cours, seed7+seed123 en
  parallèle derrière.
- **Puis** : copier les 3 dossiers d'éval vers
  `experiments/FloodCastBench/diff_sparse_v2_h216_eval/` (JAMAIS laisser dans
  /tmp — §8-R4), mettre à jour `DIFF_SPARSE_V2_EVAL_DIRS` (3 dirs) dans
  `scripts/build_fno_plus_metric_dashboard.py`, régénérer le dashboard
  (⚠ coordonner avec le Dell : pas de régénération concurrente), commit.
- **Analyse** : forme de la courbe V2 vs V1 vs FNO+ par étape ; le plateau
  plat observé sur 12 étapes tient-il sur 204 ? Où V2 croise-t-il FNO+ ?
- **Sortie papier** : Figure long-horizon (F4).

### WP1 — Jumeau déterministe (LE contrôle existentiel)
- **Design** : réutiliser exactement les encodeurs V2
  (`TemporalContextEncoder` + `SpatialContextEncoder`) et le même backbone
  UNet, en remplaçant la boucle de diffusion par une régression directe :
  un seul forward, pas de canal x_noisy (entrée = features spatiales seules),
  timestep fixe ou embedding retiré, loss MSE sur la même cible delta avec la
  même échelle par régime, mêmes clamps physiques, même pushforward (il
  s'applique aussi au déterministe), même EMA, même budget d'entraînement,
  même sélection de checkpoint (rollout_val_rmse). Paramétrage aussi proche
  que possible de 5.5M params.
- **Fichiers à créer** : `models/deterministic_twin.py`,
  `tools/train_floodcastbench_det_twin.py` (dérivé du trainer V2),
  `tools/evaluate_floodcastbench_det_twin.py` (rollout 1 scénario),
  `configs/floodcastbench_det_twin_highfid_60m*.yaml`,
  `tests/test_det_twin_smoke.py`.
- **Runs** : 3 seeds × 3 sparsités = 9 runs (structure de queue identique à
  V2 ; smoke AVANT queue sur les 3 régimes — §8-R3).
- **Évals** : identiques à V2 (test, 13 fenêtres à terme, stride 48) sans
  métriques probabilistes.
- **Critères de décision (écrits avant de voir les résultats)** :
  - V2 > jumeau en sparse ET jumeau ≥ V2 en dense → thèse confirmée,
    narratif principal.
  - Jumeau ≥ V2 partout → la thèse échoue → pivot honnête : "le déterministe
    suffit même sous sparsité ; la valeur du génératif est ailleurs
    (calibration) ou nulle" — publiable aussi, l'écrire tel quel.
  - Mixte → analyse par régime/métrique, pas de claim général.
- **Sortie papier** : Figure principale (F3), Table T2.

### WP2 — Ablation contexte (tuer le confondant ctx24/ctx12)
- **Runs** : V2 @ `context_length: 12`, seed 42, × 3 sparsités (3 runs).
  Option si résultats serrés : +2 seeds sur m50.
- **Comparaisons** : V2@12 vs V1@12 (gain d'architecture pur) ; V2@12 vs
  V2@24 (valeur du contexte long).
- **Décision** : si V2@12 ≈ V1 → le "×2 sparse" était surtout du contexte →
  le dire et recentrer sur WP1. Si V2@12 ≫ V1 → gain d'architecture réel.
- **Sortie papier** : ligne de T3 (ablations) + phrase de fair-comparison.

### WP3 — Calibration probabiliste (zéro GPU d'entraînement)
- **Analyses** (nouveau `tools/analyze_v2_calibration.py`) :
  - Reliability diagram de P(inondé) = fraction des 8 scénarios mouillés,
    aux seuils γ=0.001 et 0.01 m, par horizon (h1, h6, h12).
  - Spread–skill : std inter-scénarios par pixel vs |erreur de la moyenne|.
  - Rank histogram (position de la cible dans l'ensemble).
  - Couverture des intervalles centraux (50%, 90%).
  - Confirmation chiffrée du choix masque médian vs moyenne (les colonnes
    `_median` existent déjà dans l'évaluateur V2).
- **Prérequis technique** : vérifier ce que l'évaluateur persiste (les
  scénarios par pixel ne sont pas sauvés par défaut) → soit `--save-maps`,
  soit accumulation online des stats de calibration dans l'évaluateur
  (préférable : pas de stockage massif). Refaire les 6 évals test si
  nécessaire (~6h GPU au total).
- **Applicable aussi au jumeau ?** Non (déterministe) — c'est le point : la
  calibration est LA valeur ajoutée que le jumeau ne peut pas offrir. Si la
  calibration est mauvaise, le dire.
- **Sortie papier** : Figure F5, et l'argument central de la discussion.

### WP4 — Grille d'ablation V2 (attribution des gains)
- **Runs** : seed 42, m50 (régime de la thèse) ; dense en plus pour (a).
  - (a) `prediction.target: absolute` (architecture V2, paramétrisation V1)
    → isole la contribution delta. La plus importante de la grille.
  - (b) `include_target_rainfall: false`
  - (c) encodeur spatial désactivé (attention-only) — vérifier s'il existe un
    knob propre, sinon petit ajout modèle (flag `use_spatial_encoder`)
  - (d) `change_weight: 0`
  - (e) `pushforward_fraction: 0`
  - (f) `diffusion.steps: 20` (parité papier V1)
- ~6 runs courts (arrêt anticipé attendu 60-120 epochs), lançables par vagues
  de 3 en parallèle.
- **Sortie papier** : Table T3.

### WP5 — LULC / Manning (V2.1, entrée causale vérifiée)
- **Prérequis** :
  1. Vérifier le schéma de classes LULC contre la source officielle
     (hypothèse ESRI 10-classes NON confirmée — codes observés
     1,2,4,5,7,8,9,10,11,+15 nodata).
  2. Chercher la table LULC→Manning exacte du benchmark dans
     `external_repos/FloodCastBench_official` (constante
     `MANNING_COEFF_FLOODPLAIN = 0.05` trouvée ; `man_path` charge un fichier
     précalculé). Si irrécupérable : table standard Chow 1959 / HEC-RAS,
     approximation déclarée.
- **Implémentation** : canal Manning statique traité comme le DEM
  (broadcast temporel, dans les DEUX encodeurs), normalisation dédiée,
  dataset V2.1 = subclass, config dédiée. Tests smoke.
- **Runs** : seed 42 × 3 sparsités vs V2 de base. Si signal : +2 seeds.
- **Hypothèse pré-enregistrée** : gain surtout sur path-IoU/propagation
  (la rugosité contrôle la vitesse du front), surtout en sparse. Un résultat
  nul est publiable aussi ("le surrogate n'exploite pas une entrée causale du
  simulateur").
- **Sortie papier** : Table T4 + paragraphe.

### WP6 — Budget de convergence sparse (problème seed123)
- **Constat** : seed123 sparse (300 epochs, jamais early-stoppé) finit ~40%
  meilleur que seeds 42/7 (early-stop ~60) → nos chiffres sparse
  sous-estiment la perf et mélangent variance de seed et de convergence.
- **Runs** : relancer seeds 42 et 7 en m50 et m95 avec `epochs: 600`,
  `early_stop_patience: 120` (4 runs). Mettre à jour tous les agrégats.
- **Règle** : toute comparaison finale du papier utilise les checkpoints
  au budget rallongé ; documenter les courbes de convergence en annexe.

### WP7 — Masques structurés réalistes (éval, quasi gratuit)
- **Design** : deux familles de masques d'éval en plus de l'aléatoire i.i.d. :
  (i) capteurs placés le long du réseau de drainage (pixels à forte occupation
  d'eau au temps initial — proxy jauges de rivière) ; (ii) clusters spatiaux
  (couverture régionale inégale). Même budget de capteurs que m50/m95.
- **Implémentation** : générateur de banque de masques dans le dataset
  (les masques d'éval sont déjà une banque pluggable de 10), flag config.
- **Runs** : éval seulement, checkpoints existants (V2, jumeau), m50/m95.
  Question : le modèle entraîné sur masques aléatoires généralise-t-il à des
  sparsités structurées ? (claim "different sensor configurations without
  retraining" du papier DIFF-SPARSE, testé plus durement.)
- **Sortie papier** : Figure/Table F8 + discussion déploiement.

### WP8 — Deuxième événement : UK 2015 (60m high-fidelity)
- **Étapes** : stats de normalisation UK, delta stats UK, vérifier la grille
  (536×536 ? ranges de frames ?), config dédiée ; smoke 3 régimes ; runs
  seed 42 × {dense, m50, m95} pour V2 ET jumeau (6 runs) ; évals test.
- **But** : montrer que la *direction* des conclusions (WP1) tient sur un
  second événement — pas de re-tuning.
- **Sortie papier** : Table T2-bis ou paragraphe généralisation.

### Hors scope explicite (ne pas ouvrir sans décision consignée §10)
- Mamba (V2.2), Pakistan/Mozambique (480m), 30m, cross-event transfer,
  FNO+ à contexte étendu, remask-rollout comme mode principal (reste une
  ligne d'ablation possible), foundation models météo.

---

## 5. Protocole d'évaluation standard (obligatoire pour tout chiffre du papier)

- Split test (frames [2600,2881)), fenêtres complètes (13 au protocole natif
  12 étapes ; 3 au h216/h228), `tile_stride: 48`, patch 64, blending Hann.
- 8 scénarios en test, 2 en val. Graine d'éval fixe, banque de 10 masques.
- Persistence **oracle ET sparse** toutes deux rapportées.
- Métriques continues : relRMSE (déf. Table 4), RMSE/MAE physiques (m), NSE,
  Pearson, biais, NACRPS. Binaires (γ=0.001 et 0.01 m) : CSI, F1,
  precision/recall, path-IoU, propagation-path-IoU — **masque médian
  (vote majoritaire) = métrique principale**, moyenne gardée en
  comparabilité.
- Multi-seed : moyenne ± écart-type, N explicite partout ; tout chiffre
  single-seed est étiqueté comme tel et ne porte aucun claim.
- Parité de contexte obligatoire pour toute comparaison inter-modèles ;
  sinon l'asymétrie est déclarée dans la légende.
- Aucun chiffre issu d'un chemin `/tmp` : toute éval destinée au papier vit
  sous `experiments/FloodCastBench/`.

---

## 6. Plan d'analyse et de comparaison (ce que le papier démontre, table par table)

- **T1** : protocole/benchmark (événements, résolutions, splits, baselines).
- **F2 (mécanisme)** : distribution des deltas par pas vs champ absolu
  (ratio ~400x) ; RMSE persistence vs V1 dense — l'argument du plancher.
- **T2/F3 (résultat principal)** : {persistence oracle, persistence sparse,
  FNO+ (protocole déclaré), V1, jumeau déterministe, V2} × {dense, m50, m95},
  relRMSE + CSI médian + path-IoU, mean±std 3 seeds. La lecture attendue :
  colonne dense → le déterministe gagne ; colonnes sparse → où le génératif
  paie (ou pas).
- **F4** : courbes par étape long-horizon (h216) dense — V2 vs V1 vs FNO+.
- **F5** : calibration (reliability, spread-skill, couverture) — la valeur
  que le jumeau ne peut pas fournir.
- **T3** : ablations V2 (delta, pluie-cible, spatial, change-weight,
  pushforward, steps, ctx12).
- **T4** : LULC/Manning.
- **F6** : qualitatif — figures pipeline réelles existantes
  (`experiments/FloodCastBench/diff_sparse_v*_pipeline_figure_rainy.png`),
  + cartes erreur/scénarios sur une fenêtre de test.
- **F8** : masques structurés vs aléatoires.

---

## 7. Squelette du papier

1. **Introduction** — le problème du déploiement épars ; la question "quand le
   génératif se justifie" ; 3 contributions. (Écrite EN DERNIER.)
2. **Related work** — DIFF-SPARSE ; generative DA (météo/océan) ;
   reconstruction éparse (2409.00230, Voronoi-CNN/Senseiver) ; surrogates de
   crue (FNO+ etc.). Ancrages §2.
3. **Problem setup** — formalisation ; benchmark ; masques ; ce que la
   "calibration" peut signifier avec une GT déterministe (honnêteté).
4. **Methods** — jumeau et V2 présentés comme LA paire contrôlée ;
   paramétrisation delta + échelle par régime (avec le mécanisme en
   motivation) ; détails en annexe.
5. **Experiments** — protocole §5 ; résultats §6 dans l'ordre T2→F4→F5→T3→T4.
6. **Discussion** — réponse à la question titre ; limites (simulation, masques
   synthétiques→WP7, 1-2 événements) ; implications déploiement.
7. **Reproducibility statement** — code, configs, seeds, incident documenté.

**Venue** : cible principale = revue de domaine (Environmental Modelling &
Software / HESS / Water Resources Research — choisir selon longueur et délai
de review au moment du gel) ; plan B = AAAI track applicatif ; test rapide
possible = workshop NeurIPS/ICLR climat avec la version courte (WP1+WP2+WP3
suffisent pour la version workshop).

---

## 8. Règles méthodologiques permanentes (codifiées, non négociables)

- **R1** : tout claim = multi-seed ou étiqueté single-seed sans généralisation.
- **R2** : parité de contexte et de protocole pour toute comparaison ; toute
  asymétrie restante déclarée à côté du chiffre.
- **R3** : smoke-tester CHAQUE régime (chaque sparsité, chaque mode) qu'un
  protocole va exécuter, avant de lancer la queue (leçon incident 2026-07-09).
- **R4** : aucun artefact destiné au papier dans `/tmp` ou un scratchpad de
  session ; destination = `experiments/` (NFS, survit aux migrations).
- **R5** : le code se synchronise par git uniquement — commit+push après
  validation (tests verts), pull avant toute session ; jamais de `git add -A`
  aveugle ; le NFS ne couvre QUE experiments/checkpoints/logs.
- **R6** : critères de décision écrits AVANT de lancer une expérience
  (pré-enregistrement informel, comme WP1).
- **R7** : `kill -TERM`, jamais `kill -STOP`, pour libérer le GPU.
- **R8** : les résultats négatifs sont des résultats — ils vont dans le
  rapport et, si pertinents, dans le papier.

---

## 9. Séquencement et jalons

| Jalon | Contenu | Critère de passage |
|---|---|---|
| M0 (fait) | Queue V2 9/9 propre + comparaison courte V1/V2/FNO+ | ✅ 2026-07-10 |
| M1 | WP0 fini : dashboard 4 courbes (FNO+, V1, V2, + Table4 ref) | dashboard régénéré + commité |
| M2 | WP1 : jumeau entraîné 9 runs + évalué | tableau T2 rempli, décision de narratif prise |
| M3 | WP2 + WP3 + WP6 | confondant tué, calibration mesurée, budgets corrigés |
| M4 | WP4 + WP5 + WP7 | tables T3/T4/F8 remplies |
| M5 | Gel des expériences Australie ; WP8 en parallèle de la rédaction | plus aucun run "pour voir" |
| M6 | Draft complet | relecture critique interne (jouer le reviewer hostile) |
| M7 | Soumission | — |

Ordre de rédaction : Methods+Experiments dès M3 (les protocoles sont figés),
Discussion après M4, Intro/Related en dernier. La version workshop peut
partir dès M3 si une deadline se présente.

---

## 10. Gouvernance du document

- Ce fichier est **la** source de vérité du plan. Il est versionné git ;
  toute modification passe par un commit dont le message référence la
  section touchée.
- **Ajouter une tâche/idée** : l'inscrire dans le WP pertinent ou dans
  "Hors scope" avec une ligne de justification + date dans le changelog
  ci-dessous. Une idée non inscrite ici n'existe pas.
- **Résultat obtenu** : mettre à jour §3 (état acquis) et cocher le WP ;
  les chiffres vont dans les CSV/rapports dédiés, pas en vrac ici.
- Les deux machines (P7, Dell) travaillent depuis ce même document via git.

### Changelog
- 2026-07-10 — création (P7). État : WP0 en cours, WP1-WP8 définis,
  littérature vérifiée, règles R1-R8 codifiées.
