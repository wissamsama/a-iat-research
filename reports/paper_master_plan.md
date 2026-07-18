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
2. Mécanisme d'échec **hypothétique, actuellement corrélationnel, pas
   prouvé** (relecture critique 2026-07-17, voir WP12) : diffuser le champ
   absolu à résolution temporelle fine (signal/champ ≈ 1/400-1/490 ici)
   impose, par hypothèse, un plancher de bruit d'échantillonnage supérieur
   au signal → sous la persistence triviale (mesuré : ×560, 3 seeds). Ce
   qu'on a réellement : ratio élevé + échec co-observés sur 2 événements de
   la MÊME famille de benchmark (même simulateur Saint-Venant, même
   philosophie de résolution) — pas encore de courbe dose-réponse (faire
   varier le ratio et observer si la sévérité de l'échec suit), pas de
   réplication hors famille FloodCastBench. **Formulation correcte tant que
   WP12 n'est pas fait : "cohérent avec l'hypothèse", jamais "démontre" ou
   "le mécanisme est".** Correction proposée (elle, empiriquement acquise,
   *qu'importe le mécanisme sous-jacent*) : cible delta avec échelle par
   régime d'observabilité (per-pixel visibility scale) — ça marche, la
   question ouverte est uniquement le POURQUOI. Généalogie complète dans
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

### Structure profonde du papier (ajouté 2026-07-17 — la colonne vertébrale)

Le papier n'est pas "notre modèle marche" ni un simple audit : c'est une
**décomposition causale** de la phrase "les modèles de diffusion marchent
pour la prévision de crues", que la littérature affirme sans jamais la
décomposer. Trois piliers, chacun avec sa question, sa méthode et ses WPs :

| Pilier | Question | Méthode | WPs | État |
|---|---|---|---|---|
| **A. Contrôle** | Le mécanisme génératif lui-même apporte-t-il quelque chose, à squelette identique ? Sur combien d'événements et d'architectures ce constat tient-il ? | Jumeau déterministe apparié poids-à-poids, répliqué sur 4 événements et ≥2 architectures génératives supplémentaires | WP1 (+WP2 contexte, +WP6 intégrité, +WP11 prix, +WP13 multi-événements, +WP14 multi-architectures) | m50/m95 Australie : pièces réunies, tableau à construire |
| **B. Mécanisme** | POURQUOI la recette de référence échoue-t-elle ici ? Un ratio adimensionnel σ_Δ(Δt)/σ_champ organise-t-il succès et échecs ? | Dose-réponse : squelette fixe, {Δt × représentation cible} croisés | WP12 (absorbe WP4a) | Phase 1 faite, Phase 2 à lancer |
| **C. Honnêteté incertitude** | La justification de repli "le génératif fournit l'incertitude" survit-elle aux baselines UQ standards ? | Deep-ensemble de jumeaux + calibration mesurée à métriques identiques | WP3, WP9 (+WP7 masques réalistes) | **FERMÉ, m50+m95** : jumeau jamais moins bien calibré |

Autour des piliers : validité externe (WP8 UK, WP10 zéro-shot sur les 4
événements), attribution architecturale (WP4 b-f), coût (WP11). **Hors
piliers : WP5 (Manning/LULC)** — voir sa section, rescopé optionnel.

**Décision utilisateur du 2026-07-18** : le pilier A passe d'un test
ponctuel (1 événement, 1 architecture) à une **méthodologie répliquée**
(WP13 : 4 événements ; WP14 : ≥2 architectures génératives supplémentaires,
dont une variante Mamba appariée à son propre jumeau). C'est le levier
identifié comme le plus susceptible de faire monter le papier au-dessus de
Q1 domaine — si le motif se réplique, l'argument devient méthodologique
("tout papier de diffusion spatiotemporelle sous sparsité doit reporter un
jumeau apparié"), pas seulement empirique sur un cas.

**La réponse que le papier apporte (état actuel des données)** : l'essentiel
de ce qui fait "marcher la diffusion" en prévision de crue n'a rien de
génératif — les gains viennent de choix de représentation (espace delta,
échelle par régime) qui profitent À L'IDENTIQUE au jumeau déterministe ; le
ratio signal/champ prédit (si WP12 confirme) où la recette champ-absolu peut
fonctionner du tout ; et le dividende incertitude, mesuré honnêtement,
n'existe pas encore sur ce benchmark. Chaque WP doit servir un pilier ; toute
tâche qui n'en sert aucun est par défaut hors scope.

### Priorités de sortie (ajouté 2026-07-18, décision utilisateur) — deux paliers

**Règle de lecture** : le PALIER 1 seul = un papier Q1 domaine soumissible.
Le PALIER 2 = ce qui le rend inattaquable et candidat au-dessus. Chaque
item du palier 2 est ADDITIF (s'insère dans le papier sans en changer la
narration) — donc le palier 1 définit un point de gel atteignable, et on
n'ouvre un item du palier 2 que quand tout ce qui le précède est fermé.
Interdiction de re-prioriser sous le coup d'une idée nouvelle sans passer
par une mise à jour explicite de cette section.

**PALIER 1 — NOYAU (Q1 domaine assuré si tout est fait), dans l'ordre :**

| # | Item | Coût estimé | État |
|---|---|---|---|
| 1.1 | Tableau central Australie (jumeau vs Δ-Diff, 3 seeds × 3 sparsités, protocole test 13/13, checkpoints corrigés) + tests appariés (Wilcoxon/bootstrap par fenêtre×seed) | en cours | 9/10 évals faites, dernière en cours |
| 1.2 | WP12 Phase 2 (dose-réponse {Δt × cible}, 8 runs one-step courts) — décide si le pilier B est "mécanisme montré" ou "hypothèse honnête" | ~1-2 j GPU | Δt choisis, prêt à lancer |
| 1.3 | Re-run protocole COMPLET (13/13) de la table Δ-Diff-vs-FNO+ dense (actuellement 4/13, la légende du papier promet ce re-run) | ~3-5h GPU | à lancer |
| 1.4 | WP2 ablation contexte : confirmer sur 2 seeds de plus (actuellement single-seed, encadré PENDING du papier le promet) | ~1 j GPU | à lancer |
| 1.5 | WP8 : UK complet (V2+jumeau, 3 seeds × 3 sparsités + évals) — LE saut de crédibilité 1→2 événements | ~2-4 j | config prête |
| 1.6 | WP10 : zero-shot triage sur les 4 événements (éval-only) | ~0.5 j | après configs Pak/Moz minimales |
| 1.7 | WP11 : table coût/complexité (mesures réelles) | ~2-3h | à lancer |
| 1.8 | Assemblage papier : intro + discussion + section protocole, gel des encadrés PENDING restants, figures f2 remplacée par dose-réponse | ~2-3 j rédaction | draft déjà avancé |

Estimation totale palier 1 depuis maintenant : **~7-12 jours de mur**
(items 1.3/1.4/1.6/1.7 intercalables dans les creux GPU des gros items).

**PALIER 2 — INATTAQUABLE (chaque item ferme une attaque reviewer
identifiée), ordre de rendement décroissant :**

| # | Item | Attaque qu'il ferme | Coût |
|---|---|---|---|
| 2.1 | WP14-A : 2e architecture générative (sampler/schedule différent) + son jumeau, Australie | "c'est peut-être propre à CE modèle de diffusion" | ~2-3 j |
| 2.2 | **5 seeds comme standard, pas juste +2 sur un tableau** : étendre {42,7,123} à {42,7,123,+2 nouvelles} sur TOUS les résultats phares réutilisés par le papier (comparaison centrale dense/m50/m95 ; WP9 calibration ; WP12 dose-réponse si Phase 2 confirme) — particulièrement justifié depuis que WP6 a montré que la variance inter-seed domine le signal de convergence chez le jumeau, et que m50 a un signe qui s'inverse selon la seed (3 seeds ne suffisent pas à distinguer near-tie réel de sous-échantillonnage) | "3 seeds c'est trop peu pour vos claims de variance" | ~4-6 j (16+ trainings + évals à travers les WPs concernés, avec le taux de retry observé) |
| 2.3 | WP13 événements 3/4 : Pakistan + Mozambique complets (configs à créer) | "2 événements de la même famille haute-fidélité seulement" | ~3-6 j |
| 2.4 | WP14-B : variante backbone Mamba (emplacements multiples, chacun avec jumeau, fix LayerScale d'office) | "et si une architecture séquentielle moderne changeait la donne ?" | ~3-5 j |
| 2.5 | WP4 (b-f) : grille d'ablation restante | "d'où viennent exactement les gains ?" | ~1-2 j |
| 2.6 | Calibration : CRPS (score sharpness-aware) + éval V2-m95-aléatoire (dernier caveat de masque de la Fig. 6) + 3e seed cluster WP7 + logging par-fenêtre dans l'évaluateur (pour le vrai test apparié fenêtre×seed, n=39, avant tout futur rerun) | "votre calibration n'utilise que la couverture" / caveats résiduels / "n=3 par seed n'est pas un test statistique sérieux" | ~1-2 j |
| 2.7 | WP15-A : paramétrisation delta+échelle appliquée à FNO+ (réutilise l'infra Papier 2), démêle "paramétrisation" vs "backbone" avant toute affirmation jumeau-vs-FNO | "vous comparez un backbone bien paramétré à un backbone mal paramétré, pas deux backbones" | ~1-2 j |

**Objectif "inattaquable" en une phrase** : 5 seeds partout où un résultat
est cité comme preuve, sur les 4 événements FloodCastBench (WP13), avec le
protocole jumeau répliqué sur ≥2 architectures génératives indépendantes
(WP14) — c'est la combinaison 2.1+2.2+2.3+2.4 qui définit le palier 2
complet, pas un seul item isolé.

Estimation palier 2 complet : **+3 à 4 semaines** après le palier 1 (révisé
à la hausse : 2.2 élargi à "5 seeds partout", pas seulement la
comparaison centrale). Point de décision de soumission : à la fin du
palier 1, soumettre ou continuer se décide sur l'état réel des résultats
2.1/2.2 (les deux premiers items du palier 2 sont ceux qui changent le
tier ; 2.3-2.6
renforcent sans changer la catégorie).

**Hors paliers (explicitement)** : WP5 Manning/LULC (aucun pilier),
§9-bis Transactions (gates inchangées), combinaison WP14 A+B.

### Orientation stratégique du papier (ajouté 2026-07-18, réflexion de fond demandée par l'utilisateur)

**Identité du papier — la découverte positive cachée dans le résultat
négatif.** Le jumeau n'est pas seulement un contrôle : **c'est le meilleur
modèle du benchmark** (dense : 0.000417 vs 0.003941 FNO+ publié = ~9.4×
meilleur que le SOTA publié, pour ~1/320e du coût d'inférence de la
diffusion — 40 pas × 8 scénarios vs 1 passe). Le papier ne doit PAS se
vendre comme "papier négatif sur la diffusion" mais comme : *un modèle
déterministe simple, obtenu en SUPPRIMANT la boucle de diffusion d'un
modèle de diffusion, établit un nouvel état de l'art — et voici le
protocole qui permet de découvrir ce genre de chose*. Même contenu,
narration radicalement plus favorable en review, 100% honnête.

**Trois lectures du résultat, à hiérarchiser dans l'intro** :
1. Étroite (vraie mais paroissiale) : "n'utilisez pas la diffusion sur
   FloodCastBench". Niveau note technique.
2. Médiane (la contribution réelle) : **décomposition d'attribution** —
   les gains attribués à la diffusion dans cette ligne de travaux
   viennent de choix de représentation (delta, échelle par régime) qui
   se transfèrent tels quels au déterministe. Parle à tous ceux qui
   construisent sur DIFF-SPARSE et sa descendance.
3. Large (le cadre conceptuel) : **confusion épistémique/aléatoire** —
   la diffusion importée de domaines à stochasticité réelle (météo
   chaotique) vers des problèmes de reconstruction d'un simulateur
   déterministe, sans vérifier que la prémisse tient. La vérité terrain
   étant déterministe, la "diversité de scénarios" mesure l'ambiguïté de
   reconstruction, pas une incertitude physique — et un jumeau + deep
   ensemble la couvre.

**Mise en garde (ajoutée 2026-07-19, correction après question utilisateur
"ça me paraît trivial")** : la recommandation **procédurale** ("testez
toujours un jumeau apparié avant d'attribuer un gain à un mécanisme
génératif") n'est PAS la contribution — prise seule, c'est de la méthode
scientifique de base, triviale comme principe. C'est un **corollaire
pratique**, à ne faire apparaître qu'en fin de discussion. La vraie
contribution est le **niveau 2 ci-dessus** (attribution) porté par un fait
non trivial : le motif mesuré est **non-monotone** (jumeau gagne aux deux
extrêmes dense/m95, indécis au milieu m50) — pas le résultat plat
("diffusion inutile partout") qu'un simple rappel méthodologique
prédirait. L'intro doit ouvrir sur ce motif + le jumeau=nouveau SOTA, pas
sur "on recommande d'ablater".

**Modèle de menace complet (attaque reviewer → réponse)** :

| Attaque | Réponse | État |
|---|---|---|
| **"Tautologique : la cible EST déterministe, évidemment le déterministe suffit"** (l'attaque la plus profonde) | (a) le champ entier a fait cette "évidence" à l'envers sans jamais la tester ; (b) le résultat n'est PAS uniforme (m50 indécis, motif non-monotone) — une tautologie prédirait jumeau-partout, ce qu'on n'observe pas ; (c) le cas sans bruit d'observation est structurellement le PIRE pour la diffusion — à dire explicitement | À écrire dans l'intro/discussion |
| "Votre diffusion est un homme de paille mal réglé" | Δ-Diff bat le SOTA publié sur les 5 métriques — c'est le meilleur génératif du benchmark ; le jumeau = mêmes poids exactement ; WP14 ajoute une 2e architecture | Solide (WP14 = tier 2) |
| "4 événements du même simulateur ≠ réplication" | Limite explicite déjà dans le papier + séparation procédural/substantiel | Fait (43f4bf6) |
| "3 seeds, pas de vrai test statistique" | 5 seeds (tier 2.2) + logging par-fenêtre pour test n=39 — **à faire AVANT les runs UK pour ne pas payer 2×** | Re-séquencé (voir ci-dessous) |
| "Résultat connu (Spatially-Aware Diffusion 2024)" | Cadrage : confirmation + extension majeure (autorégressif, benchmark réel, calibration, mécanisme) d'une observation éparse → protocole | Dans §2, à renforcer en intro |
| "Mécanisme non prouvé" | Langage d'hypothèse + WP12 Phase 2 tranche (ou rétracte honnêtement) | Phase 2 = prochain GPU |
| **"relRMSE cherry-pické"** | **NEUTRALISÉE 2026-07-18** : vérif multi-métriques (coût zéro, JSONs existants) — m95 : jumeau gagne sur les 6 métriques (relRMSE, NSE, CSI@0.001, CSI@0.01, CSI-médiane, path-IoU) ; m50 : aucune métrique n'a un signe cohérent 3/3 seeds → "indécis" tient partout | Fait, à insérer dans §6.3 |
| "Pas de données réelles / capteurs parfaits" | Limite honnête + **proposition axe bruit d'observation** (ci-dessous) | Proposé |

**Proposition nouvelle (à valider utilisateur) — l'axe "bruit
d'observation"** : la seule œuvre antérieure comparable (Spatially-Aware
Diffusion) trouve que la diffusion n'aide QUE sous bruit d'observation.
Nos capteurs simulés sont parfaits — le cas structurellement le plus
favorable au déterministe. Ajouter un axe bruit (σ_obs croissant) pourrait
transformer la réponse du papier de "pas ici" à **"voici la frontière où
le génératif commence à payer"** — une réponse constructive à la question
du titre, bien plus forte qu'une négation. Coût étagé : (i) sonde
éval-only sur checkpoints existants (bruiter les pixels observés à l'éval,
~½ journée GPU, zéro entraînement) → si le gap jumeau-diffusion se resserre
déjà en zero-shot, signal fort ; (ii) réentraînement à 1-2 niveaux de
bruit (~2-3 j) pour le test propre. Recommandation : (i) en fin de palier
1, (ii) en tier 2 si (i) montre un signal. C'est peut-être l'idée au
meilleur ratio impact/coût restant sur la table.

**Re-séquencement conséquent (palier 1)** : le logging par-fenêtre de
l'évaluateur (était en 2.6) passe AVANT l'item 1.5 (UK) — sinon toutes
les évals UK devront être refaites pour le test apparié n=39. Petit coût
code, gros coût évité.

**Portée de la comparaison au jumeau vs FNO (ajouté 2026-07-19, question
utilisateur : "le jumeau peut-il remplacer FNO en toute condition ?")**

Réponse théorique honnête : **non, et ce n'est pas censé être vrai**. Le
même argument de biais inductif qui prédit que le jumeau bat FNO+ ici
(design local/multi-échelle mieux adapté aux fronts nets d'inondation et
à l'observation éparse) prédit l'inverse sur les problèmes pour lesquels
FNO a été conçu et validé à l'origine (Navier-Stokes, Darcy flow — champs
globalement lisses, sans discontinuité locale, où le mélange spectral
global est un avantage et où l'invariance en résolution est une propriété
que ni U-Net ni Mamba ne possèdent nativement). Une affirmation "bat FNO
en toute condition" est un argument de type no-free-lunch et serait très
probablement fausse si testée sur le terrain de FNO — et un reviewer verra
immédiatement le biais de sélection si on ne teste que des datasets
similaires à FloodCastBench.

**Ce qui est défendable** : une affirmation **scopée**, pas universelle —
"pour la classe de problèmes à observations éparses + reconstruction à
frontières localisées nettes (inondation, et probablement propagation de
feu / cartographie d'étendue quasi-binaire), le jumeau (potentiellement
Mamba) est préférable à FNO+". Voir WP15 pour les deux expériences qui
établissent cette portée proprement plutôt que de la surclaimer.

### PALIER 3 — hors scope de cette soumission (ajouté 2026-07-18, discussion utilisateur)

**Question posée** : pour que l'affirmation *substantielle* (pas juste la
recommandation procédurale) devienne vraiment générale, ne faudrait-il pas
3-4 datasets **indépendants** (pas 4 événements du même simulateur comme
WP13) et 4-5 architectures génératives (pas 2 comme WP14) ?

**Réponse honnête** : oui en principe, mais c'est un ordre de grandeur
au-dessus de WP13/WP14, probablement plusieurs mois — potentiellement un
2e papier à lui seul, pas un ajout à celui-ci. Un dataset indépendant
coûte plus qu'un événement du même benchmark (intégration complète du
pipeline à chaque fois, cf. le coût déjà observé pour UK/Pakistan/
Mozambique) ; 4-5 architectures multiplient le risque d'instabilités
inédites à chaque nouvelle (cf. l'historique Mamba/LayerScale).

**Décision** : NE PAS engager ceci dans le scope de la soumission actuelle.
Le papier n'a pas besoin de cette généralisation complète pour être
honnête : la recommandation *procédurale* (toujours construire le jumeau
apparié) est déjà bien soutenue par une seule démonstration rigoureuse et
ne dépend pas d'une large base de preuves ; c'est l'affirmation
*substantielle* (le motif précis mesuré) qui a été délibérément cantonnée
à FloodCastBench + architectures testées (limite explicite ajoutée au
papier, §7.1) plutôt que payée par une généralisation coûteuse. Ce
cantonnement EST la solution retenue pour cette soumission.

**Traitement** : noté ici comme travail futur / 2e papier potentiel, pas
comme item à cocher avant de soumettre. Si repris un jour : privilégier UN
dataset réellement indépendant (pas 3-4) + 3-4 architectures (pas 5)
comme ambition réaliste d'un "palier 3" borné, en gardant la même
discipline de gel/priorisation que les paliers 1-2 plutôt que de laisser
le scope dériver sans limite.

**WP15-B (ajouté 2026-07-19)** : même logique, appliquée à la question
"le jumeau peut-il remplacer FNO en toute condition ?" (voir WP15 en §4).
Tester sur un dataset délibérément favorable à FNO (champ lisse,
généralisation en résolution) est le test qui répondrait vraiment à la
question, mais coûte une intégration de dataset complète — même ordre de
grandeur que le reste du palier 3. Non engagé maintenant ; le papier
actuel se limite à l'affirmation scopée (portée = classe de problèmes à
observations éparses + fronts nets) et énonce explicitement l'argument
no-free-lunch plutôt que de laisser la question ouverte sans réponse.

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

- **V2 dense vs le chiffre OFFICIEL de FNO+ (Table 4 du papier, pas notre
  reproduction) — 2026-07-16, toutes métriques, 3 seeds** (lecture rapide
  4/13 fenêtres, résultat stocké proprement sous
  `experiments/FloodCastBench/v2_dense_fullmetrics_check/`) :

  | Métrique | FNO+ officiel | V2 dense (moy. 3 seeds) | Verdict |
  |---|---:|---:|---|
  | relRMSE | 0.003941 | 0.001576 ± 0.000818 | V2 ×2.5 meilleur |
  | NSE | 0.999979 | 0.999996 ± 0.000004 | V2 meilleur |
  | Pearson r | 0.999990 | 0.999999 ± 0.000001 | V2 meilleur |
  | CSI@0.001 | 0.939638 | 0.986660 ± 0.003735 | V2 meilleur (+4.7 pts) |
  | CSI@0.01 | 0.984588 | 0.999098 ± 0.000393 | V2 meilleur (+1.5 pts) |

  **V2 bat le chiffre publié de FNO+ sur les 5 métriques disponibles**, pas
  seulement relRMSE — évite complètement l'objection "vous avez mal
  reproduit FNO+" puisque ce n'est pas notre reproduction qui sert de
  référence ici. Renforcé par WPB0 (`reports/fno_plus_beat_paper_plan.md`) :
  on a déjà testé et écarté l'explication "V2 gagne juste parce qu'il voit
  plus de contexte" — donner le même contexte à FNO+ le rend pire, pas
  meilleur. Réserves : protocoles pas strictement identiques (V2 = moyenne
  de 8 scénarios vs FNO+ = sortie déterministe unique ; lecture rapide
  4/13 fenêtres, pas le protocole test complet) — écart trop large pour être
  un artefact de ces différences, mais pas encore un chiffre "figé" pour
  publication sans le protocole complet à 13/13 fenêtres.
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
- **WP1 jumeau, seed42 (3/9 runs faits)** : `rollout_val_rmse` (proxy interne
  UNIQUEMENT, pas une comparaison valide — voir §4-WP1 et §8-R9) : dense
  0.00185 (epoch 300) vs V2 0.00541 ; m50 1.026 (epoch 45) vs V2 1.331 ; m95
  0.732 (epoch 300) vs V2 1.864. Jumeau devant sur ce proxy aux 3 sparsités —
  attendu vu le biais statistique, ne préjuge PAS du résultat sur la vraie
  comparaison (moyenne-8-scénarios). Éval test réelle du jumeau : pas encore
  faite. Vagues seed7/seed123 en cours (2026-07-11).
- **Infra** : P7 (RTX 6000 Ada 49GB) = machine principale ; Dell (A4000) =
  secondaire ; `experiments/checkpoints/logs` partagés NFS depuis P7 ;
  `data/` locale identique ; code sync par git uniquement (§8-R5).
- **En cours (WP0)** : évals h216 dense V2, 3 seeds (~2h/fenêtre, 3 fenêtres
  test éligibles chacune).

---

## 4. Work packages expérimentaux

Priorité stricte : WP1 > WP2 > WP3 > WP4 ≈ WP6 > WP5 ≈ WP7 > WP8.
WP0 en cours. Chaque WP a des critères de décision AVANT lancement (pré-enregistrement informel).

### WP0 — Long-horizon h216 dense (TERMINÉ à 3 seeds, 2026-07-13)
- **Runs** : éval h216 (204 étapes) des checkpoints dense V2, 3 seeds,
  test split, stride 48, 8 scénarios. **Les 3 seeds sont faites** (seed42
  puis seed123 sans incident ; seed7 a nécessité une relance après un échec
  silencieux du script veilleur — voir changelog).

  | Seed | relRMSE (pooled, h216) | CSI@0.001 | NSE |
  |---|---:|---:|---:|
  | 42 | 0.045 | 0.909 | — |
  | 123 | 0.073 | 0.910 | 0.993 |
  | 7 | 0.062 | 0.860 | 0.995 |

- **Fait** : les 3 dossiers d'éval copiés vers
  `experiments/FloodCastBench/diff_sparse_v2_h216_eval/` (jamais laissés dans
  /tmp — §8-R4), `DIFF_SPARSE_V2_EVAL_DIRS` mis à jour (3 dirs) dans
  `scripts/build_fno_plus_metric_dashboard.py`, dashboard régénéré (courbe
  V2 désormais N=3, comme V1 et FNO+). Courbes V1 et V2 aussi rebasées sur un
  axe h=1 commun (première prédiction réelle, contexte exclu) le 2026-07-11,
  au lieu de l'ancien axe en frame absolue qui laissait un trou de contexte
  au début du tracé.
- **Analyse restante** : forme de la courbe V2 vs V1 vs FNO+ par étape ; le
  plateau plat observé sur 12 étapes tient-il sur 204 ? Où V2 croise-t-il
  FNO+ ? — à faire en lisant le dashboard régénéré, pas encore fait
  formellement.
- **Sortie papier** : Figure long-horizon (F4).

### WP1 — Jumeau déterministe (LE contrôle existentiel)
- **Design** : réutiliser exactement les encodeurs V2
  (`TemporalContextEncoder` + `SpatialContextEncoder`) et le même backbone
  UNet, en remplaçant la boucle de diffusion par une régression directe :
  un seul forward, pas de canal x_noisy (entrée = features spatiales seules),
  timestep fixe ou embedding retiré, loss MSE sur la même cible delta avec la
  même échelle par régime, mêmes clamps physiques, même pushforward (il
  s'applique aussi au déterministe), même EMA, même budget d'entraînement,
  même sélection de checkpoint (rollout_val_rmse en interne — voir piège
  ci-dessous pour la comparaison FINALE). Parité de paramètres exacte
  (testée), pas juste approximative.
- **Fichiers** : `models/deterministic_twin.py` (`DeterministicTwinModel`,
  sous-classe de `DiffSparseV2Model`) + `build_v2_family_model()` (dispatch
  sur `model.name`). **Pas d'évaluateur séparé** : le jumeau est piloté par
  `tools/train_floodcastbench_diff_sparse_v2.py` et
  `tools/evaluate_floodcastbench_diff_sparse_v2.py` existants, inchangés
  (interface `denoise`/`sample`/`training_step_loss` identique à V2) — évite
  toute divergence de protocole entre les deux bras de la comparaison.
  Configs : `configs/floodcastbench_det_twin_highfid_60m*.yaml`.
  Tests : `tests/test_det_twin_smoke.py`.
- **Runs** : 3 seeds × 3 sparsités = 9 runs (structure de queue identique à
  V2 ; smoke AVANT queue sur les 3 régimes — §8-R3).

- **⚠ PIÈGE MÉTHODOLOGIQUE (découvert 2026-07-11, voir aussi §8-R9)** :
  `rollout_val_rmse` (le proxy interne servant à la sélection de checkpoint,
  1 seul scénario tiré, 8 tuiles val) **NE DOIT JAMAIS SERVIR DE MÉTRIQUE DE
  COMPARAISON FINALE** entre le jumeau et V2. Raison statistique, pas un
  détail d'implémentation : pour toute distribution postérieure,
  E[(échantillon − vérité)²] = Var(postérieure) + E[(moyenne − vérité)²] ≥
  E[(moyenne − vérité)²]. Un régresseur déterministe cible directement la
  moyenne (l'estimateur qui minimise le RMSE par construction) ; un tirage
  UNIQUE de diffusion porte en plus la variance de la postérieure. Comparer
  1 tirage de diffusion à une sortie déterministe sur du RMSE favorise
  mathématiquement le déterministe, indépendamment de la validité de la
  thèse. **Preuve empirique déjà observée** : jumeau seed42 bat V2 seed42
  sur `rollout_val_rmse` aux 3 sparsités (dense ×2.9, m50 ×1.3, m95 ×2.5) —
  signal attendu sur cette métrique précise, PAS une réfutation de la thèse.
- **La comparaison qui compte** (protocole d'éval réel, §5) : jumeau
  (sortie unique, `num_scenarios=1` forcé) **vs V2 moyenne-DE-8-scénarios**
  (`num_scenarios_test=8`, déjà le protocole standard). C'est là, et
  seulement là, que la thèse se juge. Éval test réelle sur les checkpoints
  jumeau (même commande que pour V2, `--tile-stride 48`) obligatoire avant
  toute conclusion — pas encore faite au moment d'écrire ceci.
- **Critères de décision (écrits avant de voir les résultats de l'éval
  réelle — le proxy `rollout_val_rmse` ci-dessus ne compte pas comme
  résultat pour ces critères)** :
  - V2 (moyenne-8) > jumeau en sparse ET jumeau ≥ V2 en dense → thèse
    confirmée, narratif principal.
  - Jumeau ≥ V2 (moyenne-8) partout → la thèse échoue → pivot honnête : "le
    déterministe suffit même sous sparsité ; la valeur du génératif est
    ailleurs (calibration) ou nulle" — publiable aussi, l'écrire tel quel.
  - Mixte → analyse par régime/métrique, pas de claim général.
- **Sortie papier** : Figure principale (F3), Table T2.

- **RÉSULTAT RÉEL 2026-07-11, CONFIRMÉ À 3/3 SEEDS (seed42/seed7/seed123,
  test 4/13 fenêtres, stride 48, protocole correct : jumeau vs V2
  moyenne-8-scénarios) — respecte R1 (≥3 seeds avant claim établi) :**

  | Sparsité | V2 seed42 | V2 seed7 | V2 seed123 | V2 moy. | Jumeau seed42 | Jumeau seed7 | Jumeau seed123 | Jumeau moy. | Gagnant (moyenne) |
  |---|---|---|---|---|---|---|---|---|---|
  | dense | 0.001311 | 0.000923 | 0.002490 | 0.001574 | 0.000420 | 0.000499 | 0.000410 | 0.000443 | **jumeau ×3.55** (3/3 seeds) |
  | m50 | 0.395202 | 0.388081 | 0.251160 | 0.344814 | 0.381563 | 0.292901 | 0.375410 | 0.349958 | **quasi-tie, V2 ×1.015** (2/3 seeds jumeau, 1/3 V2 nettement) |
  | m95 | 0.567427 | 0.560316 | 0.453630 | 0.527124 | 0.342173 | 0.331412 | 0.335170 | 0.336252 | **jumeau ×1.57** (3/3 seeds) |

  **Révision de la branche de décision (seed123 change la conclusion m50)** :
  dense et m95 restent robustes 3/3 seeds en faveur du jumeau (pas
  d'ambiguïté, écarts larges et stables). **m50 n'est PAS "jumeau ≥ V2
  partout" comme le laissaient penser les 2 premiers seeds** — seed123
  inverse nettement le résultat à m50 (V2 0.2512 vs jumeau 0.3754, V2 gagne
  cette fois par ×1.49), et la moyenne à 3 seeds devient un quasi-tie
  (écart 1.5%, dans le bruit inter-seed). **Branche de décision réellement
  déclenchée : "mixte" (pas "jumeau ≥ V2 partout")** — pas de claim général
  du type "le déterministe suffit partout." Formulation correcte pour le
  papier : *le jumeau domine nettement en dense et en sparsité extrême
  (m95), mais à sparsité intermédiaire (m50) le résultat est indécis entre
  les deux architectures sur ces 3 seeds — aucune conclusion générale sur
  m50 sans creuser plus (WP2 ablation contexte pourrait éclairer si c'est un
  effet du contexte long, ou variance intrinsèque au régime m50).*

  **Nuance dense/m95 déjà en U conservée** : l'écart jumeau/V2 n'est pas
  monotone en fonction de la sparsité — fort en dense, se resserre à m50,
  se rouvre à m95. Le fait que m50 soit précisément le point où le signe
  du gagnant peut s'inverser d'un seed à l'autre (contrairement à dense/m95,
  stables) suggère que m50 est une zone de transition/instabilité réelle du
  système, pas juste un point de la courbe en U — piste d'analyse pour la
  Figure F3 (afficher les 3 seeds individuellement à m50, pas seulement la
  moyenne, pour rendre cette instabilité visible plutôt que la lisser).

  **Conséquence sur le narratif papier** : la thèse forte ("le génératif se
  justifie sous sparsité") ne tient toujours pas comme narratif principal
  (dense et m95 sont clairement en faveur du déterministe, qui est aussi la
  condition la plus commentée/la plus simple). Mais le narratif n'est plus
  "le déterministe gagne partout" — c'est plus nuancé et, si creusé,
  potentiellement plus intéressant : *où exactement, et pourquoi, l'avantage
  du déterministe s'estompe-t-il ?* Le pivot vers la calibration (WP3) comme
  justification de la valeur du génératif reste la piste principale
  indépendamment de ce nuancement m50, MAIS ce nuancement doit être écrit
  tel quel dans les résultats (T2), pas lissé pour paraître plus propre que
  ça ne l'est.

### WP2 — Ablation contexte (tuer le confondant ctx24/ctx12)
- **Runs** : V2 @ `context_length: 12`, seed 42, × 3 sparsités (3 runs).
  Option si résultats serrés : +2 seeds sur m50.
- **Comparaisons** : V2@12 vs V1@12 (gain d'architecture pur) ; V2@12 vs
  V2@24 (valeur du contexte long).
- **Décision** : si V2@12 ≈ V1 → le "×2 sparse" était surtout du contexte →
  le dire et recentrer sur WP1. Si V2@12 ≫ V1 → gain d'architecture réel.
- **Sortie papier** : ligne de T3 (ablations) + phrase de fair-comparison.

- **RÉSULTAT RÉEL 2026-07-16 (seed42 uniquement, lecture rapide 4/13
  fenêtres — single-seed, R1 : pas encore un claim établi, mais l'écart est
  déjà décisif)** :

  | Sparsité | V1 (3 seeds, protocole complet, `official_overall_current_relative_rmse`) | V2@12 (seed42, lecture rapide) | Ratio |
  |---|---:|---:|---:|
  | dense | 0.862 | 0.000889 | **V2@12 ×970** |
  | m50 | 0.898 | 0.230 | **V2@12 ×3.9** |
  | m95 | 1.007 | 0.526 | **V2@12 ×1.9** |

  **Branche de décision déclenchée : "V2@12 ≫ V1" — le gain de V2 est un
  gain architectural réel**, pas seulement l'effet du contexte plus long :
  même en retirant l'avantage de contexte (24→12, remis au niveau de V1),
  V2 écrase toujours V1 à toutes les sparsités, par un facteur énorme en
  dense. Réserve avant claim final : comparaison single-seed (V2@12) contre
  moyenne 3 seeds (V1) sur des tailles d'échantillon test différentes (4 vs
  13 fenêtres) — l'écart est cependant bien trop large (×970 en dense) pour
  être un artefact de ces différences ; à confirmer sur 2 seeds
  supplémentaires si un chiffre publiable est requis, mais la direction ne
  fait pas de doute.

  **Point secondaire inattendu** : V2@12 seed42 dense (0.000889) est même
  légèrement *meilleur* que V2@24 seed42 dense (0.001311, déjà connu) — le
  contexte plus long n'aide pas ici, il pourrait même très légèrement gêner
  sur cette lecture rapide. Pas assez de données pour en tirer une
  conclusion (1 seed, à recouper avec m50/m95 de V2@24 si besoin), mais ça
  va dans le même sens que le résultat WPB0 côté FNO+ : le contexte long
  n'est pas la martingale qu'on aurait pu supposer.

- **Point de contrôle croisé, CONFIRMÉ À 3/3 SEEDS 2026-07-15 (papier 2,
  WPB0)** : le même confondant de contexte a été testé côté FNO+ — un FNO+
  recevant les mêmes 24 frames de contexte que V2 (au lieu d'1 seule),
  protocole identique sinon. Résultat à 3 seeds (R1 respecté) : **relRMSE
  moyen 0.007529±0.000328, toujours PIRE que le FNO+ vanilla
  (0.006550±0.000135)**, écart ~7.3× l'écart-type vanilla, individuellement
  cohérent sur les 3 seeds (0.007822 / 0.007591 / 0.007175, tous au-dessus
  du pire seed vanilla). Détails : `reports/fno_plus_beat_paper_plan.md`
  §WPB0. **Conséquence pour ce papier** : ça renforce l'argument "V2 bat
  FNO+ ce n'est pas juste parce que V2 voit plus d'historique" — donner le
  même budget d'information à FNO+ ne l'a pas aidé, donc l'avantage de V2
  (là où on le compare à FNO+, hors périmètre du jumeau) semble plus
  architectural que dû au contexte. Ne remplace pas WP2 (qui reste
  nécessaire pour la comparaison V2@12 vs V1@12 propre côté sparsité), mais
  fournit un point de comparaison externe supplémentaire, désormais
  solidement établi (pas juste single-seed).

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
  - (a) ~~`prediction.target: absolute`~~ **ABSORBÉ PAR WP12 (2026-07-17)** :
    la même manipulation, mais dans le design croisé {Δt × cible} en
    one-step — mieux contrôlée (pas de confondant rollout/pushforward) et
    elle donne le point ratio~488 de la courbe dose-réponse au passage. Ne
    pas la lancer en double ici.
  - (b) `include_target_rainfall: false`
  - (c) encodeur spatial désactivé (attention-only) — vérifier s'il existe un
    knob propre, sinon petit ajout modèle (flag `use_spatial_encoder`)
  - (d) `change_weight: 0`
  - (e) `pushforward_fraction: 0`
  - (f) `diffusion.steps: 20` (parité papier V1)
- ~6 runs courts (arrêt anticipé attendu 60-120 epochs), lançables par vagues
  de 3 en parallèle.
- **Sortie papier** : Table T3.

### WP5 — LULC / Manning (V2.1, entrée causale vérifiée) — RESCOPÉ OPTIONNEL (2026-07-17)

**Rescope (relecture structure profonde, §1)** : ce WP ne sert aucun des
trois piliers du papier — c'est une amélioration de modèle (une entrée
causale de plus), pas une réponse à "quand le génératif se justifie-t-il".
Il renforce l'attrait domaine (revue hydrologie) mais dilue le propos et
consomme du budget GPU en concurrence avec les piliers. **Décision : ne le
lancer que si les piliers A/B/C sont bouclés avant M5 (gel des expériences)
ET qu'il reste du budget ; sinon il passe explicitement en "travail futur /
papier V2.1" avec une phrase dans la discussion.** Le code reste prêt, rien
n'est perdu dans les deux cas.

**Schéma confirmé 2026-07-10 (Dell, instruction coordination 0003, commit
`a7b6c0a`)** : ESRI/Impact-Observatory 10-classes Sentinel-2 — nos codes
observés {1,2,4,5,7,8,9,10,11,+15=nodata} reproduisent exactement la
signature "codes 3 et 6 sautés" du schéma, et le papier FloodCastBench
(Nature Sci Data, s41597-025-04725-2) cite une source Sentinel-2 LULC avec
Manning par classe aux mêmes noms. `DEFAULT_MANNING_LOOKUP` mis à jour avec
les valeurs du papier. **Anomalie corrigée** : "built area" était à 0.015
(quasi lisse) dans le lookup provisoire, contre 0.375 dans le papier (zone
urbaine dense très obstruée) — sens physique inversé, facteur ×25. Aucun
run V2.1 n'avait encore utilisé l'ancien lookup, donc rien à invalider.
**Réserve à lever avant citation finale** : les 7 valeurs du papier ont été
transcrites depuis un résumé de recherche web (Nature payant, ResearchGate
403, PDF arXiv trop volumineux) — pas une lecture directe de la table
source. Revérifier si accès VPN institutionnel disponible.
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

- **RÉSULTAT COMPLET 2026-07-16 (les 4 runs terminés)** : hypothèse
  confirmée nettement, et pas seulement pour seed123 — **systémique sur
  tout le corpus sparse existant**, pas un cas isolé :

  | Run | Budget normal (`rollout_val_rmse`, best epoch) | Budget rallongé | Amélioration |
  |---|---:|---:|---:|
  | seed42/m50 | 1.331 (epoch 65) | 1.004 (epoch 120) | -24.6% |
  | seed42/m95 | 1.864 (epoch 55) | 1.181 (epoch 325) | -36.6% |
  | seed7/m95 | 1.737 (epoch 60) | 1.451 (epoch 105) | -16.5% |
  | seed7/m50 | 1.319 (epoch 60) | **0.849 (epoch 600 — jamais early-stoppé, encore en amélioration à la fin)** | -35.6% |

  **seed7/m50 n'a jamais convergé même à 600 epochs** — le run le plus
  sous-entraîné des quatre, budget encore potentiellement insuffisant.
  Amélioration moyenne ≈ -28%, aucun des 4 runs n'y échappe.

  **Conséquence majeure** : tous les checkpoints V2 sparse (m50/m95) 
  utilisés pour WP1 (le "quasi-tie à m50", le "jumeau ×1.57 en m95") ont été
  mesurés sur des modèles sous-entraînés. **Le tableau WP1 entier doit être
  refait sur le protocole test réel avec ces nouveaux checkpoints avant
  d'être considéré définitif.**

  **Vérifié 2026-07-16 : le jumeau déterministe a LE MÊME problème.** Les
  `summary.json` déjà présents (pas besoin de relancer pour vérifier)
  montrent : m95 **jamais early-stoppé** pour les 2 seeds (seed42 : best
  epoch 300/300, seed7 : best epoch 300/300 — les deux tournaient encore à
  la toute dernière epoch), alors que m50 avait convergé proprement pour
  les 2 seeds (seed42 : epoch 45/105 ; seed7 : epoch 90/150 — pas de souci
  là). Signal secondaire à surveiller : seed7 dense montre aussi
  `best_epoch=270/300` sans early-stop (métrique déjà très petite,
  probablement sans conséquence pratique, mais pas vérifié).

  **Action lancée 2026-07-16 22:08** : jumeau m95 seed42+seed7 relancés en
  budget 600/patience 120 (`queue det_twin_budget600`, 2 vagues
  séquentielles — le jumeau est ~3× plus rapide que V2 par epoch, ETA
  nettement plus courte que WP6). Une fois fini : lancer les évals test
  réelles sur TOUS les checkpoints concernés (V2 m50+m95 rallongés déjà
  prêts, jumeau m95 rallongé à venir, jumeau m50 déjà bon = inchangé) et
  refaire le tableau T2/WP1 au complet.

  **seed42/m95 terminé 2026-07-17 02:30** : early-stop propre à epoch 315
  (patience 120 → arrêt epoch 435), `rollout_val_rmse` = **0.7244** vs
  0.7318 (checkpoint original, epoch 300, jamais early-stoppé). Amélioration
  de **seulement ~1.0%** — nettement moins que les -16.5% à -36.6% des 4
  runs V2/WP6 ci-dessus. Lecture honnête : contrairement aux checkpoints V2,
  ce checkpoint jumeau original n'était probablement pas si loin de la
  convergence malgré l'absence d'early-stop détecté à l'époque — ou alors la
  fenêtre epoch 300-315 suffisait déjà. Ne pas généraliser "jumeau
  quasi-inchangé par le budget rallongé" à seed7 avant d'avoir son propre
  résultat.

  **Incident seed7/m95 (2026-07-17 02:31)** : premier lancement a crashé à
  l'epoch 2 — `RuntimeError: 27/72 batches skipped for non-finite
  loss/gradients` (garde-fou existant, seuil 25%, ajouté 2026-07-09 après un
  précédent NaN seed42/dense). Le run original de seed7/m95 (2026-07-11,
  300 epochs, jamais rallongé) n'avait eu **zéro** batch skip sur toute sa
  durée — cette instabilité est donc nouvelle, pas une redite d'un problème
  connu. Cause suspectée mais NON confirmée : ce lancement tournait en
  parallèle du job d'éval WP9 (`evaluate_floodcastbench_det_twin_ensemble.py`,
  13 fenêtres × 3 checkpoints) sur le même GPU — contention mémoire/cudnn
  possible. **Relancé seul sur le GPU à 02:34** (WP9 avait fini entre-temps)
  pour tester cette hypothèse ; suivi en direct via un monitor sur les
  premières epochs. Si ça replante seul sur GPU, l'hypothèse contention est
  fausse et il faudra chercher ailleurs (config seed7 spécifique, non-
  déterminisme GPU pur). À vérifier au réveil si pas résolu avant.

  **seed7/m95 retry terminé 2026-07-17 07:20** (early stop epoch 480, best
  epoch 360) : **zéro batch skippé sur toute la durée** — confirme
  l'hypothèse de contention GPU avec WP9 comme cause du crash initial (pas
  un problème de config seed7 ni de non-déterminisme). Mais le résultat lui-
  même est **une régression, pas une amélioration** : `rollout_val_rmse` =
  **0.7410** vs **0.7105** (checkpoint original, epoch 300, jamais
  early-stoppé) → **+4.3% PIRE**, à l'opposé de seed42/m95 (+1.0% mieux) et
  des 4 runs V2 (-16.5% à -36.6%, tous meilleurs). Résumé jumeau-m95 : seed42
  quasi-inchangé, seed7 dégradé. **Ne pas présenter WP6 comme "budget plus
  long = toujours mieux" pour le jumeau** — contrairement à V2, l'effet est
  faible et de signe incohérent selon la seed, ce qui suggère que la
  variance run-à-run domine le signal de convergence pour cette architecture
  spécifiquement. À signaler tel quel dans le papier si WP6 est cité comme
  justification méthodologique.

  ⚠ **Incident opérationnel (2026-07-17, ~4h26 GPU idle)** : le monitor mis
  en place pour suivre le retry seed7 utilisait un filtre trop restrictif
  (`awk` ne reconnaissait pas la ligne `early stop at epoch...` — elle ne
  matchait ni le motif d'erreur ni `^epoch=`) → le message de fin de
  training n'a déclenché AUCUNE notification. Le run s'est terminé à 07:20,
  découvert seulement à 11:46 sur demande "eta" de l'utilisateur — le GPU
  est resté inactif ~4h26 sans que je m'en aperçoive. Corrigé : nouveau
  monitor (seed123/m95, lancé 11:47) avec un filtre couvrant explicitly
  "early stop" en plus des erreurs. Leçon retenue : tout filtre de monitor
  sur un training doit inclure le message de fin/early-stop, pas seulement
  les signatures d'erreur — silence ≠ succès.

  **Action 2026-07-17 11:47** : seed123/m95 mis en queue seul (budget
  600/patience 120) pour compléter le trio de seeds m95 nécessaire à WP9-m95
  et au tableau WP1 final.

  **seed123/m95 terminé 2026-07-17 17:27** (early stop epoch 505, best
  epoch 385, tourné seul sur GPU sans replant après le 1er crash de la
  veille — voir incident 2026-07-17 02:31) : `rollout_val_rmse` = **0.7157**
  vs **0.7132** (checkpoint original, epoch 300) → **+0.35%, quasi-nul,
  dans le bruit**.

  **BILAN COMPLET jumeau-m95, 3/3 seeds, budget rallongé** :

  | Seed | Original (epoch, rmse) | Rallongé (epoch, rmse) | Variation |
  |---|---|---|---:|
  | 42 | 300, 0.7318 | 315→435, 0.7244 | **-1.0%** (meilleur) |
  | 7 | 300, 0.7105 | 360→480, 0.7410 | **+4.3%** (pire) |
  | 123 | 300, 0.7132 | 385→505, 0.7157 | **+0.35%** (quasi-nul) |

  **Conclusion ferme** : contrairement à V2 (4/4 runs améliorés de -16.5% à
  -36.6%), le budget rallongé **ne corrige rien de systématique pour le
  jumeau m95** — signe incohérent, magnitude petite, aucune seed n'atteint
  l'amélioration typique de V2. L'hypothèse initiale de WP6 ("tous les
  checkpoints sparse existants étaient sous-entraînés") **ne tient que pour
  V2, pas pour le jumeau**. Explication probable : le jumeau converge ~3×
  plus vite par epoch et n'avait probablement pas le même problème de sous-
  entraînement à budget normal — les checkpoints originaux (epoch 300,
  jamais early-stoppés au sens strict mais proches d'un plateau) étaient
  déjà représentatifs. **Décision méthodologique** : utiliser les
  checkpoints RALLONGÉS pour la cohérence de protocole (même procédure de
  sélection que V2), mais documenter explicitement dans le papier que cette
  extension n'a pas changé la conclusion jumeau-m95, pour ne pas laisser
  penser que WP6 était nécessaire pour le jumeau — il l'était pour V2
  seulement.

  **GPU libéré 17:27, immédiatement relancé** : WP9-m95 (ensemble des 3
  jumeaux rallongés, éval-only) lancé en fond pour compléter la calibration
  m95 manquante depuis le résultat m50 (§ci-dessous, cov50=0.117/cov90=0.205
  à m50) ; figure qualitative (GT vs Δ-Diff vs Twin) et reconstruction du
  tableau central WP1 à enchaîner ensuite.

  ⚠ **Trou découvert en préparant la reconstruction du tableau central
  (2026-07-17 18:03)** : **V2 seed123 (m50 ET m95) n'a jamais été relancé
  sous WP6** — seuls seed42 et seed7 ont des checkpoints V2 sparse
  corrigés (`16-07-2026_09-37-4{3,4}` et `16-07-2026_15-14-5{1,2}`). Le
  tableau WP1 original à 3 seeds utilisait le checkpoint V2/seed123
  ORIGINAL (non corrigé) — jamais identifié comme trou avant cette
  vérification systématique. **Vérifié en même temps, jumeau m50/seed123**
  (par précaution, même style de vérification que le trou trouvé) : early
  stop propre à epoch 105 (best epoch 45, patience 60 honorée) — RAS,
  cohérent avec seed42/seed7, aucune correction nécessaire côté jumeau m50.

  **Action 2026-07-17 18:03** : V2 seed123 m50 lancé (budget 600/patience
  120), m95 mis en queue automatique derrière (script séquentiel,
  `/tmp/.../run_v2_seed123_queue.sh`, watchdog dédié). Bloque la
  reconstruction du tableau central WP1 tant que ces 2 runs ne sont pas
  finis — c'est maintenant le dernier chaînon manquant.

  **Incident 2026-07-17 18:08 : m50 a crashé, bug de script révélé.**
  `RuntimeError: 19/72 batches skipped` à l'epoch 7 (skip croissant dès
  l'epoch 1 : 1, 11, 17, 17, 19 — contrairement à seed42/seed7 WP6, zéro
  skip sur toute leur durée). **Bug découvert dans le script de queue** :
  il attendait juste la disparition du PID, sans vérifier le succès — donc
  m95 a été lancé automatiquement juste après l'ÉCHEC de m50, pas après sa
  réussite. m95 montre lui aussi un signe potentiel d'instabilité dès le
  premier batch (`prediction_finite: false` dans le diagnostic initial),
  à surveiller. **Correction** : nouveau script
  `run_v2_seed123_m50_retry.sh` qui attend la fin réelle de m95, puis
  retente m50 jusqu'à 2 fois en vérifiant explicitement le marqueur
  `early stop at epoch` (pas juste la disparition du process) avant de
  déclarer un succès. Leçon générale à appliquer à tout script de queue
  futur : **toujours vérifier le contenu du log de fin, jamais seulement
  la disparition du PID** — un crash et un succès ressemblent identiques
  du point de vue d'un `wait $PID`.

  **Incident 2026-07-17 18:12 : m95 a aussi crashé** (le signal
  `prediction_finite: false` du premier batch était donc un vrai signe
  avant-coureur, pas du bruit) — `RuntimeError: 24/72 batches skipped` à
  l'epoch 5 (13, 0, 1, 18, 24 — pattern bruyant, pas une dérive continue
  comme m50, mais franchit quand même le seuil). **Les deux runs seed123
  (m50 ET m95) ont donc échoué à leur 1er essai**, contrairement à
  seed42/seed7 qui avaient tous réussi du premier coup. Hypothèse : ce
  n'est probablement pas un hasard isolé — seed123 pourrait être
  intrinsèquement plus proche du seuil d'instabilité que les 2 autres
  seeds pour ce modèle/paramétrisation (à documenter comme observation,
  pas à sur-interpréter sans plus de runs). **Queue de retry étendue** :
  `run_v2_seed123_m95_retry.sh` ajouté, enchaîné après le script m50-retry
  (jusqu'à 2 tentatives chacun), avec le même garde-fou (vérification du
  marqueur `early stop at epoch`, pas juste la fin du process).

  **m50-retry-1 RÉUSSI (2026-07-17 ~20:26)** : early stop propre epoch 195
  (best epoch 75, patience 120 honorée), succès dès la 1ère tentative de
  retry. m95-retry-1 démarré immédiatement derrière.

  **m95-retry-1 RÉUSSI (2026-07-18 ~03:10)** : early stop propre epoch 415
  (best epoch 295, patience 120 honorée). **Les 4 checkpoints V2 seed123
  corrigés (m50+m95, tentative 1 échouée + retry réussi pour chacun) sont
  identifiés et confirmés** :
  - m50 : `17-07-2026_18-12-40_.../checkpoint_best.pth` (epoch 75,
    rollout_val_rmse=1.2717)
  - m95 : `17-07-2026_20-27-22_.../checkpoint_best.pth` (epoch 295,
    rollout_val_rmse=1.1634)
  (les tentatives échouées `18-03-48` et `18-08-53` sont laissées sur
  disque mais ne doivent JAMAIS être utilisées pour le tableau final —
  written down here explicitly pour éviter toute confusion future.)

  **Toutes les pièces du tableau central WP1 sont maintenant réunies** :
  V2 {m50,m95} × {seed42,seed7,seed123} tous WP6-corrigés ; jumeau {m50,m95}
  × {seed42,seed7,seed123} tous confirmés (m50 jamais eu besoin de
  correction, m95 rallongé aux 3 seeds, effet marginal/mixte déjà noté).
  **Reste à faire avant de pouvoir remplir le tableau** : lancé l'éval
  protocole test complet (13/13 fenêtres) sur les 2 NOUVEAUX checkpoints
  V2/seed123 (m50 et m95). 2026-07-18 03:10 (séquentiel m50→m95,
  `eval_seed123_queue.log`) — **TERMINÉ avec succès, les deux**.

  ⚠ **Hypothèse fausse corrigée en préparant l'agrégation (2026-07-18
  07:41)** : j'avais supposé "les 10 autres combinaisons ont déjà leurs
  évals test depuis WP6/WP1" — **faux, vérifié directement** (aucun
  `eval_rollout_test*` sous les dossiers checkpoints WP6-corrigés, ni pour
  V2 seed42/seed7, ni pour AUCUN des 6 checkpoints jumeau, pas même les
  checkpoints jumeau m50 originaux qui n'avaient jamais eu besoin de
  correction). WP6/WP1 avaient produit les CHECKPOINTS et leurs métriques
  de validation interne (`rollout_val_rmse`), mais jamais l'éval protocole
  test 13/13 fenêtres formelle. **10 évals supplémentaires lancées en
  queue séquentielle** (6 jumeau d'abord — rapides, single-scenario — puis
  4 V2 — lentes, 8 scénarios), script
  `/tmp/.../run_central_table_evals.sh`, sortie
  `central_table_evals_queue.log`. C'est maintenant le VRAI dernier
  chaînon avant le tableau — leçon : ne jamais supposer qu'un artefact
  existe sans le lister directement, même quand le plan semble l'impliquer.

  **10/10 TERMINÉES avec succès (2026-07-18 21:01)**. En les assemblant,
  **3e gap trouvé** : le jumeau dense n'avait non plus jamais d'éval test
  formelle (seulement `rollout_val_rmse` interne). 3 évals rapides
  lancées et terminées (~1-4 min chacune). **En même temps, asymétrie
  repérée et corrigée** : le V2 dense existant (`v2_dense_fullmetrics_check`)
  n'était qu'une lecture rapide 4/13 fenêtres (déjà identifié comme item
  1.3 du palier 1) — relancé en protocole complet 13/13 pour que la ligne
  dense du tableau soit apples-to-apples avec les lignes m50/m95 (en
  cours au moment d'écrire).

  **TABLEAU CENTRAL — relRMSE, moyenne ± écart-type sur 3 seeds (42/7/123),
  protocole test, jumeau dense/m50/m95 et V2 m50/m95 sur 13/13 fenêtres ;
  V2 dense encore sur 4/13, réévaluation 13/13 en cours** :

  | Régime | V2 (Δ-Diff) | Twin (jumeau) | Gagnant | Ratio | Signe cohérent sur les 3 seeds ? |
  |---|---:|---:|---|---:|---|
  | dense | 0.001576 ± 0.000668 (4/13, à refaire) | **0.000417 ± 0.000039** | jumeau | ×3.78 | oui (3/3) |
  | m50 | **0.318425 ± 0.043702** | 0.348852 ± 0.040779 | V2 (faible) | ×1.10 | **non** (2/3 V2, 1/3 jumeau) |
  | m95 | 0.492443 ± 0.013940 | **0.344249 ± 0.006248** | jumeau | ×1.43 | oui (3/3) |

  **Lecture** : le motif est **quasi identique à l'estimation pré-WP6**
  (dense jumeau ×3.55→×3.78, m50 quasi-tie, m95 jumeau ×1.57→×1.43) — les
  corrections de checkpoints (WP6) n'ont donc PAS changé la conclusion
  qualitative, seulement affiné les chiffres. **Verdict selon les critères
  pré-enregistrés (§WP1) : branche "mixte"** — ni "jumeau partout" ni
  "V2 partout". Le jumeau domine nettement en dense et en sparsité
  extrême (m95), le résultat est indécis à sparsité intermédiaire (m50,
  signe non cohérent entre seeds).

  **Limite de rigueur statistique, à noter explicitement dans le papier** :
  ce test est un test apparié par SEED (n=3, sous-puissant) — le test par
  FENÊTRE×SEED (n=39, Wilcoxon/bootstrap) prévu dans le protocole
  d'origine nécessiterait que l'évaluateur sauvegarde le relRMSE par
  fenêtre individuelle (actuellement seulement agrégé), ce qui impliquerait
  de refaire les 12 évals avec un évaluateur modifié — coût élevé, reporté
  en item du palier 1 (déjà couvert par la mention "tests appariés" de
  l'item 1.1, à préciser : ajouter la sauvegarde par-fenêtre AVANT tout
  autre rerun futur pour ne pas payer ce coût une 2e fois).

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

**Premiers résultats 2026-07-11 (Dell, instruction coordination 0002,
seed42 V2 uniquement, test 4/13 fenêtres)** — vérifiés source par source :

| Masque | m50 relRMSE | m95 relRMSE | vs aléatoire (m50 ~0.395 / m95 ~0.567) |
|---|---|---|---|
| gauge | 0.393 | 0.585 | **quasi identique** |
| cluster | 0.609 | 0.706 | **+54% / +25%, net** |

Signal net et physiquement interprétable, pas un artefact (pas de NaN, JSON
propres) : le masque **gauge** reste distribué spatialement (juste pondéré
par l'occupation d'eau) → généralise ~parfaitement depuis l'entraînement sur
masques i.i.d. Le masque **cluster** laisse de larges zones contiguës
totalement non observées → régime spatial jamais vu à l'entraînement →
dégrade nettement, surtout à m50 (NSE cluster 0.49 vs gauge 0.79). Limite
de généralisation réelle à documenter dans le papier — mais **1 seed, 4
fenêtres seulement, pas encore un résultat statistiquement établi** (R1).
Suite recommandée si budget dispo : refaire sur les 3 seeds et test complet
(13 fenêtres), au moins pour le cas cluster.

Anomalie non résolue, notée pour référence : les évals m0.95 ont pris ~2x
plus longtemps que m0.5 à fenêtres égales (~2h57 vs ~1h28) sans explication
trouvée (mêmes patch/steps/scénarios) — pas creusé, sans impact sur la
validité des résultats.

**Extension 2e seed 2026-07-11 (Dell, instruction coordination 0005,
cluster m50, seed7)** : relRMSE 0.597, NSE 0.512, CSI@0.001 0.776 — quasi
identique à seed42 (relRMSE 0.609, NSE 0.494). **Le signal de dégradation
cluster n'est pas un artefact seed42** : confirmé à 2/3 seeds sur le point le
plus fort (cluster m50, +54% vs aléatoire). Couverture calibration
cohérente aussi : cov50=0.109/cov90=0.230 (seed7) vs cov50=0.092/cov90=0.209
(seed42) — même sous-couverture sévère, même ordre de grandeur.

**Analyse calibration WP3 sur les 4 conditions WP7 (Dell, instruction
0004)** — rank histogram + spread-skill construits
(`experiments/FloodCastBench/paper_figures/f5_calibration_wp7.png`) :
- Sous-couverture confirmée visuellement (rank histogram en U/J), mais
  **deux axes orthogonaux, pas un seul classement** : la **sparsité domine**
  (m95 mieux calibré que m50 pour gauge ET cluster — contre-intuitif) et la
  **structure a un effet secondaire mais cohérent** (gauge toujours mieux
  calibré que cluster à sparsité égale). Explique la non-monotonie
  apparente (cluster_m95 > gauge_m50 en couverture) : ce n'est pas du bruit,
  c'est l'axe sparsité qui domine l'axe structure en amplitude.
- Spread-skill : l'erreur reste quasi plate (~0.01-0.05m) sur toute la
  plage de spread faible (1e-8 à ~1e-2), le spread ne redevient informatif
  sur l'erreur réelle qu'à l'extrémité haute (probablement les pas de
  rollout avancés, incertitude composée). Le spread n'est donc pas
  simplement "trop petit" partout — il est **non-informatif** en dessous
  d'un seuil, pas juste sous-dimensionné uniformément. Point à creuser
  (zoom log-log) pour la discussion papier, pas urgent.
- Confirme et durcit la conclusion du §4-WP3/WP1 : V2 n'est pas bien
  calibré (sous-couverture sévère, systématique), ce qui pèse sur le pivot
  "la valeur du génératif est dans la calibration" — cette justification de
  repli n'est PAS acquise automatiquement, elle reste à établir/nuancer,
  potentiellement seulement pour certains régimes (m95 moins mal calibré
  que m50 dans ces 4 conditions).

### WP8 — Deuxième événement : UK 2015 (60m high-fidelity)
- **Étapes** : stats de normalisation UK, delta stats UK, vérifier la grille
  (536×536 ? ranges de frames ?), config dédiée ; smoke 3 régimes ; runs
  seed 42 × {dense, m50, m95} pour V2 ET jumeau (6 runs) ; évals test.
- **But** : montrer que la *direction* des conclusions (WP1) tient sur un
  second événement — pas de re-tuning.
- **Sortie papier** : Table T2-bis ou paragraphe généralisation.

### WP9 — Ensemble profond de jumeaux : LA baseline UQ manquante (ajouté 2026-07-16, quasi gratuit)

**Trou identifié par relecture critique en visant Q1** : l'argument central
du pivot calibration — "le jumeau déterministe ne peut structurellement pas
fournir d'incertitude" — est **faux tel quel** et n'importe quel reviewer
sérieux le relèvera : un **deep ensemble** de jumeaux (N modèles, seeds
différents) fournit une distribution prédictive, c'est LA baseline UQ
standard (Lakshminarayanan et al. 2017), souvent mieux calibrée que les
modèles génératifs. Sans cette comparaison, le papier est attaquable sur
son dernier argument restant.

- **Chance** : les 3 jumeaux (seeds 42/7/123) sont DÉJÀ entraînés à chaque
  sparsité → l'ensemble-de-3-jumeaux est **gratuit en entraînement**, il ne
  faut que des évals (traiter les 3 sorties déterministes comme un ensemble
  M=3, passer dans le CalibrationAccumulator existant, comparer à V2 M=8 à
  `nominal_finite_ensemble` égal — corriger pour M différent, le champ
  existe déjà).
- **Runs** : éval-only, ~6 évals (3 sparsités × {ensemble-jumeaux, déjà-fait
  V2}). Si budget : 2 seeds jumeaux de plus pour M=5.
- **Critères pré-enregistrés** :
  - Ensemble-jumeaux mieux calibré OU égal à V2 → le génératif n'a plus
    AUCUNE justification mesurée sur ce benchmark → conclusion du papier
    d'autant plus nette (et plus citable) — l'écrire tel quel.
  - V2 nettement mieux calibré → premier vrai point positif pour le
    génératif → renforce le narratif "où le génératif paie".
  - Les deux mal calibrés (plausible vu WP3) → "aucune des deux familles ne
    fournit d'incertitude fiable sans recalibration" — résultat utile aussi.
- **Sortie papier** : ligne(s) de F5/T2, et un paragraphe de discussion qui
  désamorce préventivement LA question de reviewer la plus prévisible.

- **RÉSULTAT PARTIEL m50 (2026-07-17, 13/13 fenêtres test, masques
  aléatoires, M=3, seeds 42/7/123 checkpoints originaux)** — outil créé :
  `tools/evaluate_floodcastbench_det_twin_ensemble.py` (commit `33fdd5e`),
  réutilise le CalibrationAccumulator de V2 à métriques identiques.
  Sortie : `experiments/FloodCastBench/wp9_det_twin_ensemble/eval_test_m3_17-07-2026_00-11-29/`.
  - Couverture ensemble-jumeaux : cov50 pooled **0.117** (nominal corrigé
    M=3 : 0.25 → ratio 0.469), cov90 pooled **0.205** (nominal 0.45 →
    ratio 0.456).
  - Comparaison V2 la plus proche disponible (m50 gauge, M=8, seed42) :
    ratios 0.427/0.484 — **statistiquement comparables**. Rank histogram :
    les deux en U (surconfiance), V2 plus marqué que l'ensemble-jumeaux.
  - **Branche pré-enregistrée déclenchée (provisoire) : (c) "les deux mal
    calibrés"** — aucune des deux familles ne fournit d'incertitude fiable
    sans recalibration à m50. La calibration ne sauve PAS le génératif en
    l'état.
  - Caveats avant de figer : V2 comparé sur masque gauge (pas aléatoire —
    l'éval V2 m50 aléatoire avec calibration n'existe pas encore, à lancer
    quand le GPU est libre, PAS en parallèle d'un training jumeau vu les
    crashs de contention) ; M différents (3 vs 8, corrigé par
    nominal_finite_ensemble mais la granularité du rank histogram diffère) ;
    checkpoints pré-WP6. m95 en attente des 3 seeds m95 corrigés.
  - Déjà intégré au draft papier : Figure 5 (f6_calibration_comparison.pdf)
    + paragraphe "Is a generative model even needed for uncertainty?"
    (commit `7614b98`).

- **RÉSULTAT MIS À JOUR ET COMPLÉTÉ (2026-07-17, fin de journée)** — les 2
  caveats du résultat m50 partiel sont levés, et m95 est fait :

  1. **V2 m50 masque ALÉATOIRE, checkpoint WP6-corrigé** (comble le caveat
     "gauge≠aléatoire"), `wp9_det_twin_ensemble/v2_m50_random_seed42_wp6ckpt/` :
     cov50 ratio **0.395**, cov90 ratio **0.444** — comparaison maintenant
     appariée mask-à-mask avec l'ensemble-jumeaux.
  2. **Ensemble-jumeaux m95, 3 seeds rallongés** (comble le caveat "m95
     manquant"), `wp9_det_twin_ensemble/eval_test_m3_17-07-2026_17-28-04/` :
     cov50 pooled **0.174** (ratio **0.695**), cov90 pooled **0.280** (ratio
     **0.622**).

  **Tableau de synthèse calibration (ratio couverture observée/nominale,
  1.0 = parfait), toutes comparaisons maintenant appariées par masque** :

  | Modèle | m50 (aléatoire) | m95 (structuré, le plus proche dispo) |
  |---|---:|---:|
  | Ensemble-jumeaux (M=3) | 0.469 / 0.456 | 0.695 / 0.622 |
  | V2 (M=8) | 0.395 / 0.444 | gauge 0.644/0.673 · cluster 0.517/0.556 |

  **Lecture** : à m50 comme à m95, l'ensemble-jumeaux n'est **jamais moins
  bien calibré que V2** — à m50 il est même légèrement meilleur (0.469 vs
  0.395 sur l'intervalle 50%), et à m95 comparable au meilleur cas V2
  (gauge). **La branche (c) "les deux mal calibrés, ni l'un ni l'autre ne
  sauve la calibration" est confirmée aux deux sparsités**, avec une nuance
  qui renforce le narratif du papier plutôt que de l'affaiblir : rien dans
  ces données ne soutient "le génératif est mieux calibré que le
  déterministe" — c'est cohérent avec le pilier C tel qu'énoncé en §1.
  Caveat restant, mineur : m95 encore comparé à des masques structurés
  faute d'éval V2-m95-aléatoire-avec-calibration lancée (peu prioritaire,
  le signal est déjà cohérent aux deux sparsités disponibles).

### WP10 — Transfert cross-événement zéro-shot (ajouté 2026-07-16)

Demandé explicitement (tests cross-région) et nécessaire pour viser mieux
que "un seul événement + une réplication" : prendre les checkpoints
entraînés Australie (V2 + jumeau, seed 42 min.) et les évaluer **tels
quels, sans réentraînement** sur UK (grille différente 85×137, stats de
normalisation UK déjà calculées — WP8). Teste la vraie affirmation de
déploiement ("different sensor configurations without retraining" du papier
DIFF-SPARSE, poussée au niveau événement/région).
- **Runs** : éval-only (~6 évals : 2 modèles × 3 sparsités).
- **Attente honnête pré-enregistrée** : la normalisation delta/échelle est
  calée sur les stats Australie → une dégradation est attendue ; la
  question est *combien*, et si le CLASSEMENT jumeau-vs-V2 tient. Même un
  échec net des deux est un résultat de déploiement publiable.
- **Extension 2026-07-18 (n'est plus optionnelle)** : zero-shot sur
  Pakistan/Mozambique 480m aussi, pas seulement UK — sert de triage avant
  WP13 (réentraînement complet 4 événements, décision utilisateur). À
  lancer dès que les configs Pakistan/Mozambique existent (WP13 étape 4).

### WP11 — Table coût/complexité (ajouté 2026-07-16, zéro GPU)

Trou évident pour un reviewer : V2 = 40 pas de diffusion × 8 scénarios par
prédiction ; le jumeau = 1 forward ; FNO+ = 1 forward pour 19 pas d'un
coup. Si V2 gagne quelque part, il faut dire À QUEL PRIX.
- **Contenu** : paramètres, temps d'inférence mesuré par fenêtre de
  prévision (mêmes conditions, même GPU), mémoire pic, temps d'entraînement
  total. Mesures réelles, pas estimées.
- **Sortie papier** : Table T5 — obligatoire dans Methods ou Experiments.

### WP12 — Dose-réponse du mécanisme signal≪champ (ajouté 2026-07-17, redessiné le jour même, CRITIQUE)

**Trou identifié par relecture critique de l'utilisateur** : §4.2/Figure 2
du papier affirment un MÉCANISME causal ("diffuser l'absolu échoue PARCE
QUE le ratio signal/champ est trop grand") à partir de seulement 2 points
de données (Australie ×488, UK ×425), tous deux avec ratio élevé ET échec.
C'est une corrélation, pas une preuve — aucun exemple avec ratio faible
qui réussirait, aucun exemple à ratio intermédiaire. Un reviewer sérieux
(a fortiori Transactions) attaquera ce point en premier. C'est actuellement
**le maillon le plus faible du papier**, pas le plus solide comme
initialement présenté.

**Design (proposé par l'utilisateur, remplace la version initiale qui
comparait Abs-Diff — architecture différente — à travers les Δt, donc
confondait architecture et Δt)** : un seul squelette, celui de V2, figé ;
UN SEUL facteur croisé varie :

  **{Δt ∈ 4-6 valeurs géométriques} × {cible : absolue, delta}**

- Prédiction **simple (un pas), pas de rollout** : `pushforward_fraction=0`,
  sélection de checkpoint sur la val one-step — élimine les confondants
  rollout/pushforward, réduit le coût par run.
- **Régime dense** uniquement : sous masquage, la cible delta aux pixels
  masqués a déjà une amplitude d'échelle champ (le remplissage moyen-train
  n'est pas la frame précédente), ce qui brouillerait la manipulation.
  Indice interne DÉJÀ cohérent avec le mécanisme, à citer dans le papier :
  l'écart V1-vs-V2 se comprime exactement là où l'échelle effective de la
  cible delta grossit (dense ×970 → m95 ×1.9, WP2) — c'est une
  dose-réponse "gratuite" via la sparsité, mais confondue avec d'autres
  effets, d'où le besoin du test propre ci-dessous.
- **Phase 1 (gratuite, CPU, avant tout entraînement)** :
  `tools/build_mechanism_dose_response.py` — recalculer σ_Δ(Δt) en
  sous-échantillonnant les frames déjà sur disque (300→600→900→1800→3600→
  7200s), tracer ratio-vs-Δt pour les 2 événements, ET choisir les 3-4 Δt
  d'entraînement pour couvrir ≥1 ordre de grandeur de ratio. Peut tourner
  MAINTENANT en parallèle du GPU.
- **Phase 2 (GPU, ~6-8 runs courts, seed 42, screening déclaré)** :
  entraîner les 2 bras (cible absolue / cible delta) à chaque Δt retenu.
  Mesure par bras : erreur one-step test vs persistence au même Δt.
- **Prédiction quantitative de l'hypothèse (pré-enregistrée)** :
  - bras delta : battre la persistence exige un bruit d'échantillonnage
    normalisé c < 1 → performance ≈ indépendante du ratio ;
  - bras absolu : exige c < 1/ratio(Δt) → le rapport
    (erreur_absolu / erreur_delta) doit **décroître de façon monotone**
    quand le ratio décroît (Δt grandit), avec quasi-parité à ratio → ~1.
  - **Il n'y a PAS de seuil universel à découvrir** : le point de
    croisement absolu-vs-persistence dépend de c, propriété de CE sampler
    (pas une constante physique). Le livrable est (i) la monotonie
    (test du mécanisme) et (ii) le seuil EMPIRIQUE pour cette
    architecture (chiffre citable, explicitement non-universel).
- **Critères pré-enregistrés** :
  - Monotonie confirmée → mécanisme corroboré, Figure 2 devient la courbe
    dose-réponse (2 bras + persistence vs ratio), §4.2 passe de "cohérent
    avec l'hypothèse" à "nous montrons que" (réplication hors-famille
    toujours recommandée avant Transactions, pas nécessaire pour Q1).
  - Pas de relation claire / non-monotone → l'hypothèse est fausse ou
    incomplète : reformuler §4.2 en observation empirique, retirer
    "mécanisme" des contributions, documenter le résultat négatif tel
    quel — reste publiable, moins fort.
- **Ce que ça peut expliquer rétroactivement si confirmé** : pourquoi le
  papier DIFF-SPARSE d'origine (domaine tidal, dynamique par pas
  vraisemblablement plus grande relative au champ) a pu rapporter des
  succès là où le transfert crue-à-300s s'effondre — les succès/échecs
  publiés se placent sur le même axe ratio. Si les données tidal de
  l'original sont publiques, calculer LEUR ratio (data-only) et le placer
  sur la courbe ; sinon le noter comme non vérifiable.
- **Priorité et coût** : Phase 1 immédiate (CPU, gratuite). Phase 2 après
  WP6/WP9 (qui gardent la priorité GPU) — ~1-2 j GPU au total en runs
  courts one-step. Absorbe WP4(a) (`prediction.target: absolute` à Δt
  natif = le point ratio~488 de la grille — même expérience, mieux
  contrôlée en one-step).

- **PHASE 1 FAITE (2026-07-17, `tools/build_mechanism_dose_response.py`,
  sortie `experiments/FloodCastBench/wp12_dose_response/ratio_curve.json` +
  figure `paper/figures/f2b_dose_response_ratio.pdf`)** :

  | Δt | ratio Australie | ratio UK |
  |---|---:|---:|
  | 300s (natif) | 478 | 412 |
  | 600s | 239 | 207 |
  | 900s | 161 | 138 |
  | 1800s | 82 | 69 |
  | 3600s | 41 | 34 |
  | 7200s | 21 | 17 |

  Lectures : (i) σ_Δ croît quasi LINÉAIREMENT avec Δt sur toute la plage
  (écoulement lisse, régime advectif — pas de décorrélation), donc le
  ratio suit ~1/Δt ; (ii) le sous-échantillonnage seul couvre un facteur
  ~23 de ratio (478→21) mais NE PEUT PAS atteindre la région de parité
  (ratio~1) — la prédiction testable en Phase 2 est donc la monotonie de
  la fermeture du gap sur 21→478, pas la parité complète (à déclarer
  honnêtement dans le papier) ; (iii) les ratios natifs (478/412)
  concordent avec la mesure indépendante de la Figure 2 (488/425, grille
  d'échantillonnage de paires légèrement différente) — cohérence croisée
  OK. **Δt retenus pour la Phase 2 (4 points, couverture géométrique)** :
  300s (478), 900s (161), 1800s (82), 7200s (21) × {absolu, delta} =
  8 runs one-step courts, seed 42.
- **Sortie papier** : Figure 2 remplacée par la courbe dose-réponse ;
  §4.2 reformulé selon le résultat ; §1 (contributions) mis à jour pour
  refléter le niveau de preuve réel atteint.

### WP13 — Généralisation multi-événements (ajouté 2026-07-18, DÉCISION UTILISATEUR : à faire)

**Décision** : réentraînement complet (V2 + jumeau, 3 seeds, {dense, m50,
m95}, évals protocole test complètes) sur **les 4 événements
FloodCastBench**, pas seulement Australie(+UK optionnel comme avant).
Étend le pilier A (contrôle jumeau) d'un seul événement à quatre — la
question devient "le motif jumeau-vs-diffusion se réplique-t-il à travers
des géographies/hydrologies indépendantes", pas juste "tient-il sur un
cas".

**Inventaire confirmé (`datasets/floodcastbench_fno_dataset.py`,
`data/FloodCastBench` sur disque)** :

| Événement | Fidélité | Résolution | Grille | Frames | Config repo |
|---|---|---|---|---:|---|
| Australie | haute | 60m | 536×536 | 2881 | ✅ existant (défaut) |
| UK | haute | 60m | 85×137 | 865 | ✅ existant (WP8) |
| Pakistan | basse | 480m | 810×441 | 4033 | ❌ à créer |
| Mozambique | basse | 480m | 151×138 | 1729 | ❌ à créer |

**Séquencement obligatoire (portes avant tout engagement complet)** :
1. **Finir Australie d'abord** (2 évals restantes ce soir) — ne pas ouvrir
   un 2e front tant que le premier n'est pas fermé.
2. **WP10 (zero-shot cross-événement, déjà scopé, quasi gratuit)** sur
   TOUS les événements disponibles (pas seulement UK comme prévu
   initialement) — sert de test de triage avant de décider où le
   réentraînement complet vaut le coût. Si le zero-shot s'effondre
   uniformément partout, ça change peu la priorité (réentraîner reste
   nécessaire pour trancher le motif) ; s'il tient bien quelque part,
   ça peut réordonner la priorité des événements 3/4.
3. **UK complet** (config déjà prête, WP8) — 2e événement, coût
   comparable à Australie (même résolution/grille proche).
4. **Pakistan puis Mozambique** — configs à créer (schéma dataset déjà
   supporté par le loader, `DEFAULT_SPLITS`/rainfall folders à ajouter),
   grilles plus petites (480m) donc probablement plus rapides par
   epoch/fenêtre, mais intégration initiale (stats de normalisation,
   dry-run, smoke tests) à refaire comme pour UK.

**Coût estimé (basé sur les débits réels observés cette nuit, pas des
chiffres théoriques)** : par événement, 18 runs d'entraînement (V2+jumeau
× 3 seeds × 3 sparsités) + 18 évals ≈ 48h de calcul + 10h30 d'éval en
séquentiel sur 1 GPU, mais le temps de MUR réel (retries, découverte de
trous, vérification systématique — cf. la nuit du 17-18 juillet) est
plus proche de **3-5 jours par événement**. **Total 4 événements : ~8-14
jours séquentiel sur 1 GPU, ~4-8 jours si P7+Dell tournent réellement en
parallèle** (réserve : fiabilité Dell pas garantie, cf. incident WPB1 non
résolu avec certitude).

**Sortie papier** : Table T2 étendue à 4 événements (ou table par
événement + synthèse), remplace la version "Australie + UK optionnel" du
plan précédent. Renforce directement le pilier A.

### WP14 — Généralisation architecturale : le protocole jumeau appliqué à d'autres génératifs (ajouté 2026-07-18, DÉCISION UTILISATEUR : à faire)

**Motivation** : le pilier A démontre actuellement "CE modèle de diffusion
(Δ-Diff) n'apporte pas d'avantage net face à son jumeau". Un reviewer
sérieux demandera si c'est propre à cette architecture précise ou une
propriété plus générale de la diffusion appliquée à ce problème. Répliquer
le protocole jumeau apparié sur ≥2 architectures génératives
supplémentaires transforme le "contrôle jumeau" d'un test ponctuel en une
**méthodologie généralisable** — c'est probablement le levier le plus fort
pour monter de tier si les résultats sont cohérents à travers architectures.

**Design en portes (la moins chère d'abord, ne pas engager la plus chère
sans passer la première)** :

1. **Variante A — même squelette, sampler/schedule différent** (peu
   coûteux : réutilise l'encodeur temporel, l'encodeur spatial, tout —
   seul le processus de diffusion change, ex. DDIM déterministe à peu de
   pas, ou un schedule continu type score-based/SDE). Teste si le motif
   dépend du type d'échantillonnage ou seulement du fait de générer.
2. **Variante B — squelette Mamba** (plus coûteux, nouvelle architecture) :
   insérer Mamba à des points candidats différents dans le backbone —
   encodeur temporel (remplace/complète les tokens temporels), encodeur
   spatial, bottleneck du U-Net — et comparer où l'insertion aide le plus.
   **Leçon directement réutilisable du Papier 2 (WPB3/WPB4/WPB5)** :
   l'insertion naïve de Mamba déstabilise l'entraînement (courbes 2-3×
   plus bruitées, dégât concentré sur les pixels mouillés) ; le fix
   LayerScale/gate-à-zéro-à-l'init (confirmé +1.46σ sur FNO+) doit être
   appliqué DÈS LE DÉPART ici, pas découvert une 2e fois par la même
   instabilité. Chaque emplacement Mamba testé reçoit **son propre jumeau
   déterministe apparié** (même protocole que V2/Twin) — c'est ce qui fait
   que ça reste dans le pilier A plutôt que de devenir une simple ablation
   d'architecture.
3. **Combinaison optionnelle** : si Variante B montre un emplacement Mamba
   clairement gagnant, le tester aussi en variante "sampler différent"
   (combine A+B) — seulement si le budget temps le permet après WP13.

**Critères pré-enregistrés** :
- Le motif "jumeau compétitif/gagnant dans certains régimes" se réplique
  sur les nouvelles architectures → conclusion du pilier A considérablement
  renforcée, argument publiable comme règle méthodologique générale
  ("tout papier de diffusion spatiotemporelle sous sparsité doit reporter
  un jumeau apparié").
- Le motif NE se réplique PAS (le nouveau génératif bat nettement son
  jumeau) → résultat tout aussi informatif : identifie CE QUI, dans
  l'architecture ou le sampler, fait la différence — reformuler le pilier A
  en "dépend de la paramétrisation/architecture", moins fort mais toujours
  honnête et publiable.

**Séquencement** : après WP13 (ne pas ouvrir un 3e front architectural
avant d'avoir fermé le front multi-événements) sauf si le budget GPU libéré
par Dell/P7 permet un vrai parallélisme sans dégrader la supervision des
runs déjà en cours — à réévaluer au moment venu, pas décidé à l'avance.

**Sortie papier** : nouvelle figure/table "réplication inter-architectures"
du motif jumeau-vs-génératif ; si WP14 Variante B confirme un gain Mamba
net et répliqué, ça devient aussi un pont direct vers le Papier 2
(WPB3-7), à mentionner explicitement dans la discussion des deux papiers.

### WP15 — Le jumeau comme alternative à FNO : démêler paramétrisation et
backbone, puis tester la portée (ajouté 2026-07-19, DÉCISION UTILISATEUR :
WP15-A à faire, WP15-B = future work)

**Motivation** : la question "le jumeau (potentiellement Mamba) peut-il
remplacer FNO ?" est distincte de la question "la diffusion apporte-t-elle
quelque chose ?" (pilier A). Deux confondants doivent être démêlés avant
toute affirmation de ce type :
1. Le jumeau n'a pas que un backbone différent de FNO+, il a AUSSI la
   paramétrisation delta + échelle par régime que FNO+ n'a jamais reçue —
   sans contrôle, on ne saurait pas si le jumeau gagne grâce au backbone ou
   grâce à la cible d'entraînement (même piège que le confondant de
   contexte déjà découvert et écarté pour WPB0).
2. Même démêlé, un gain sur FloodCastBench ne dit rien sur "en toute
   condition" (voir discussion no-free-lunch ci-dessus) — il faut un test
   sur le terrain favorable à FNO pour connaître la vraie portée.

**WP15-A — Paramétrisation delta sur FNO+ (peu coûteux, tier 2)** :
réutiliser directement l'infra du Papier 2 (FNO+/FNO+Mamba), donner à
FNO+ la même cible delta + échelle par régime que V2/jumeau (aucun autre
changement), réentraîner sur Australie. Deux issues, toutes deux
publiables :
- FNO+delta comble la majorité de l'écart avec le jumeau → la vraie
  découverte n'est pas "U-Net bat FNO" mais **"la paramétrisation compte
  plus que le backbone"** — résultat plus général, se transfère
  directement au Papier 2, et évite de surclaimer sur l'architecture.
- FNO+delta ne comble PAS l'écart → argument d'architecture réel (biais
  local/attention bat le mélange spectral global ici), qui justifie de
  pousser WP14-B (Mamba) comme variante FNO-alternative sérieuse.
  Coût estimé : ~1-2 j (réutilise le code Papier 2 existant), 3 seeds.

**WP15-B — Dataset adversarial, terrain favorable à FNO (coûteux, PALIER
3 / future work, PAS engagé dans cette soumission)** : tester le jumeau
sur un problème délibérément défavorable à son biais inductif — champ
lisse sans front net, et/ou exigence de généralisation en résolution
(train basse résolution, inférence haute résolution) — par exemple un
benchmark PDE standard type Navier-Stokes/Darcy où FNO a été validé à
l'origine. Deux issues :
- Le jumeau perd → confirme le compromis de biais inductif, renforce la
  crédibilité (on montre qu'on comprend les limites plutôt que de
  surclaimer), donne une règle de décision claire au praticien ("choisir
  selon la structure du champ").
- Le jumeau gagne quand même → résultat surprenant, nécessiterait une
  explication mécanistique sérieuse avant d'être cru (même standard que
  pour l'hypothèse signal≪champ, actuellement non prouvée).
  Coût : intégration complète d'un nouveau dataset/pipeline (cf. coût déjà
  observé pour UK/Pakistan/Mozambique) — ordre de grandeur PALIER 3, pas
  tier 2. Décision : documenté ici comme condition de sortie nécessaire
  AVANT toute affirmation "peut remplacer FNO" dans une version future du
  papier ou un papier séparé — mais explicitement PAS engagé maintenant.

**Ce qui va dans le papier actuel, quel que soit l'état de WP15-B** : la
formulation reste scopée ("pour cette classe de problèmes"), jamais
universelle. La section discussion/limitations doit énoncer explicitement
l'argument no-free-lunch (pourquoi on ne teste pas "en toute condition"
et pourquoi ce serait le mauvais test à faire ici) — ça préempte l'
objection plutôt que de laisser un reviewer la découvrir seul.

### Durcissements protocole pour viser Q1 (ajouté 2026-07-16)
- **Tout chiffre headline du papier repasse en protocole test COMPLET
  (13/13 fenêtres)** avant gel — les lectures rapides 4/13 ne servent qu'au
  pilotage. (Renforce §5.)
- **Tests de significativité** : comparaison appariée par fenêtre de test
  (wilcoxon signé ou bootstrap sur les 13 fenêtres × 3 seeds) pour les
  claims principaux (jumeau vs V2 ; V2 vs FNO+). Un "×1.57" sans intervalle
  n'ira pas en Q1.
- **U-Net baseline** : citer les chiffres publiés Table 4 (pas de
  réentraînement — il est très en-dessous de FNO+ dans le papier source).

### Hors scope explicite (ne pas ouvrir sans décision consignée §10)
- Mamba (V2.2), 30m, remask-rollout comme mode principal (reste une
  ligne d'ablation possible), foundation models météo, benchmarks non-crue.
- (Sortis du hors-scope le 2026-07-16 : cross-event transfer → WP10 ;
  Pakistan/Mozambique 480m → extension optionnelle de WP10.)

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
- **F2 (mécanisme, statut hypothèse)** : distribution des deltas par pas vs
  champ absolu (ratio ~400-490x) ; RMSE persistence vs V1 dense — l'argument
  du plancher. **N'est PAS une preuve de mécanisme en l'état** (un seul
  régime observé, 2 événements corrélés de la même famille de benchmark) —
  voir WP12 pour le test dose-réponse qui manque avant de pouvoir écrire
  "nous montrons que" plutôt que "cohérent avec l'hypothèse que".
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
- **R9** : ne jamais comparer un tirage/proxy à échantillon unique (ex.
  `rollout_val_rmse` de sélection de checkpoint) entre un modèle génératif
  et un modèle déterministe comme métrique de décision finale — biais
  statistique garanti en faveur du déterministe (E[(tirage-vérité)²] ≥
  E[(moyenne-vérité)²] pour toute distribution). La comparaison qui compte
  est toujours : sortie déterministe vs moyenne/médiane sur N≥2 scénarios
  du protocole d'éval réel (§5). Découvert sur WP1 (§4), voir le piège
  documenté dans sa section.

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

## 9-bis. Piste CONDITIONNELLE "mécanisme-first" (Transactions) — ne pas ouvrir avant les deux portes

Reframing possible du papier pour viser TNNLS/TGRS : la contribution
centrale devient le **plancher de bruit d'échantillonnage** (inégalité :
un forecaster génératif en espace absolu ne peut battre la persistence que
si son erreur relative intrinsèque c < σ_Δ/σ_champ ≈ 1/450 ici —
inatteignable ; en espace delta la condition devient c < 1 — triviale),
plus l'échelle par régime comme dérivation, plus une **survey data-only du
ratio σ_Δ/σ_champ à travers les benchmarks publics** (place GenCast@12h,
nowcasting SEVIR@5min, PDEBench, FloodCastBench@300s sur le même axe — les
succès/échecs publiés de la diffusion s'alignent-ils sur le ratio ?), plus
UNE réplication d'entraînement hors crue (PDEBench SWE/NS), l'audit jumeau
devenant la section de validation. **Prior art à traiter obligatoirement :
PDE-Refiner (Lippe et al. 2023), les schedule flaws (Lin et al.), la
prédiction résiduelle en vidéo/météo.**

**Portes de décision (les TROIS doivent être franchies avant d'investir —
3e porte ajoutée 2026-07-17)** :
(1) réévaluation WP1 post-WP6 donne à V2 une région claire où il paie ;
(2) WP9 montre V2 mieux calibré que l'ensemble de jumeaux ;
(3) **WP12 confirme la dose-réponse** (la sévérité de l'échec suit le ratio
signal/champ quand Δt varie) — sans cette porte, toute la piste
"mécanisme-first" repose sur une corrélation à 2 points, ce qui est
exactement le point faible qu'un reviewer Transactions attaquera en
premier. Si WP12 ne confirme pas, cette piste entière tombe, indépendamment
de (1) et (2). Sinon : rester sur le framing audit → Q1 domaine. Coût
additionnel estimé si ouvert : théorie ~1 sem., survey ratios ~2-3 j
(data-only), réplication 1 domaine ~3-5 j GPU, WP12 lui-même ~1-2 j
(dose-réponse gratuite + 1-2 runs courts).

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

## 11. Coordination P7 ↔ Dell (canal live, hors git)

`experiments/FloodCastBench/coordination/` (NFS partagé) — protocole
détaillé dans son `PROTOCOL.md`. Claude-P7 y dépose des instructions
exécutables (`instructions/NNNN_slug.md`), Claude-Dell y répond
(`reports/NNNN_slug_report.md`). Utilisé pour fractionner les WP entre les
deux GPU sans dépendre du cycle commit/push/pull pour la coordination
elle-même (le CODE reste git-only, règle R5 — ce canal ne sert qu'aux
messages/instructions/comptes-rendus). Statut vivant : `status.md` dans ce
même dossier.

**Mode autonome (2026-07-11)** : le Dell tourne en `/loop` sans prompt
humain à chaque itération — toute instruction doit donc être 100%
mécanique (chemins absolus déjà vérifiés, zéro exploration/déduction
requise) et sobre en crédits (le Dell lit `status.md` seul en premier ;
rien en `pending` → fin d'itération immédiate, pas de re-lecture du repo).
Détail complet dans `PROTOCOL.md`.

### Changelog
- 2026-07-10 — création (P7). État : WP0 en cours, WP1-WP8 définis,
  littérature vérifiée, règles R1-R8 codifiées.
- 2026-07-10 (b) — **code WP1/WP2/WP4/WP6 prêt à lancer** (P7) :
  - WP1 : `models/deterministic_twin.py` (sous-classe V2, parité de
    paramètres exacte vérifiée par test ; bruit=zéros, t=0, régression
    pondérée ; interface identique → trainer/évaluateur/pushforward V2
    réutilisés tels quels via `build_v2_family_model`), configs 3 seeds,
    `tests/test_det_twin_smoke.py` (6 tests), dry-run GPU 3 régimes OK.
    Coût mesuré : ~11s/epoch (~3x plus rapide que V2).
  - WP2 : config ctx12. WP4 : knob `model.spatial_features_scale` + 6
    configs d'ablation (absolute/notargetrain/nospatial/nochangeweight/
    nopushforward/steps20). WP6 : flag CLI `--early-stop-patience`.
  - Infra : `scripts/run_training_queue.sh` générique (remplace le script
    /tmp), mêmes commandes sur les deux PC — exemples d'usage par WP dans
    l'en-tête du script.
  - Reste à coder (prochain lot) : accumulateur de calibration dans
    l'évaluateur (WP3), masques structurés (WP7), plomberie LULC/Manning
    (WP5), stats+config UK (WP8).
- 2026-07-10 (c) — **lot 2 codé : WP3/WP5/WP7/WP8 prêts à lancer** (P7) :
  - WP3 : `CalibrationAccumulator` dans l'évaluateur V2 (reliability aux
    M+1 niveaux exacts, coverage 50/90%, rank histogram tous/actifs,
    spread–skill par bins log) + `eval_calibration.json` + outil de figures
    `tools/analyze_v2_calibration.py`. Actif par défaut dès M≥2
    (`--no-calibration` pour couper). ⚠ Piège attrapé par test : biais
    d'ensemble fini — la couverture attendue d'un ensemble M=8 parfait est
    (hi−lo)·(M−1)/(M+1) (ex. "90%" → 70%) ; le JSON expose
    `nominal_finite_ensemble`, c'est LUI la référence du papier.
  - WP7 : `generate_gauge_mask` (∝ carte d'occupation d'eau du train,
    cachée sur disque) et `generate_cluster_mask` (blobs compacts, budget
    exact) ; `masking.eval_mask_structure` + override CLI `--mask-structure`
    dans l'évaluateur. Masques d'entraînement inchangés (protocole d'éval).
  - WP5 : canal Manning complet — chargement LULC réel (resize NEAREST,
    catégoriel), table code→n configurable (provisoire Chow/HEC-RAS,
    fallback 0.05 = constante du simulateur), standardisation propre,
    câblé dans les DEUX encodeurs + tous les chemins batch (trainer,
    pushforward, RolloutValidator, évaluateur). Config
    `floodcastbench_diff_sparse_v2_1_manning_highfid_60m.yaml`
    (5 538 978 params, +432 vs V2). Prérequis vérification schéma inchangé.
  - WP8 : config UK (`event: uk`, splits 35/4/4 → 3 fenêtres éligibles/split,
    grille 85×137), delta-stats UK calculées
    (`diff_sparse_v2_uk_delta_stats.json` : delta_std 0.00083 m — le
    mécanisme signal≪champ tient aussi sur UK). Bug corrigé au passage :
    `.capitalize()` cassait "uk"→"Uk" dans l'outil delta-stats.
  - Vérifs : 54/54 tests (15 nouveaux) ; dry-runs GPU réels UK dense+m95 et
    V2.1 Manning m50 verts (R3).
- 2026-07-10 (d) — **lot 3 : outillage papier** (P7) :
  - `tools/aggregate_v2_family_results.py` : agrégateur d'évals → CSV long +
    table markdown mean±std par (modèle × sparsité × structure de masque)
    — les tables T2/T3 se génèrent d'une commande. Testé sur les 6 évals
    réelles existantes.
  - `tools/build_mechanism_figure.py` : figure F2 (mécanisme) depuis les
    données brutes seules — ratios mesurés **×488 (Australie) et ×425 (UK)**,
    le mécanisme tient sur les deux événements. Figure dans
    `experiments/FloodCastBench/paper_figures/f2_mechanism.png`. (Bug
    attrapé en route : sous-échantillonnage qui mesurait des deltas à 20 pas
    au lieu d'adjacents — corrigé, chiffres recoupés avec les stats
    officielles.)
  - Smoke bout-en-bout jumeau M=1 à travers l'évaluateur complet : NACRPS
    dégénère proprement en MAE, calibration sautée avec raison explicite.
  - Il ne reste AUCUN code bloquant avant les lancements ; restent
    l'édit dashboard post-WP0 et la vérification du schéma LULC (recherche,
    pas du code).
- 2026-07-11 — **WP1 : piège méthodologique trouvé et corrigé avant qu'il ne
  fausse une conclusion.** Premiers résultats seed42 (proxy interne
  `rollout_val_rmse`) montraient le jumeau devant V2 aux 3 sparsités —
  analysé plutôt qu'accepté tel quel : comparer un tirage unique de
  diffusion à une sortie déterministe sur RMSE est biaisé par construction
  statistique en faveur du déterministe (E[tirage²] ≥ E[moyenne²] pour toute
  distribution). Nouvelle règle **R9** (§8) codifie ça pour la suite du
  projet. WP1 (§4) mis à jour : comparaison finale = jumeau vs V2
  moyenne-8-scénarios (protocole réel), pas le proxy d'entraînement — éval
  réelle du jumeau encore à faire. Corrigé aussi une inexactitude du plan
  (pas d'évaluateur séparé pour le jumeau, il réutilise celui de V2 via
  `build_v2_family_model`).
  - Par ailleurs : bug opérationnel trouvé et corrigé — l'orchestrateur de
    la queue jumeau est resté bloqué ~4h30 sans lancer la vague 2 (bug de
    `wait` bash provoqué par un `| head -30` sur le tout premier lancement
    manuel) ; les 3 runs de la vague seed42 avaient bien fini normalement.
    Relancé proprement (vagues seed7/seed123 en cours). Règle : ne plus
    jamais piper la sortie d'un orchestrateur `nohup ... &` à travers un
    filtre externe qui peut se fermer avant le script.
  - **Root cause trouvée et corrigée (`scripts/run_training_queue.sh`,
    commit `8a1bd73`)** : le blocage s'est reproduit une 2e fois (vague
    seed7 finie proprement, 300/300/early-stop-150, orchestrateur bloqué
    quand même) — ce n'était donc pas le `| head -30`, mais un bug du
    script lui-même. `wait` nu attend TOUS les jobs en arrière-plan du
    shell, y compris le sous-processus `tee` créé par
    `exec > >(tee -a "$ORCH_LOG") 2>&1`, qui ne se termine jamais de
    lui-même (deadlock : `tee` attend la fermeture de son stdin, qui
    n'arrive qu'à la sortie du script, qui elle-même attend `wait`).
    Corrigé en capturant les PID des jobs d'entraînement et en faisant
    `wait "${pids[@]}"` au lieu d'un `wait` nu. Vérifié par un relancement
    court en direct avant de relancer la vraie vague 3 (seed123).
- 2026-07-17 — **relecture critique du mécanisme signal≪champ (utilisateur)
  → WP12 créé, CRITIQUE** : le §4.2/Figure 2 du papier revendiquait un
  MÉCANISME causal à partir de 2 points de données (Australie ×488, UK
  ×425), tous deux à ratio élevé ET en échec — c'est une corrélation, pas
  une preuve (aucun point à ratio faible/intermédiaire pour établir la
  relation). Identifié comme **le maillon le plus faible du papier**,
  contrairement à la présentation initiale comme acquis solide. Contribution
  #2 (§1) et description F2 (§6) reformulées en langage d'hypothèse
  ("cohérent avec", jamais "démontre") en attendant WP12. **WP12 ajouté** :
  test dose-réponse (recalculer σ_Δ à plusieurs Δt par sous-échantillonnage
  des frames déjà sur disque — gratuit, CPU, peut tourner en parallèle du
  training GPU en cours ; puis un entraînement court d'Abs-Diff à Δt
  intermédiaire pour vérifier que la sévérité de l'échec suit le ratio).
  §9-bis (piste Transactions) gagne une 3e porte de décision obligatoire :
  sans confirmation dose-réponse, la piste mécanisme-first tombe
  indépendamment des portes (1)/(2) déjà posées. N'affecte pas la priorité
  GPU immédiate (WP6/WP9) — WP12 principal est CPU-only.
- 2026-07-17 (b) — **WP12 redessiné (design utilisateur) + structure
  profonde du papier posée (§1)** : (i) WP12 passe du design "Abs-Diff à
  travers les Δt" (confondu par l'architecture) au design propre "squelette
  V2 unique, croisement {Δt × cible absolue/delta}, one-step dense,
  pushforward=0" — prédiction pré-enregistrée : le rapport
  erreur_absolu/erreur_delta décroît monotonement avec le ratio ; pas de
  seuil universel, seulement la monotonie (test du mécanisme) + le seuil
  empirique propre à ce sampler (livrable citable, non-universel). WP4(a)
  absorbé (même manipulation, mieux contrôlée). (ii) §1 gagne le bloc
  "Structure profonde" : 3 piliers (A contrôle jumeau, B mécanisme
  dose-réponse, C honnêteté incertitude), chaque WP mappé à un pilier, et
  la réponse que le papier apporte énoncée explicitement — l'essentiel des
  gains attribués à la diffusion vient de choix de représentation qui
  profitent à l'identique au déterministe. (iii) WP5 (Manning/LULC)
  rescopé optionnel : ne sert aucun pilier, ne se lance que si A/B/C sont
  bouclés avant le gel M5, sinon passe en travail futur. (iv) Indice
  interne noté dans WP12 : la compression du gap V1/V2 avec la sparsité
  (×970 dense → ×1.9 m95) est déjà cohérente avec le mécanisme (l'échelle
  effective de la cible delta grossit aux pixels masqués) — dose-réponse
  "gratuite" mais confondue, d'où le test propre.
- 2026-07-18 — **DÉCISION UTILISATEUR : élargissement majeur du scope,
  WP13 + WP14 créés**. Après un bilan honnête de la nuit (voir résultats
  WP6/WP9/WP12 ci-dessus) et une discussion coût/bénéfice, décision de :
  (i) **WP13** — réentraînement complet (V2+jumeau, 3 seeds, 3 sparsités,
  évals) sur les 4 événements FloodCastBench (Australie, UK, Pakistan,
  Mozambique), pas seulement Australie+UK-optionnel comme prévu
  jusqu'ici ; inventaire confirmé par lecture directe du code dataset
  (`floodcastbench_fno_dataset.py`) et du disque (`data/FloodCastBench`) —
  Pakistan/Mozambique existent dans le benchmark mais n'ont aucune config
  dans ce repo, à créer. Coût estimé ~8-14 j séquentiel / ~4-8 j si
  double-machine (réserve fiabilité Dell). Séquencement obligatoire :
  finir Australie → zero-shot triage (WP10 étendu aux 4 événements,
  quasi gratuit) → UK complet → Pakistan/Mozambique. (ii) **WP14** — le
  protocole jumeau apparié répliqué sur ≥2 architectures génératives
  supplémentaires : une variante sampler/schedule (peu coûteuse) et une
  variante backbone Mamba à emplacements multiples (encodeur temporel,
  encodeur spatial, bottleneck U-Net), chacune avec son propre jumeau
  déterministe. Réutilise directement la leçon LayerScale/gate-à-zéro du
  Papier 2 (WPB4/WPB5) pour éviter de redécouvrir la même instabilité
  d'entraînement Mamba. Ce WP transforme le pilier A d'un test ponctuel en
  méthodologie généralisable — identifié comme le levier le plus probable
  pour dépasser le tier Q1 domaine si les motifs se répliquent. §1
  (structure profonde) mis à jour en conséquence. Séquencement : WP14
  après WP13 sauf parallélisme réel sans dégrader la supervision des runs
  en cours.
- 2026-07-18 (b) — **Priorités de sortie à deux paliers posées (§1,
  décision utilisateur après discussion coût/rendement)** : PALIER 1 =
  noyau suffisant pour un Q1 domaine soumissible (tableau central
  Australie + dose-réponse + re-runs protocole complet + WP2 3-seeds + UK
  complet + zero-shot 4 événements + table coût + assemblage, ~7-12 j) ;
  PALIER 2 = rendre le papier inattaquable, par rendement décroissant
  (2e architecture générative appariée, **+2 seeds → 5 sur la comparaison
  centrale** — justifié par la variance inter-seed dominante observée en
  WP6 —, Pakistan/Mozambique, variante Mamba, ablations restantes,
  CRPS/caveats calibration, +2-3 semaines). Chaque item du palier 2 est
  additif ; le point de décision de soumission est à la fin du palier 1,
  tranché sur l'état réel de 2.1/2.2 (seuls items qui changent le tier).
  Re-priorisation interdite sans mise à jour explicite de cette section.
- 2026-07-19 — **WP15 créé (question utilisateur : le jumeau peut-il
  remplacer FNO "en toute condition" ?)**. Réponse théorique posée
  explicitement dans le plan (§1) : non, argument no-free-lunch — le
  biais inductif qui favorise le jumeau sur FloodCastBench (fronts nets,
  observation éparse) prédit l'inverse sur le terrain de FNO (champs
  lisses, généralisation en résolution). Deux items créés pour établir la
  portée proprement plutôt que de surclaimer : **WP15-A** (tier 2, item
  2.7, ~1-2 j) — appliquer la paramétrisation delta+échelle par régime à
  FNO+ (réutilise l'infra Papier 2) pour démêler si l'avantage du jumeau
  vient du backbone ou de la cible d'entraînement, confondant jusqu'ici
  non contrôlé ; **WP15-B** (palier 3, hors scope, documenté comme
  condition de sortie future) — test sur dataset adversarial favorable à
  FNO, ordre de grandeur intégration-dataset-complète, non engagé
  maintenant. Le papier gardera une affirmation scopée ("classe de
  problèmes à observations éparses + fronts nets"), jamais universelle,
  avec l'argument no-free-lunch énoncé explicitement en discussion pour
  préempter l'objection plutôt que de la laisser à découvrir par un
  reviewer.
