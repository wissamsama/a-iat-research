# Results Log — journal canonique des résultats (créé 2026-07-23, audit action 3)

**Règle de gouvernance** (voir aussi l'en-tête de `paper_master_plan.md`) :
- Ce fichier est **append-only** : tout résultat d'expérience terminé y est
  consigné (date, chiffres, provenance disque exacte).
- **Règle du commit unique** : tout résultat qui débloque ou périme un
  encadré `\blocked{}` du papier doit mettre à jour le papier **dans le
  même commit** que l'entrée ici. C'est l'absence de cette règle qui a
  causé le doublon d'éval de 9h28 du 22→23-07 (encadré §6.7 périmé depuis
  le 18-07 sans que personne ne s'en aperçoive).
- Le narratif historique d'avant le 2026-07-23 reste dans
  `paper_master_plan.md` (non déplacé pour préserver l'audit trail) ; ce
  fichier fait foi pour l'état courant.

---

## État canonique au 2026-07-23 (snapshot de l'audit complet)

### In-domain, protocole complet (2 modèles × 3 seeds × 3 régimes), relRMSE

**Australie (60m)** — complet 18/18 :
| Régime | Δ-Diff | Twin | Verdict |
|---|---|---|---|
| dense | 0.001558 ± 0.000836 | 0.000417 ± 0.000039 | Twin ×3.7, signe 3/3 |
| m50 | 0.318425 ± 0.043702 | 0.348852 ± 0.040779 ⚠️ | indécis (signe flippe) |
| m95 | 0.492443 ± 0.013940 | 0.344249 ± 0.006248 | Twin ×1.4, signe 3/3 |

⚠️ Twin m50 : checkpoints budget ORIGINAL (105/150/105 epochs, 10-11/07) —
jamais réentraînés au budget étendu contrairement à m95 (les deux modèles)
et Δ-Diff m50. Asymétrie divulguée dans le papier (Annexe A item 2),
retrain ×3 seeds en queue (`run_twin_m50_retrain.sh`, budget epochs=600/
patience=120 identique aux retrains m95).

**UK (60m)** — complet 18/18, 3 fenêtres de test (petit domaine) :
| Régime | Δ-Diff | Twin | Verdict |
|---|---|---|---|
| dense | 0.006018 ± 0.000481 | 0.005674 ± 0.000340 | Twin ×1.06, 3/3 |
| m50 | 0.174123 ± 0.011544 | 0.156572 ± 0.004881 | Twin ×1.11, 3/3 |
| m95 | 0.223112 ± 0.008981 | 0.194904 ± 0.008246 | Twin ×1.14, 3/3 |

Note : les 6 runs dense UK se sont tous arrêtés à 65 epochs (early-stop,
best epoch ~5) — uniformité à garder à l'œil, comparaison équitable
(même règle des deux côtés) mais chiffres absolus possiblement prudents.

**Pakistan (480m)** — Twin dense 3/3 seulement (WP16 étend au reste) :
Twin dense : 0.000317 ± 0.000172 (per-seed 0.000202/0.000189/0.000560 —
seed123 = échec de convergence documenté, LR effondré à 1.5e-08,
train_loss ~50× au-dessus des autres seeds ; rapporté tel quel).
5 métriques vs FNO+ publié (0.002107) : devant sur les 5 à 3 seeds.

**Mozambique** — aucun in-domain (WP16 en queue).

### Transfert zero-shot (poids gelés, full-event)

| Paire | Modèle | relRMSE | vs FNO+ publié |
|---|---|---|---|
| AUS→UK (42 fen.) | Twin 3 seeds | 0.002874 ± 0.000177 | ×8.6 (FNO+ 0.024771) |
| AUS→UK (42 fen.) | Δ-Diff 3 seeds | 0.005232 ± 0.000820 | ×4.7 |
| PAK→MOZ (85 fen.) | Twin 3 seeds | 0.006249 ± 0.003625 | ×12.6 (FNO+ 0.078633) ; seeds sains ~×21 |
| UK→AUS | — | en queue WP16 | pas de chiffre FNO+ publié |
| MOZ→PAK | — | en queue WP16 (nécessite in-domain MOZ d'abord) | pas de chiffre FNO+ publié |

### WP12 dose-réponse — FINAL
Diffusion : monotone 8/8, pente log-log −1.06, R²=0.98.
Contrôle jumeau : 8/8, pente −0.916, R²=0.987. Mécanisme = échelle de
cible, indépendant du sampler. Croisement delta-vs-persistance ≈ 3910s.

### Coût mesuré (2026-07-23, checkpoints réels, GPU GB10)
Params identiques 5 538 546 (vérifié au chargement). NFE 320 vs 1.
Latence/tuile : 0.792s vs 0.0137s (×57.6). Mémoire pic : 152 vs 92 MB.
`reports/floodcastbench_cost_table.json`.

### Calibration m95 masque aléatoire (Δ-Diff seed42)
relRMSE 0.5084 — existait depuis le 18-07
(`16-07-2026_09-37-43_.../eval_rollout_test_18-07-2026_11-05-40`),
reproduit indépendamment le 23-07 à 4 décimales près (9h28 GPU).
Reste : intégration coverage corrigée dans la figure f6.

**CRPS (nacrps) — FAIT 2026-07-24, zéro coût GPU.** Résultat : déjà
calculé et stocké dans TOUS les eval_summary.json existants (clé
`model.overall.nacrps`), jamais remonté au papier. Interprétation :
pas une expérience manquante, un oubli d'exploitation de données déjà
produites. Décision : extraction immédiate (3 seeds, mr=0.5/0.95,
masques aléatoires) plutôt que relancer quoi que ce soit.
m50 : 0.1599 ± 0.0257 (seeds 0.1533/0.1322/0.1941).
m95 : 0.3488 ± 0.0077 (seeds 0.3385/0.3510/0.3569).
Inséré dans §6.7 du papier (nouveau paragraphe "Sharpness").

### Ablations composants (§6.8) — EN COURS

**abl_absolute — FINI 2026-07-23** (V2 archi complète + cible ABSOLUE au
lieu de delta ; dense, screening 4/13 fenêtres, seed42, 8 scénarios) :
relRMSE 0.309516, NSE 0.868941, r 0.935669, CSI@0.001 0.703782,
CSI@0.01 0.800702. Isole la contribution de la paramétrisation delta
seule (architecture V2 tenue fixe des deux côtés) : Δ-Diff/abl_absolute
= 0.001558/0.309516 ≈ **×199**. Décompose le saut ~3 ordres de grandeur
du papier : la cible delta domine largement (×199), l'architecture V2
contribue peu (abl_absolute nettement meilleur que Abs-Diff/V1 ~560×
pire que persistance). Screening — pas encore 3 seeds/protocole complet.

5 restantes en queue (abl_nochangeweight, nopushforward, nospatial,
notargetrain, steps20), évals à --max-windows 4 (screening assumé).

### ctx12 (ablation contexte) — décision d'audit (action 4)
Chiffres du papier (970×/3.9×/1.9×, étiquetés screening) : provenance =
lecture rapide 4/13 fenêtres seed42 consignée au plan le 16-07, AUCUN
artefact d'éval sur disque. Décision 2026-07-23 : PAS de régénération
séparée (~9-28h GPU pour un claim screening) — la régénération d'artefacts
est fusionnée dans l'item déjà PENDING « context ablation at 3 seeds »
du papier, qui produira des artefacts complets sous protocole intégral
quand il tournera.

---

## Journal (append-only à partir d'ici)

- **2026-07-23** : audit complet (doc+plans+runs). Papier mis à jour avec
  tous les résultats débloqués (commit `81374b5`). Chaîne GPU re-séquencée :
  ablations → twin m50 retrain → WP16.
