# Logistics — machines, queues GPU, scripts en vol (créé 2026-07-23, audit action 3)

Vérité pour « qu'est-ce qui tourne / dans quel ordre ». Voir
`paper_master_plan.md` (stratégie) et `results_log.md` (chiffres).

## Machine courante — DGX Spark (Acer Veriton GN100)
- NVIDIA GB10 Grace-Blackwell, ARM64 (aarch64), 119 Go mémoire unifiée,
  sm_121, driver 580.95.05, CUDA 13.0.
- Environnement : VSCodium Flatpak-sandboxé. `nvidia-smi` via
  `flatpak-spawn --host`. Venv : `$HOME/Desktop/Wissam/venvs/floodcast-mamba`.
- Workspace réel : `/home/altos/Desktop/Wissam/Ubuntu-Research-Wissam_2026-07-21/home/wissam/utem-workspace`.
  Les configs hardcodent `/home/wissam/utem-workspace/...` (correct sur
  P7/Dell) → overrides CLI (`--dataset-root` etc.) pour le trainer,
  `scripts/make_spark_local_config.py` pour l'évaluateur (pas d'override CLC).
- LaTeX : `tectonic` statique aarch64 dans `$HOME/Desktop/Wissam/tools/tectonic/`.
- Rapatriement prévu à terme sur P7 via SSH (intention utilisateur).

## Utilisateur — fuseau
Malaisie (UTC+8). **Toujours convertir les heures en heure Malaisie dans les
messages** (UTC + 8h), jamais d'UTC brut.

## Pilotage autonome (instauré 2026-07-24, instruction utilisateur explicite)
Supervision double active tant que la chaîne GPU n'est pas terminée :
- **Monitor persistant** (tail -F des 3 logs orchestrateurs, alerte sur
  `Traceback|CUDA error|OOM|NaN|Inf|Killed|===|ALL_*_DONE`) — réactif,
  immédiat.
- **ScheduleWakeup** toutes les 10 min — ré-analyse globale périodique,
  décide et enchaîne la suite. Doit être réarmé À CHAQUE tour sans
  exception (règle qui a cassé la cadence 2× avant d'être corrigée).

**Frontière décisionnelle (instruction utilisateur explicite) :**
- **Autonome, sans validation** : relancer un job interrompu depuis un
  checkpoint valide, corriger un chemin/commande évidemment cassé,
  interrompre un job manifestement défaillant, lancer une éval déjà
  prévue au protocole, enchaîner l'étape suivante déjà décidée.
- **JAMAIS seul, arrêt + demande d'intervention** : suppression de
  données/checkpoints, modification profonde du modèle, changement
  majeur de protocole (nb de seeds, régimes, cibles), abandon d'une
  expérience importante, écrasement de résultats, opération à coût/
  conséquences incertains.

Toute décision prise en autonomie (relance, correction) doit être
consignée ici ou dans `results_log.md` selon sa nature.

### Journal de pilotage autonome (résultat → interprétation → décision → expérience)

**2026-07-24, 15h55** — *Observé* : les 6 leviers d'ablation §6.8 sont
finis (ALL_OVERNIGHT_QUEUE_DONE, 07h50 UTC). En relisant les
`\blocked{}` restants, 2 leviers (pushforward, steps20) + 2 effets CSI
(nospatial, notargetrain) nécessitent une confirmation 3-seeds —
**aucun orchestrateur actif ne couvre ça**. *Interprétation* : nouveau
trou plan/exécution, même nature que ctx12. *Décision* : PAS lancé
immédiatement (coût : 4 leviers × 2 seeds × ~1h train + ~3h éval ≈
32h, need réévaluer priorité vs WP16 une fois ctx12/WP16 avancés) —
noté ici pour action future plutôt que lancé sous pression. En
attendant, gain zéro-coût trouvé et exécuté : intégration figure f6
(calibration m95 random déjà calculée, jamais mise dans la figure).

**2026-07-24, 01h15** — *Observé* : audit du plan a confirmé 5
`\blocked{}` restants dans le papier, dont 2 (context ablation 3 seeds,
ligne 202+522) ne sont couverts par AUCUN orchestrateur actif (stage3,
twin_m50_retrain, wp16 ne les mentionnent pas). *Interprétation* :
trou réel entre le plan et l'exécution, pas juste une tâche en attente
— exactement le risque signalé par l'utilisateur (« les orchestrateurs
sont des moyens d'exécution, pas la limite du mandat »). Checkpoint
seed42 ctx12 (3 régimes) déjà entraîné le 15-07 mais ZÉRO artefact
d'éval sur disque (audit du 23-07). *Décision* : construire
`run_ctx12_completion.sh` (seed7+123 nouveaux, seed42 réutilisé sans
réentraînement — vérifié via `config.yaml` de chaque dossier, pas
d'ambiguïté comme le cas UK), éval screening 4 fenêtres (même logique
coût que WP16). Re-gaté **avant** WP16 (moins cher, ferme 2 items
bloquants directement) plutôt qu'en parallèle (contention GPU) ou après
(WP16 = plusieurs jours, retarderait inutilement un gain moins cher).
*Expérience lancée* : `ctx12_completion` (attend twin_m50_retrain),
WP16 re-gaté pour attendre ctx12_completion en plus. *Résultat
attendu* : confirmation à 3 seeds que l'écart V2@12 vs V1 reste
massif (déjà 970×/3.9×/1.9× en screening single-seed) — effet
attendu trop large pour qu'un screening à 3 seeds ne suffise pas à
le confirmer.

**2026-07-24, 01h10** — *Observé* : `nacrps` (score CRPS normalisé)
déjà présent dans tous les `eval_summary.json` existants, jamais
exploité. *Interprétation* : le `\blocked{}` "CRPS manquant" du papier
(ligne 824) était en partie faux — la donnée existait, seule
l'exploitation manquait. *Décision* : extraction immédiate (zéro coût
GPU), insertion dans le papier (§6.7, nouveau paragraphe Sharpness),
au lieu de laisser un item bloqué qui n'en était pas vraiment un.
*Résultat* : m50 nacrps 0.1599±0.0257, m95 0.3488±0.0077 (3 seeds
chacun) — cohérent avec la sous-couverture déjà documentée (ensemble
trop étroit ET pas particulièrement resserré).

## Coût des évals = nombre de tuiles × scénarios × étapes (vérifié 2026-07-23)
Le coût d'une éval Δ-Diff est dominé par le **nombre de tuiles** de la grille
(patch 64, stride 32), PAS par un bug ni par la machine :
| Événement | Grille | Tuiles | Vitesse éval 8-scén |
|---|---|---|---|
| UK | 137×85 | 8 | ~1,5 min/fenêtre |
| Mozambique | 138×151 | 16 | ~3 min/fenêtre (extrap.) |
| Australie | 536×536 | 256 | ~44 min/fenêtre |
| Pakistan | 441×810 | 325 | ~55-60 min/fenêtre (extrap.) |
Ratio tuiles Australie/UK = 32× → éval 29× plus lente : mécanique, cohérent.
L'éval m95 de 9h28 (23-07) n'était PAS anormale (256 tuiles × 8 scén × 40 pas
× 13 fenêtres). Corollaire : les évals Twin (ns=1) coûtent 1/8 des évals
Δ-Diff (ns=8) ; le screening WP16 (--max-windows 4 pour Δ-Diff) est le bon
levier sur les grosses grilles. **ENTRAÎNEMENTS : aucune anomalie, Spark plus
rapide que P7 (Twin AUS 35,6 vs 62,6 s/epoch).**

## Leçons de supervision (coûteuses, à ne pas répéter)
1. **Les lignes de log ne prouvent PAS la progression.** L'évaluateur
   `evaluate_floodcastbench_diff_sparse_v2.py` n'écrit ses fichiers qu'à la
   toute fin ; les lignes « window N/13 done » peuvent s'afficher groupées.
   Vérifier la **vérité disque** (`ls -la --time-style=full-iso <output_dir>`),
   pas le tail du log.
2. **Comparer le rythme réel à l'ETA annoncée** à chaque réveil — signaler
   toute dérive >2× même sans crash (ne pas reprogrammer en silence).
3. **Power draw bas + util « 94% »** = signal d'un souci de batching, pas d'un
   GPU saturé — mais tile-chunk n'a PAS aidé l'éval Δ-Diff (goulot insensible).
   Le vrai levier de coût est `--num-scenarios` (÷8) ou `--max-windows`.
4. `PYTHONUNBUFFERED=1` dans tous les scripts de queue pour un log lisible.

## Queue GPU courante (séquencée, 2026-07-23 ~20:00 heure Malaisie)
Chaîne de wait-gates (chaque script attend le marqueur du précédent) :

```
[EN COURS] Ablations §6.8  (run_overnight_stage3_only.sh)
              │  marqueur: ALL_OVERNIGHT_QUEUE_DONE
              │  abl_absolute éval en cours (4 fen.), puis 5 ablations restantes
              │  (train + éval --max-windows 4 chacune)
              ▼
[ATTENTE]  Twin m50 retrain ×3 seeds  (run_twin_m50_retrain.sh)
              │  marqueur: ALL_TWIN_M50_RETRAIN_DONE
              │  budget epochs=600/patience=120 (= retrains m95) ; ferme
              │  l'asymétrie d'audit (P0-1)
              ▼
[ATTENTE]  WP16 protocole complet  (run_wp16_full_protocol.sh)
              │  marqueur: ALL_WP16_DONE
              │  Pakistan 15 runs + Mozambique 18 runs + cross-région
              │  UK→AUS et MOZ→PAK.
              │  DÉCISION 2026-07-23 : évals Δ-Diff (8 scén.) plafonnées à
              │  --max-windows 4 (SCREENING) — full protocol coûtait ~9j.
              │  Évals Twin (ns=1) restent complètes. Full-protocol Δ-Diff
              │  à re-lancer plus tard uniquement sur les cellules
              │  décisives. Queue ainsi ramenée de ~9j à ~2j.
```

Logs : `logs/FloodCastBench/background_jobs/{overnight_stage3_*, twin_m50_retrain_*, wp16_full_protocol_*}.log`.
Scripts (scratch, non versionnés) : `/tmp/claude-1000/scratchpad/run_*.sh`.

## Historique des queues terminées (2026-07-22→23)
- Pakistan seeds chain (seed7+123 in-domain + Mozambique zero-shot + Δ-Diff
  UK full-event 3 seeds) — DONE 2026-07-22 ~19:42 UTC.
- WP12 twin control 8/8 — DONE.
- Overnight stage 1 (m95 calibration, 9h28 — doublon involontaire d'un
  résultat du 18-07) + stage 2 (cost benchmark) — DONE 2026-07-23 09:47 UTC.

## Incidents notables
- **Doublon d'éval 9h28** (22→23-07) : encadré §6.7 périmé depuis le 18-07,
  relancé sans le savoir. Origine → règle du commit unique instaurée.
- **glibc 2.42 / CUDA 13.0** : conflit `rsqrt` noexcept, résolu par un arbre
  d'includes CUDA patché local (`$HOME/Desktop/Wissam/local-cuda-13.0-patched`).
- **DataLoader persistent_workers** : cache frames perdu entre epochs sans
  `persistent_workers: true` — corrigé (×26 mesuré sur Pakistan).
