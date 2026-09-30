# Faire tourner la pipeline uniquement sur les 400 lignes gold

But : valider un changement de prompt en quelques minutes/heures sur les 400
lignes du gold standard, au lieu de relancer les ~25 790 lignes du corpus
complet à chaque itération.

Deux choses à changer par rapport à un run normal : (1) l'INPUT de step3 (et
donc, en cascade, de tous les steps suivants), et (2) la taille de l'array
SLURM, sans quoi 400 lignes se retrouvent quand même éparpillées sur 9 GPU
en parallèle pour rien.

## 1. Construire le fichier d'entrée à 400 lignes (une seule fois par version du corpus)

```bash
cd /work/FAC/FDCA/IDHEAP/mhinterl/parp/ACCOUNTABILITY_REPO
python scripts/make_gold400_subset.py \
    --step2_input data/processed/step2_clean.parquet \
    --gold_csv    data/output/Accountability_GOLD_sample400.csv \
    --output      data/processed/step2_clean_GOLD400.parquet
```

Le script vérifie que les 400 paires (article_id, keyword) du gold sont bien
retrouvées dans `step2_clean.parquet` et prévient si ce n'est pas le cas
(par exemple si le corpus a été régénéré depuis le tirage du gold). Le
fichier `Accountability_GOLD_sample400.csv` doit être accessible depuis le
cluster -- si ce n'est pas déjà le cas (il vit aujourd'hui sur le NAS de
recherche, un montage différent de `/work`), copiez-le d'abord dans
`data/output/` du dépôt sur le cluster.

Refaire cette étape seulement si `step2_clean.parquet` change (nouveau
scraping, nouvelle version de step2). Sinon le fichier `step2_clean_GOLD400.parquet`
reste valable d'une itération de prompt à l'autre.

## 2. Lancer chaque étape sur ce fichier, avec un seul GPU au lieu de 9

`NUM_TASKS` est maintenant surchargeable par variable d'environnement dans
les 5 scripts `sbatch_step*_array.sh` (avant : `NUM_TASKS=9` en dur). La
taille de l'array (`--array=0-8`) se surcharge, elle, directement en ligne
de commande de `sbatch` (SLURM donne toujours priorité à ce qui est passé en
ligne de commande sur ce qui est écrit dans le script).

Pour un run gold-only, on ne s'appuie **pas** sur l'auto-chaînage habituel
(step3 → step4a → step4b → step5 → step6 automatique) : celui-ci relance
toujours l'étape suivante avec les réglages par défaut (9 tâches), pas avec
les vôtres. On soumet donc chaque étape à la main, en attendant que la
précédente soit terminée, et en récupérant son fichier de sortie fusionné
comme entrée de la suivante :

```bash
# step3
NUM_TASKS=1 sbatch --array=0-0 sbatch_step3_array.sh data/processed/step2_clean_GOLD400.parquet
# -> attendre la fin (squeue -u $USER), noter le job id, ex. 65300111
# -> sortie : data/output/step3_merged_job65300111.parquet

# step4a (entrée = sortie de step3)
NUM_TASKS=1 sbatch --array=0-0 sbatch_step4a_array.sh data/output/step3_merged_job65300111.parquet

# step4b (entrée = sortie de step4a)
NUM_TASKS=1 sbatch --array=0-0 sbatch_step4b_array.sh data/output/step4a_merged_job<ID>.parquet

# step5 (entrée = sortie de step4b)
NUM_TASKS=1 sbatch --array=0-0 sbatch_step5_array.sh data/output/step4b_merged_job<ID>.parquet

# step6 (entrée = sortie de step5)
NUM_TASKS=1 sbatch --array=0-0 sbatch_step6_array.sh data/output/step5_merged_job<ID>.parquet
```

Chaque étape sur 400 lignes avec un seul GPU devrait se terminer en
quelques minutes à quelques dizaines de minutes selon la charge du cluster,
contre plusieurs heures pour le corpus complet sur 9 GPU. Le fichier
`step6_merged_job<ID>.csv` obtenu à la fin se compare au gold standard
exactement comme les runs précédents.

**Important : le run final "officiel" (celui qu'on documente et qui sert de
référence) doit toujours être refait sur les 25 790 lignes complètes** une
fois qu'un changement de prompt est jugé satisfaisant sur les 400 lignes --
le gold standard est un échantillon, pas la population, et certains biais
(rares catégories, longueur d'article, langue) ne se voient qu'à pleine
échelle. Pour ce run complet, revenir simplement à l'usage normal :
`sbatch sbatch_step3_array.sh data/processed/step2_clean.parquet` (sans
`NUM_TASKS=1` ni `--array=0-0`), qui déclenche l'auto-chaînage habituel
jusqu'à step6.
