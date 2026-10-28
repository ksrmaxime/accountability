#!/bin/bash -l
#SBATCH --job-name=check_determinism
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/check_determinism_%j.out
#SBATCH --error=logs/check_determinism_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

set -euo pipefail

# =============================================================================
# 2026-10-28 -- diagnostic ponctuel, PAS une etape du pipeline.
#
# But : comprendre pourquoi job66621913 et job66666813, sur le MEME GPU
# (A100 GPU-28e0793b-80c9-2fd6-f254-e4f304d0ad55, confirme par leurs logs),
# avec le MEME code et le MEME prompt de calibration (fixe, independant de
# toute donnee -- juste "N/A" + "the subject"), ont quand meme produit des
# p0 differents. Le correctif de determinisme du 2026-10-27 dans
# src/client.py (CUBLAS_WORKSPACE_CONFIG + use_deterministic_algorithms +
# attn_implementation="eager") etait suppose eliminer exactement ca.
#
# Ce script ne lance AUCUNE etape du pipeline (pas de donnees, pas de SLURM
# array) -- juste : charger le modele une fois, verifier ce qui a reellement
# ete applique, et refaire le MEME appel deux fois DANS LE MEME PROCESSUS
# pour voir si le non-determinisme existe deja intra-processus ou seulement
# entre deux soumissions sbatch separees.
# =============================================================================

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/ACCOUNTABILITY_REPO
cd "$WORKDIR"

module purge
module load python/3.12.1
source .venv/bin/activate

mkdir -p logs

echo "=== ENV ==="
echo "HOST=$(hostname)"
echo "DATE=$(date -Is)"
nvidia-smi -L || true
python3 -c "
import torch, transformers
print('torch       ', torch.__version__)
print('transformers', transformers.__version__)
print('cuda        ', torch.version.cuda)
"

echo
echo "=== MODEL LOAD + ATTENTION IMPLEMENTATION CHECK ==="
python3 - <<'PYEOF'
import inspect
import torch

from src.client import TransformersClient, LLMConfig
from src.step4b_logit_prompts import build_calibration_prompt, CATEGORIES

print("deterministic algorithms enabled (global torch flag, BEFORE client init):",
      torch.are_deterministic_algorithms_enabled())

c = TransformersClient(LLMConfig(
    model_path="/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509",
    dtype="bf16",
    trust_remote_code=True,
    backend="transformers",
))

print("deterministic algorithms enabled (AFTER client init):",
      torch.are_deterministic_algorithms_enabled())
print("attn_implementation effectif (model.config):", c.model.config._attn_implementation)

attn_module = c.model.model.layers[0].self_attn
attn_cls = type(attn_module)
print("classe du module d'attention:", attn_cls)
try:
    print("fichier source de cette classe:", inspect.getfile(attn_cls))
except TypeError:
    print("fichier source: introuvable (classe built-in ou compilee)")

print()
print("=== DETERMINISME INTRA-PROCESSUS : meme prompt de calibration, 2 appels separes ===")
print("(prompt fixe, independant de toute donnee -- le meme que celui qui donne des p0")
print(" differents entre job66621913 et job66666813)")

prompt = build_calibration_prompt(CATEGORIES[0], order=1, context_label="SUMMARY OF THE CRITICISM")
r1 = c.logprob_many(system_prompt="", user_prompts=[prompt], candidate_letters=["A", "B"], max_input_tokens=16384)
r2 = c.logprob_many(system_prompt="", user_prompts=[prompt], candidate_letters=["A", "B"], max_input_tokens=16384)
r3 = c.logprob_many(system_prompt="", user_prompts=[prompt, prompt], candidate_letters=["A", "B"], max_input_tokens=16384)

print("appel 1 (seul dans son batch)              :", r1)
print("appel 2 (seul dans son batch, meme process) :", r2)
print("appel 3 (le MEME prompt duplique dans UN SEUL batch de 2):", r3)
print()
print("appel1 == appel2 (meme prompt, 2 appels separes, meme processus):", r1 == r2)
print("dans appel3, les 2 lignes (meme prompt, meme batch) sont identiques:", r3[0] == r3[1])
PYEOF

echo
echo "Termine."
