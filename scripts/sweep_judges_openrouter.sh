#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  Sweep: run agreement on one dataset with multiple OpenRouter judges.
#
#  Usage:
#    bash scripts/sweep_judges_openrouter.sh
#    bash scripts/sweep_judges_openrouter.sh --submit
#    bash scripts/sweep_judges_openrouter.sh --submit --detach
#
#  Notes:
#    - Judge entries are OpenRouter model paths without the provider prefix,
#      e.g. "openai/gpt-5-mini" or "deepseek/deepseek-chat-v3.1".
#    - OpenJury will run these API judges as local bash jobs, not GPU sbatch
#      jobs, even though we still use the --slurm script-generation path.
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."

# Load repo .env so OPENROUTER_API_KEY and any cluster settings are available.
if [ -f ".env" ]; then
    set -a
    source ".env"
    set +a
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY is not set. Export it or add it to .env." >&2
    exit 1
fi

# ── Configuration ────────────────────────────────────────────────
DATASET="comparia"
N_INSTRUCTIONS=25000
SEED=42
LANGUAGE="null"
BALANCE_BY="models"
CRITERIA="default"
MODE="pairwise"
INCLUDE_COMPLETIONS="false"
IGNORE_SCORE_CACHE="true"
SLURM_TIME="16:00:00"
SLURM_OUTPUT_DIR="results/agreement_sweep_comparia_openrouter"
DEFAULT_JUDGE_MAX_TOKENS=4096
JUDGE_TEMPERATURE="0.0"
JUDGE_TOP_P="1.0"

# Judge models.
# Format: "OPENROUTER_MODEL_PATH[:JUDGE_MAX_TOKENS]"
# Example:
#   "openai/gpt-5-mini:4096"
#   "deepseek/deepseek-chat-v3.1:4096"
JUDGES=(
    deepseek/deepseek-v3.2
)

# ── Flags forwarded to openjury-evaluate ─────────────────────────
EXTRA_FLAGS=("$@")

# ── Generate & submit ───────────────────────────────────────────
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

for entry in "${JUDGES[@]}"; do
    IFS=':' read -r MODEL_PATH JUDGE_MAX_TOKENS <<< "$entry"
    JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-$DEFAULT_JUDGE_MAX_TOKENS}"
    MODEL_TAG=$(echo "$MODEL_PATH" | tr '/' '-')
    TAG="${DATASET}-${MODEL_TAG}"

    CFG="$TMPDIR/${TAG}.yaml"
    cat > "$CFG" <<EOF
dataset: ${DATASET}
n_instructions: ${N_INSTRUCTIONS}
seed: ${SEED}
language: ${LANGUAGE}
balance_by: ${BALANCE_BY}
criteria: ${CRITERIA}
output_dir: ${SLURM_OUTPUT_DIR}

generation:
  ignore_score_cache: ${IGNORE_SCORE_CACHE}

judge:
  model: OpenRouter/${MODEL_PATH}
  gpus: 0
  mode: ${MODE}
  max_tokens: ${JUDGE_MAX_TOKENS}
  temperature: ${JUDGE_TEMPERATURE}
  top_p: ${JUDGE_TOP_P}

include_completions: ${INCLUDE_COMPLETIONS}
EOF

    echo "═══════════════════════════════════════════════════"
    echo "  Judge: OpenRouter/${MODEL_PATH}"
    echo "  Tag:   ${TAG}"
    echo "  Config: ${CFG}"
    echo "═══════════════════════════════════════════════════"

    uv run openjury-evaluate agreement \
        --config "$CFG" \
        --slurm \
        --slurm_output_dir "$SLURM_OUTPUT_DIR" \
        --slurm_tag "$TAG" \
        --slurm_time "$SLURM_TIME" \
        "${EXTRA_FLAGS[@]}"

    echo ""
done

echo "Done. All jobs generated."
echo "Results will be in: ${SLURM_OUTPUT_DIR}/"
