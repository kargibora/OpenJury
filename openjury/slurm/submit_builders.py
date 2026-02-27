"""Helpers for building SLURM/login-node submission wrapper scripts."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any

SLURM_WAIT_HELPER = """\
# Helper: poll squeue until a SLURM job finishes, then verify success.
_wait_slurm_job() {
    local job_id=$1 name=$2
    echo "⏳ Waiting for $name (SLURM job $job_id)..."
    while squeue -j "$job_id" -h 2>/dev/null | grep -q .; do
        sleep 30
    done
    local state
    state=$(sacct -j "$job_id" -n -o State --parsable2 | head -1)
    if [[ "$state" != "COMPLETED" ]]; then
        echo "❌ $name failed (state: $state)"
        exit 1
    fi
    echo "✅ $name completed"
}
"""


def build_arena_submit_all(
    pipeline: Any,
    gen_paths: list[Path],
    gen_locals: list[bool],
    path_judge: Path,
    tag_suffix: str,
    results_dir: str,
    *,
    models_cached: list[str] | None = None,
    stage: str = "all",
) -> str:
    """Build ``submit_all.sh`` for arena mode."""
    has_gen = bool(gen_paths)
    cached_names = models_cached or []
    is_analyze_only = stage == "analyze"
    pipeline_title = "OpenJury Arena Analyze Pipeline" if is_analyze_only else "OpenJury Arena Pipeline"
    phase_label = "Arena Analyze" if is_analyze_only else "Arena Judge"
    phase_label_lower = "arena analyze" if is_analyze_only else "arena judge"

    if has_gen:
        gen_model_lines = "\n".join(
            f'echo "  [{i+1}] {m} ({"compute node" if gen_locals[i] else "login node"})"'
            for i, m in enumerate([p.stem.split("_", 2)[-1] for p in gen_paths])
        )
    elif is_analyze_only:
        gen_model_lines = 'echo "  (generation skipped: --stage analyze)"'
    else:
        gen_model_lines = 'echo "  (all completions loaded from cache)"'

    n_gen = len(gen_paths)
    n_cached = len(cached_names)
    if is_analyze_only:
        summary = f"{n_gen} generate job(s) → 1 arena analyze"
    else:
        summary = f"{n_gen} generate job(s)"
    if n_cached:
        summary += f" ({n_cached} cached)"
    if not is_analyze_only:
        summary += " → 1 arena judge"

    preamble = dedent(f"""\
        #!/bin/bash
        # Submit arena pipeline: {summary}
        set -euo pipefail
        mkdir -p {pipeline.logs_dir} {pipeline.work_dir}

        echo "═══════════════════════════════════════════════════════"
        echo "  {pipeline_title}"
        echo "  Dataset:    {pipeline.dataset}"
        echo "  Models:     {len(pipeline.models)} total ({n_cached} cached, {n_gen} to generate)"
        {gen_model_lines}
        echo "  Judge:      {pipeline.judge_model} ({"compute node" if pipeline.judge_local else "login node"})"
        echo "  Matchmaker: {pipeline.matchmaker}"
        echo "  Tag:        {pipeline.tag}"
        echo "═══════════════════════════════════════════════════════"
        echo ""
    """)

    lines: list[str] = [preamble]

    if has_gen:
        if any(gen_locals):
            lines.append(SLURM_WAIT_HELPER)

        lines.append("# ── Generation Phase ───────────────────────────────")
        lines.append("SLURM_GEN_JOBS=()  # collect sbatch job IDs for dependency")
        lines.append("")

        for i, (path, is_local) in enumerate(zip(gen_paths, gen_locals)):
            short = path.stem.split("_", 2)[-1][:25]
            if is_local:
                lines.append(f'JOB_{i+1}=$(sbatch --parsable {path})')
                lines.append(f'echo "✅ Submitted Gen [{i+1}] {short}: $JOB_{i+1}"')
                lines.append(f'SLURM_GEN_JOBS+=("$JOB_{i+1}")')
            else:
                log = f"{pipeline.logs_dir}/oj_gen{i+1}{tag_suffix}.log"
                lines.append(f'echo "🚀 Starting Gen [{i+1}] {short} on login node..."')
                lines.append(f'bash {path} > {log} 2>&1 &')
                lines.append(f'PID_{i+1}=$!')
                lines.append(f'echo "   PID: $PID_{i+1}  Log: {log}"')
            lines.append("")

        lines.append("# ── Synchronise: wait for ALL generation jobs ─────")

        for i, is_local in enumerate(gen_locals):
            if not is_local:
                short = gen_paths[i].stem.split("_", 2)[-1][:25]
                lines.append(f'echo "⏳ Waiting for Gen [{i+1}] {short} (PID $PID_{i+1})..."')
                lines.append(f'wait $PID_{i+1} || {{ echo "❌ Gen [{i+1}] failed"; exit 1; }}')
                lines.append(f'echo "✅ Gen [{i+1}] {short} completed"')

        for i, is_local in enumerate(gen_locals):
            if is_local:
                short = gen_paths[i].stem.split("_", 2)[-1][:25]
                lines.append(f'_wait_slurm_job "$JOB_{i+1}" "Gen [{i+1}] {short}"')

        lines.append("")
        lines.append('echo ""')
        lines.append('echo "All generation jobs completed."')
        lines.append('echo ""')
    else:
        if is_analyze_only:
            lines.append('echo "✅ Generation skipped for --stage analyze"')
        else:
            lines.append('echo "✅ All completions loaded from cache — skipping generation"')
        lines.append('echo ""')

    lines.append("")
    lines.append(f"# ── {phase_label} Phase ──────────────────────────────")

    if not has_gen:
        if pipeline.judge_local:
            lines.append(f'JOB_ARENA=$(sbatch --parsable {path_judge})')
            lines.append(f'echo "✅ Submitted {phase_label}: $JOB_ARENA  ({path_judge.name})"')
            lines.append(f'echo "Results: {results_dir}/"')
        else:
            log_judge = f"{pipeline.logs_dir}/oj_arena{tag_suffix}.log"
            lines.append(f'echo "🚀 Running {phase_label_lower} on login node..."')
            lines.append(f'bash {path_judge} > {log_judge} 2>&1')
            lines.append(f'echo "✅ {phase_label} completed"')
            lines.append(f'echo "Results: {results_dir}/"')
            lines.append(f'echo "Log:     {log_judge}"')
    elif pipeline.judge_local and all(gen_locals):
        lines.append('DEP_STR=$(IFS=:; echo "${SLURM_GEN_JOBS[*]}")')
        lines.append(f'JOB_ARENA=$(sbatch --parsable --dependency=afterok:$DEP_STR {path_judge})')
        lines.append(f'echo "✅ Submitted {phase_label}: $JOB_ARENA  ({path_judge.name})"')
        lines.append('echo "   └── depends on: $DEP_STR"')
        lines.append("")
        lines.append('echo "Monitor: squeue -j $JOB_ARENA"')
        lines.append(f'echo "Results: {results_dir}/"')
    elif pipeline.judge_local:
        lines.append(f'JOB_ARENA=$(sbatch --parsable {path_judge})')
        lines.append(f'echo "✅ Submitted {phase_label}: $JOB_ARENA  ({path_judge.name})"')
        lines.append(f'echo "Results: {results_dir}/"')
    else:
        log_judge = f"{pipeline.logs_dir}/oj_arena{tag_suffix}.log"
        lines.append(f'echo "🚀 Running {phase_label_lower} on login node..."')
        lines.append(f'bash {path_judge} > {log_judge} 2>&1')
        lines.append(f'echo "✅ {phase_label} completed"')
        lines.append(f'echo "Results: {results_dir}/"')
        lines.append(f'echo "Log:     {log_judge}"')

    lines.append("")
    return "\n".join(lines) + "\n"


def build_generate_submit_all(
    pipeline: Any,
    gen_paths: list[Path],
    gen_locals: list[bool],
    tag_suffix: str,
) -> str:
    """Build ``submit_all.sh`` for generate mode."""
    n_gen = len(gen_paths)

    lines: list[str] = []
    lines.append(dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        mkdir -p {pipeline.logs_dir} {pipeline.work_dir}

        echo "═══════════════════════════════════════════════════"
        echo "  OpenJury — Generate Completions ({n_gen} model(s))"
        echo "  Dataset: {pipeline.dataset}"
        echo "═══════════════════════════════════════════════════"
    """))

    if any(gen_locals):
        lines.append(SLURM_WAIT_HELPER)

    lines.append("SLURM_GEN_JOBS=()")
    lines.append("")

    for i, (path, is_local) in enumerate(zip(gen_paths, gen_locals)):
        short = path.stem.split("_", 2)[-1][:25]
        if is_local:
            lines.append(f'JOB_{i+1}=$(sbatch --parsable {path})')
            lines.append(f'echo "✅ Submitted Gen [{i+1}] {short}: $JOB_{i+1}"')
            lines.append(f'SLURM_GEN_JOBS+=("$JOB_{i+1}")')
        else:
            log = f"{pipeline.logs_dir}/oj_gen{i+1}{tag_suffix}.log"
            lines.append(f'echo "🚀 Starting Gen [{i+1}] {short} on login node..."')
            lines.append(f'bash {path} > {log} 2>&1 &')
            lines.append(f'PID_{i+1}=$!')
            lines.append(f'echo "   PID: $PID_{i+1}  Log: {log}"')
        lines.append("")

    lines.append("# Wait for all generation jobs")
    for i, is_local in enumerate(gen_locals):
        short = gen_paths[i].stem.split("_", 2)[-1][:25]
        if not is_local:
            lines.append(f'wait $PID_{i+1} || {{ echo "❌ Gen [{i+1}] failed"; exit 1; }}')
            lines.append(f'echo "✅ Gen [{i+1}] {short} completed"')
        else:
            lines.append(f'_wait_slurm_job "$JOB_{i+1}" "Gen [{i+1}] {short}"')

    lines.append("")
    lines.append('echo ""')
    lines.append('echo "✅ All generation jobs completed."')
    return "\n".join(lines) + "\n"


def build_judge_submit_all(
    pipeline: Any,
    path_judge: Path,
    results_dir: str,
    tag_suffix: str,
) -> str:
    """Build ``submit_all.sh`` for judge-only mode."""
    if pipeline.judge_local:
        return dedent(f"""\
            #!/bin/bash
            set -euo pipefail
            mkdir -p {pipeline.logs_dir} {pipeline.work_dir}

            echo "═══════════════════════════════════════════════════"
            echo "  OpenJury — Judge Only"
            echo "  Dataset:  {pipeline.dataset}"
            echo "  Models:   {len(pipeline.models)}"
            echo "  Judge:    {pipeline.judge_model} (compute node, {pipeline.judge_gpus} GPU(s))"
            echo "  Mode:     {pipeline.judge_mode}"
            echo "═══════════════════════════════════════════════════"

            JOB=$(sbatch --parsable {path_judge})
            echo "✅ Submitted arena judge: $JOB  ({path_judge.name})"
            echo "Monitor: squeue -j $JOB"
            echo "Results: {results_dir}/"
        """)

    log_judge = f"{pipeline.logs_dir}/oj_arena{tag_suffix}.log"
    return dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        mkdir -p {pipeline.logs_dir} {pipeline.work_dir}

        echo "═══════════════════════════════════════════════════"
        echo "  OpenJury — Judge Only (login node)"
        echo "  Dataset:  {pipeline.dataset}"
        echo "  Models:   {len(pipeline.models)}"
        echo "  Judge:    {pipeline.judge_model} (API — no SLURM needed)"
        echo "  Mode:     {pipeline.judge_mode}"
        echo "═══════════════════════════════════════════════════"

        echo "🚀 Running arena judge on login node..."
        bash {path_judge} 2>&1 | tee {log_judge}
        echo "✅ Arena judge completed"
        echo "Results: {results_dir}/"
        echo "Log:     {log_judge}"
    """)


def build_agreement_submit_all(
    pipeline: Any,
    path_agreement: Path,
    results_dir: str,
    tag_suffix: str,
    *,
    stage: str = "all",
) -> str:
    """Build ``submit_all.sh`` for agreement mode."""
    phase_label = "Agreement Analyze" if stage == "analyze" else "Agreement Evaluation"
    phase_mode = "analyze" if stage == "analyze" else "annotate+analyze" if stage == "all" else "annotate"
    if pipeline.judge_local:
        return dedent(f"""\
            #!/bin/bash
            set -euo pipefail
            mkdir -p {pipeline.logs_dir} {pipeline.work_dir}

            echo "═══════════════════════════════════════════════════"
            echo "  OpenJury — {phase_label}"
            echo "  Dataset:  {pipeline.dataset}"
            echo "  Judge:    {pipeline.judge_model} (compute node, {pipeline.judge_gpus} GPU(s))"
            echo "  Mode:     {pipeline.judge_mode}"
            echo "  Stage:    {phase_mode}"
            echo "═══════════════════════════════════════════════════"

            JOB=$(sbatch --parsable {path_agreement})
            echo "✅ Submitted agreement job: $JOB  ({path_agreement.name})"
            echo "Monitor: squeue -j $JOB"
            echo "Results: {results_dir}/"
        """)

    log_agreement = f"{pipeline.logs_dir}/oj_agree{tag_suffix}.log"
    return dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        mkdir -p {pipeline.logs_dir} {pipeline.work_dir}

        echo "═══════════════════════════════════════════════════"
        echo "  OpenJury — {phase_label} (login node)"
        echo "  Dataset:  {pipeline.dataset}"
        echo "  Judge:    {pipeline.judge_model} (API — no SLURM needed)"
        echo "  Mode:     {pipeline.judge_mode}"
        echo "  Stage:    {phase_mode}"
        echo "═══════════════════════════════════════════════════"

        echo "🚀 Running agreement pipeline on login node..."
        bash {path_agreement} 2>&1 | tee {log_agreement}
        echo "✅ {phase_label} completed"
        echo "Results: {results_dir}/"
        echo "Log:     {log_agreement}"
    """)
