"""Helpers for building SLURM/login-node submission wrapper scripts."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from openjury.annotate_config import AnnotateConfig
from openjury.arena.config import AgreementConfig, ArenaConfig
from openjury.slurm.plans import SlurmRunPlan

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
    plan: SlurmRunPlan,
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
    task = plan.task
    if not isinstance(task, ArenaConfig):
        raise TypeError("Arena submit builder requires an ArenaConfig task.")

    execution = plan.execution
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
        mkdir -p {execution.logs_dir} {execution.work_dir}

        echo "═══════════════════════════════════════════════════════"
        echo "  {pipeline_title}"
        echo "  Dataset:    {task.dataset}"
        echo "  Models:     {len(task.models)} total ({n_cached} cached, {n_gen} to generate)"
        {gen_model_lines}
        echo "  Judge:      {task.judge.model} ({"compute node" if execution.judge_local else "login node"})"
        echo "  Matchmaker: {task.matchmaker.strategy}"
        echo "  Tag:        {execution.tag}"
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
                log = f"{execution.logs_dir}/oj_gen{i+1}{tag_suffix}.log"
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
        if execution.judge_local:
            lines.append(f'JOB_ARENA=$(sbatch --parsable {path_judge})')
            lines.append(f'echo "✅ Submitted {phase_label}: $JOB_ARENA  ({path_judge.name})"')
            lines.append(f'echo "Results: {results_dir}/"')
        else:
            log_judge = f"{execution.logs_dir}/oj_arena{tag_suffix}.log"
            lines.append(f'echo "🚀 Running {phase_label_lower} on login node..."')
            lines.append(f'bash {path_judge} > {log_judge} 2>&1')
            lines.append(f'echo "✅ {phase_label} completed"')
            lines.append(f'echo "Results: {results_dir}/"')
            lines.append(f'echo "Log:     {log_judge}"')
    elif execution.judge_local and all(gen_locals):
        lines.append('DEP_STR=$(IFS=:; echo "${SLURM_GEN_JOBS[*]}")')
        lines.append(f'JOB_ARENA=$(sbatch --parsable --dependency=afterok:$DEP_STR {path_judge})')
        lines.append(f'echo "✅ Submitted {phase_label}: $JOB_ARENA  ({path_judge.name})"')
        lines.append('echo "   └── depends on: $DEP_STR"')
        lines.append("")
        lines.append('echo "Monitor: squeue -j $JOB_ARENA"')
        lines.append(f'echo "Results: {results_dir}/"')
    elif execution.judge_local:
        lines.append(f'JOB_ARENA=$(sbatch --parsable {path_judge})')
        lines.append(f'echo "✅ Submitted {phase_label}: $JOB_ARENA  ({path_judge.name})"')
        lines.append(f'echo "Results: {results_dir}/"')
    else:
        log_judge = f"{execution.logs_dir}/oj_arena{tag_suffix}.log"
        lines.append(f'echo "🚀 Running {phase_label_lower} on login node..."')
        lines.append(f'bash {path_judge} > {log_judge} 2>&1')
        lines.append(f'echo "✅ {phase_label} completed"')
        lines.append(f'echo "Results: {results_dir}/"')
        lines.append(f'echo "Log:     {log_judge}"')

    lines.append("")
    return "\n".join(lines) + "\n"


def build_generate_submit_all(
    plan: SlurmRunPlan,
    gen_paths: list[Path],
    gen_locals: list[bool],
    tag_suffix: str,
) -> str:
    """Build ``submit_all.sh`` for generate mode."""
    execution = plan.execution
    n_gen = len(gen_paths)

    lines: list[str] = []
    lines.append(dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        mkdir -p {execution.logs_dir} {execution.work_dir}

        echo "═══════════════════════════════════════════════════"
        echo "  OpenJury — Generate Completions ({n_gen} model(s))"
        echo "  Dataset: {plan.dataset}"
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
            log = f"{execution.logs_dir}/oj_gen{i+1}{tag_suffix}.log"
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
    plan: SlurmRunPlan,
    path_judge: Path,
    results_dir: str,
    tag_suffix: str,
) -> str:
    """Build ``submit_all.sh`` for judge-only mode."""
    task = plan.task
    if not isinstance(task, ArenaConfig):
        raise TypeError("Judge submit builder requires an ArenaConfig task.")

    execution = plan.execution
    if execution.judge_local:
        return dedent(f"""\
            #!/bin/bash
            set -euo pipefail
            mkdir -p {execution.logs_dir} {execution.work_dir}

            echo "═══════════════════════════════════════════════════"
            echo "  OpenJury — Judge Only"
            echo "  Dataset:  {task.dataset}"
            echo "  Models:   {len(task.models)}"
            echo "  Judge:    {task.judge.model} (compute node, {task.judge.gpus} GPU(s))"
            echo "  Mode:     {task.judge.mode}"
            echo "═══════════════════════════════════════════════════"

            JOB=$(sbatch --parsable {path_judge})
            echo "✅ Submitted arena judge: $JOB  ({path_judge.name})"
            echo "Monitor: squeue -j $JOB"
            echo "Results: {results_dir}/"
        """)

    log_judge = f"{execution.logs_dir}/oj_arena{tag_suffix}.log"
    return dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        mkdir -p {execution.logs_dir} {execution.work_dir}

        echo "═══════════════════════════════════════════════════"
        echo "  OpenJury — Judge Only (login node)"
        echo "  Dataset:  {task.dataset}"
        echo "  Models:   {len(task.models)}"
        echo "  Judge:    {task.judge.model} (API — no SLURM needed)"
        echo "  Mode:     {task.judge.mode}"
        echo "═══════════════════════════════════════════════════"

        echo "🚀 Running arena judge on login node..."
        bash {path_judge} 2>&1 | tee {log_judge}
        echo "✅ Arena judge completed"
        echo "Results: {results_dir}/"
        echo "Log:     {log_judge}"
    """)


def build_agreement_submit_all(
    plan: SlurmRunPlan,
    path_agreement: Path,
    results_dir: str,
    tag_suffix: str,
    *,
    stage: str = "all",
) -> str:
    """Build ``submit_all.sh`` for agreement mode."""
    task = plan.task
    if not isinstance(task, AgreementConfig):
        raise TypeError("Agreement submit builder requires an AgreementConfig task.")

    execution = plan.execution
    phase_label = "Agreement Analyze" if stage == "analyze" else "Agreement Evaluation"
    phase_mode = "analyze" if stage == "analyze" else "annotate+analyze" if stage == "all" else "annotate"
    if execution.judge_local:
        return dedent(f"""\
            #!/bin/bash
            set -euo pipefail
            mkdir -p {execution.logs_dir} {execution.work_dir}

            echo "═══════════════════════════════════════════════════"
            echo "  OpenJury — {phase_label}"
            echo "  Dataset:  {task.dataset}"
            echo "  Judge:    {task.judge.model} (compute node, {task.judge.gpus} GPU(s))"
            echo "  Mode:     {task.judge.mode}"
            echo "  Stage:    {phase_mode}"
            echo "═══════════════════════════════════════════════════"

            JOB=$(sbatch --parsable {path_agreement})
            echo "✅ Submitted agreement job: $JOB  ({path_agreement.name})"
            echo "Monitor: squeue -j $JOB"
            echo "Results: {results_dir}/"
        """)

    log_agreement = f"{execution.logs_dir}/oj_agree{tag_suffix}.log"
    return dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        mkdir -p {execution.logs_dir} {execution.work_dir}

        echo "═══════════════════════════════════════════════════"
        echo "  OpenJury — {phase_label} (login node)"
        echo "  Dataset:  {task.dataset}"
        echo "  Judge:    {task.judge.model} (API — no SLURM needed)"
        echo "  Mode:     {task.judge.mode}"
        echo "  Stage:    {phase_mode}"
        echo "═══════════════════════════════════════════════════"

        echo "🚀 Running agreement pipeline on login node..."
        bash {path_agreement} 2>&1 | tee {log_agreement}
        echo "✅ {phase_label} completed"
        echo "Results: {results_dir}/"
        echo "Log:     {log_agreement}"
    """)


def build_annotate_submit_all(
    plan: SlurmRunPlan,
    gen_paths: list[Path],
    gen_locals: list[bool],
    path_annotate: Path,
    results_dir: str,
    tag_suffix: str,
    *,
    task_for_annotate: AnnotateConfig | None = None,
) -> str:
    """Build ``submit_all.sh`` for annotate mode."""
    task = task_for_annotate or plan.task
    if not isinstance(task, AnnotateConfig):
        raise TypeError("Annotate submit builder requires an AnnotateConfig task.")

    execution = plan.execution
    challenger_label = (
        task.challenger.name
        if task.challenger is not None and task.challenger.name
        else "dataset pairs"
    )
    has_gen = bool(gen_paths)

    lines: list[str] = [dedent(f"""\
        #!/bin/bash
        set -euo pipefail
        mkdir -p {execution.logs_dir} {execution.work_dir}

        echo "═══════════════════════════════════════════════════"
        echo "  OpenJury — Annotate"
        echo "  Dataset:     {task.dataset}"
        echo "  Challenger:  {challenger_label}"
        echo "  Judge:       {task.judge.model} ({'compute node' if execution.judge_local else 'API/login node'})"
        echo "  Pairing:     {task.pairing.source} / {task.pairing.strategy}"
        echo "═══════════════════════════════════════════════════"
        echo ""
    """)]

    if any(gen_locals):
        lines.append(SLURM_WAIT_HELPER)

    if has_gen:
        lines.append("# ── Challenger Generation Phase ───────────────────")
        lines.append("SLURM_GEN_JOBS=()")
        lines.append("")
        for i, (path, is_local) in enumerate(zip(gen_paths, gen_locals), start=1):
            if is_local:
                lines.append(f'JOB_GEN_{i}=$(sbatch --parsable {path})')
                lines.append(f'echo "✅ Submitted challenger generation: $JOB_GEN_{i}  ({path.name})"')
                lines.append(f'SLURM_GEN_JOBS+=("$JOB_GEN_{i}")')
            else:
                log = f"{execution.logs_dir}/oj_annotate_gen{tag_suffix}.log"
                lines.append('echo "🚀 Running challenger generation on login node..."')
                lines.append(f'bash {path} > {log} 2>&1 &')
                lines.append(f'PID_GEN_{i}=$!')
                lines.append(f'echo "   PID: $PID_GEN_{i}  Log: {log}"')
            lines.append("")

        for i, is_local in enumerate(gen_locals, start=1):
            if is_local:
                lines.append(f'_wait_slurm_job "$JOB_GEN_{i}" "challenger generation"')
            else:
                lines.append(
                    f'wait $PID_GEN_{i} || {{ echo "❌ Challenger generation failed"; exit 1; }}'
                )
                lines.append('echo "✅ Challenger generation completed"')
        lines.append("")

    lines.append("# ── Annotate Phase ─────────────────────────────────")
    if execution.judge_local:
        if has_gen and all(gen_locals):
            lines.append('DEP_STR=$(IFS=:; echo "${SLURM_GEN_JOBS[*]}")')
            lines.append(
                f'JOB_ANNOTATE=$(sbatch --parsable --dependency=afterok:$DEP_STR {path_annotate})'
            )
            lines.append(
                f'echo "✅ Submitted annotate job: $JOB_ANNOTATE  ({path_annotate.name})"'
            )
            lines.append('echo "   └── depends on: $DEP_STR"')
        else:
            lines.append(f'JOB_ANNOTATE=$(sbatch --parsable {path_annotate})')
            lines.append(
                f'echo "✅ Submitted annotate job: $JOB_ANNOTATE  ({path_annotate.name})"'
            )
        lines.append('echo "Monitor: squeue -j $JOB_ANNOTATE"')
        lines.append(f'echo "Results: {results_dir}/"')
    else:
        log_annotate = f"{execution.logs_dir}/oj_annotate{tag_suffix}.log"
        lines.append('echo "🚀 Running annotate pipeline on login node..."')
        lines.append(f'bash {path_annotate} 2>&1 | tee {log_annotate}')
        lines.append('echo "✅ Annotate completed"')
        lines.append(f'echo "Results: {results_dir}/"')
        lines.append(f'echo "Log:     {log_annotate}"')

    lines.append("")
    return "\n".join(lines) + "\n"
