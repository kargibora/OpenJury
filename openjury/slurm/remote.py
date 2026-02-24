"""Remote SLURM submission via slurmpilot (SSH from laptop → cluster).

This module provides an alternative to the local ``bash submit_all.sh``
workflow.  Instead of generating scripts and manually submitting on the
cluster, :func:`submit_remote` uses slurmpilot to SSH into a configured
cluster, upload the generated scripts, and call ``sbatch`` remotely.

**API models** (OpenRouter, ChatOpenAI, etc.) that need network access are
run **locally on the developer machine** (not via sbatch), since they don't
need GPUs and the login node may not be reachable from compute nodes.

Setup (one-time)::

    # 1. Install the remote extra
    uv sync --extra remote

    # 2. Add your cluster(s) to slurmpilot
    sp-add-cluster --cluster leonardo \\
        --host login.leonardo.cineca.it \\
        --user YOUR_USERNAME \\
        --check-ssh-connection

    # 3. (Optional) set default cluster
    export SP_DEFAULT_CLUSTER=leonardo

Usage::

    uv run python -m openjury.slurm.generate_slurm \\
        --arena_config configs/arena_5model.json \\
        --submit --remote --cluster leonardo

Architecture::

    Local machine                   Remote cluster (SLURM)
    ─────────────                   ──────────────────────
    generate scripts (dry-run)
    │
    ├─ GPU models ──SSH/sbatch───→  compute nodes (sbatch)
    │                               │
    ├─ API models ──run locally──   (no sbatch needed)
    │                               │
    └─ wait for completion ←────── poll sacct via SSH
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from openjury._logging import logger


def _import_slurmpilot():
    """Import slurmpilot with a helpful error if not installed."""
    try:
        from slurmpilot import SlurmPilot, JobCreationInfo
        return SlurmPilot, JobCreationInfo
    except ImportError:
        logger.error(
            "slurmpilot is not installed.  Install the 'remote' extra:\n"
            "  uv sync --extra remote\n"
            "Or:  pip install 'slurmpilot @ git+https://github.com/geoalgo/slurmpilot.git@main'"
        )
        raise SystemExit(1)


def submit_remote(
    script_dir: Path,
    gen_scripts: list[Path],
    gen_locals: list[bool],
    judge_script: Path | None,
    judge_local: bool,
    cluster: str,
    *,
    project_dir: str = "",
    remote_project_dir: str | None = None,
    wait: bool = True,
    wait_timeout: int = 7200,
) -> dict[str, list[int]]:
    """Submit a pipeline to a remote cluster via slurmpilot.

    Args:
        script_dir: Local directory containing the generated ``.sh`` scripts.
        gen_scripts: Paths to generation scripts (one per model).
        gen_locals: For each gen script, ``True`` if it's a GPU (sbatch) job,
            ``False`` if it's an API model (run locally).
        judge_script: Path to the arena judge script (``None`` for
            generate-only mode).
        judge_local: ``True`` if the judge needs sbatch (GPU),
            ``False`` if it's an API model (run locally).
        cluster: slurmpilot cluster name (e.g. ``"leonardo"``).
        project_dir: Local OpenJury project root (for source sync).
        remote_project_dir: If set, scripts ``cd`` here on the remote
            instead of uploading sources. Use this when code is already
            deployed on the cluster.
        wait: If ``True``, poll until all jobs complete.
        wait_timeout: Max seconds to wait per job (default 2h).

    Returns:
        Dict with ``"gen_job_ids"`` and ``"judge_job_id"`` lists.

    Raises:
        SystemExit: If slurmpilot is not installed or a job fails.
    """
    SlurmPilot, JobCreationInfo = _import_slurmpilot()

    logger.info("Connecting to cluster '%s' via SSH...", cluster)
    slurm = SlurmPilot(clusters=[cluster])

    gen_job_ids: list[int] = []
    local_pids: list[tuple[int, str, subprocess.Popen]] = []

    # ── Phase 1: Submit generation jobs ──────────────────────────
    logger.info("═══ Phase 1: Generation (%d models) ═══", len(gen_scripts))

    for i, (script_path, is_local) in enumerate(zip(gen_scripts, gen_locals)):
        model_tag = script_path.stem  # e.g. "01_generate_Qwen2.5-0.5B"

        if is_local:
            # GPU model → submit via sbatch on remote
            job_id = _submit_script_remote(
                slurm, cluster, script_path, script_dir,
                job_name=model_tag,
                remote_project_dir=remote_project_dir,
            )
            gen_job_ids.append(job_id)
            logger.info(
                "  ✅ [%d/%d] %s → SLURM job %d",
                i + 1, len(gen_scripts), model_tag, job_id,
            )
        else:
            # API model → run locally on this machine
            logger.info(
                "  🚀 [%d/%d] %s → running locally (API model)",
                i + 1, len(gen_scripts), model_tag,
            )
            proc = subprocess.Popen(
                ["bash", str(script_path)],
                cwd=str(script_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            local_pids.append((i, model_tag, proc))

    # ── Phase 2: Wait for generation ─────────────────────────────
    if wait and (gen_job_ids or local_pids):
        logger.info("═══ Phase 2: Waiting for generation jobs ═══")

        # Wait for local API processes first
        for idx, tag, proc in local_pids:
            logger.info("  ⏳ Waiting for local process: %s (PID %d)", tag, proc.pid)
            stdout, _ = proc.communicate()
            if proc.returncode != 0:
                logger.error(
                    "  ❌ %s failed (exit %d):\n%s",
                    tag, proc.returncode,
                    stdout.decode(errors="replace")[-2000:] if stdout else "",
                )
                raise SystemExit(1)
            logger.info("  ✅ %s completed locally", tag)

        # Wait for remote SLURM jobs
        # slurmpilot uses job *names* for wait, but we need to track
        # by the metadata. Use the schedule_job return convention.
        for job_id in gen_job_ids:
            _wait_remote_job(slurm, job_id, cluster, timeout=wait_timeout)

    # ── Phase 3: Submit judge job ────────────────────────────────
    judge_job_id: int | None = None

    if judge_script is not None:
        logger.info("═══ Phase 3: Arena Judge ═══")

        if judge_local:
            # GPU judge → sbatch with dependency on all gen jobs
            dep_str = (
                ":".join(str(jid) for jid in gen_job_ids)
                if gen_job_ids else None
            )
            judge_job_id = _submit_script_remote(
                slurm, cluster, judge_script, script_dir,
                job_name=judge_script.stem,
                remote_project_dir=remote_project_dir,
                dependency=f"afterok:{dep_str}" if dep_str else None,
            )
            logger.info(
                "  ✅ Judge → SLURM job %d (depends on: %s)",
                judge_job_id,
                dep_str or "none",
            )

            if wait:
                _wait_remote_job(slurm, judge_job_id, cluster, timeout=wait_timeout)
        else:
            # API judge → run locally
            logger.info("  🚀 Running judge locally (API model)")
            result = subprocess.run(
                ["bash", str(judge_script)],
                cwd=str(script_dir),
            )
            if result.returncode != 0:
                logger.error("  ❌ Judge failed (exit %d)", result.returncode)
                raise SystemExit(1)
            logger.info("  ✅ Judge completed locally")

    # ── Summary ──────────────────────────────────────────────────
    logger.info("")
    logger.info("═══ Pipeline Complete ═══")
    if gen_job_ids:
        logger.info("  Remote gen jobs:  %s", gen_job_ids)
    if local_pids:
        logger.info("  Local gen procs:  %d completed", len(local_pids))
    if judge_job_id:
        logger.info("  Remote judge job: %d", judge_job_id)

    return {
        "gen_job_ids": gen_job_ids,
        "judge_job_id": [judge_job_id] if judge_job_id else [],
    }


# ═════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═════════════════════════════════════════════════════════════════════


def _submit_script_remote(
    slurm,
    cluster: str,
    script_path: Path,
    script_dir: Path,
    job_name: str,
    *,
    remote_project_dir: str | None = None,
    dependency: str | None = None,
) -> int:
    """Upload and submit a single sbatch script via slurmpilot.

    Instead of using slurmpilot's ``JobCreationInfo`` (which expects a
    Python entrypoint), we use its SSH connection to upload the pre-built
    script and call ``sbatch`` directly.  This preserves our rich script
    templates (env setup, HF offline flags, etc.).
    """
    from slurmpilot import unify

    # Use a unique job name to avoid collisions
    unique_name = unify(f"oj/{job_name}", method="date")

    # Get the SSH connection for the cluster
    connections = slurm.connections
    if cluster not in connections:
        raise RuntimeError(
            f"No SSH connection to cluster '{cluster}'. "
            f"Available: {list(connections.keys())}"
        )
    conn = connections[cluster]

    # Determine remote directory
    if remote_project_dir:
        remote_dir = remote_project_dir
    else:
        # Upload to slurmpilot's default job area
        remote_dir = f"slurmpilot/jobs/{unique_name}"
        conn.run(f"mkdir -p {remote_dir}")

    # Upload the script and any config files in the same directory
    script_name = script_path.name
    _upload_file(conn, script_path, f"{remote_dir}/{script_name}")

    # Also upload arena_config.json if it exists
    config_path = script_dir / "arena_config.json"
    if config_path.exists():
        _upload_file(conn, config_path, f"{remote_dir}/arena_config.json")

    # Build sbatch command
    sbatch_cmd = f"cd {remote_dir} && sbatch --parsable"
    if dependency:
        sbatch_cmd += f" --dependency={dependency}"
    sbatch_cmd += f" {script_name}"

    logger.debug("Remote command: %s", sbatch_cmd)
    stdout, stderr = conn.run(sbatch_cmd)

    # Parse job ID from sbatch --parsable output
    try:
        job_id = int(stdout.strip().split(";")[0])
    except (ValueError, IndexError):
        logger.error(
            "Failed to parse sbatch output.\nstdout: %s\nstderr: %s",
            stdout, stderr,
        )
        raise SystemExit(1)

    return job_id


def _upload_file(conn, local_path: Path, remote_path: str) -> None:
    """Upload a single file to the remote cluster."""
    # slurmpilot's connection object supports rsync/scp
    # Use the underlying SSH to do a simple file transfer
    try:
        conn.upload(str(local_path), remote_path)
    except AttributeError:
        # Fallback: use the run method with a heredoc
        content = local_path.read_text()
        escaped = content.replace("'", "'\\''")
        conn.run(f"mkdir -p $(dirname {remote_path})")
        conn.run(f"cat > {remote_path} << 'OPENJURY_EOF'\n{escaped}\nOPENJURY_EOF")
        conn.run(f"chmod +x {remote_path}")


def _wait_remote_job(
    slurm,
    job_id: int,
    cluster: str,
    timeout: int = 7200,
) -> None:
    """Poll a remote SLURM job until completion or failure."""
    import time

    conn = slurm.connections[cluster]
    start = time.monotonic()

    logger.info("  ⏳ Waiting for SLURM job %d on '%s'...", job_id, cluster)

    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            logger.error(
                "  ⏰ Timeout (%ds) waiting for job %d", timeout, job_id,
            )
            raise SystemExit(1)

        # Query job state via sacct
        stdout, _ = conn.run(
            f"sacct -j {job_id} -n -o State --parsable2 2>/dev/null | head -1"
        )
        state = stdout.strip().upper()

        if state in ("COMPLETED",):
            logger.info("  ✅ Job %d completed", job_id)
            return
        elif state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"):
            logger.error("  ❌ Job %d failed (state: %s)", job_id, state)
            # Try to fetch logs
            try:
                log_stdout, _ = conn.run(
                    f"sacct -j {job_id} -n -o ExitCode,MaxRSS,Elapsed --parsable2 | head -1"
                )
                logger.error("  Job details: %s", log_stdout.strip())
            except Exception:
                pass
            raise SystemExit(1)
        elif state in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING"):
            pass  # still going
        elif not state:
            pass  # sacct not ready yet

        time.sleep(30)
