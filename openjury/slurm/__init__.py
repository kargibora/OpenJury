"""SLURM script generation and remote submission for LLM evaluation pipelines.

Modules:
    - ``generate_slurm``: Generate sbatch scripts (dry-run or --submit).
    - ``modes``: Mode-specific script planning/orchestration helpers.
    - ``render``: Shell script renderers for compute/login-node execution.
    - ``config_builders``: Typed config -> JSON payload builders for jobs.
    - ``submit_builders``: ``submit_all.sh`` wrapper builders/helpers.
    - ``remote``: Submit to a remote cluster via slurmpilot SSH (--remote).
"""
