"""Standalone pipeline steps for SLURM job chaining.

Each step is a self-contained script that reads/writes parquet files,
allowing models to run in separate SLURM jobs with ``--dependency``.

Steps:
    - ``generate_step``: Generate completions from a single model → parquet.
    - ``arena_step``: K-model arena evaluation (scoring, matchmaking, ratings).
"""
