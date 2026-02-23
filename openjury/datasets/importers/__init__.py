"""Dataset importers — tools for ingesting external benchmark results.

Available importers:

- :mod:`~openjury.datasets.importers.alpaca_eval` — import ~160 model
  completions from the ``tatsu-lab/alpaca_eval`` GitHub repository into
  the OpenJury CompletionCache (prefix: ``AlpacaEval/``).

- :mod:`~openjury.datasets.importers.arena_hard` — import model answers
  from the ``lmarena/arena-hard-auto`` GitHub repository into the
  OpenJury CompletionCache (prefix: ``ArenaHard/``).
"""
