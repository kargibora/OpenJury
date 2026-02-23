"""Built-in dataset loaders.

Importing this package auto-registers all loaders with
:class:`~openjury.datasets.registry.DatasetRegistry`.
"""

# Import all loader modules so their @register decorators execute.
from openjury.datasets.loaders import (  # noqa: F401
    alpaca_eval,
    arena_hard,
    m_arenahard,
    lmsys,
    comparia,
)
