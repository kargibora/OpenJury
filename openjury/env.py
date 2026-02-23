"""Lightweight .env file loader and env-var helpers for OpenJury.

Reads ``KEY=VALUE`` lines from a ``.env`` file and injects them into
``os.environ`` **only when the key is not already set** — environment
variables always win.

Lookup order for the ``.env`` file:

1. ``$OPENJURY_ENV_FILE`` — explicit path override.
2. ``<project-root>/.env`` — next to ``pyproject.toml``.
3. ``<cwd>/.env`` — current working directory.

If no file is found, this is a no-op (not an error).

File format
-----------
* One ``KEY=VALUE`` per line.
* Lines starting with ``#`` are comments.
* Empty lines are ignored.
* Optional surrounding quotes on values (``'…'`` or ``"…"``) are stripped.
* Inline ``# comments`` are **not** supported (value includes everything
  after ``=``).

The same file can be sourced by bash::

    set -a; source .env; set +a
"""

from __future__ import annotations

import os
from pathlib import Path

_loaded: set[str] = set()  # Track already-loaded files to avoid double-loading.


# ═════════════════════════════════════════════════════════════════════
#  .env file discovery & parsing
# ═════════════════════════════════════════════════════════════════════


def _find_env_file() -> Path | None:
    """Locate the ``.env`` file using the lookup order described above."""
    # 1. Explicit override
    explicit = os.environ.get("OPENJURY_ENV_FILE")
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None

    # 2. Project root (walk up from this file until we find pyproject.toml)
    here = Path(__file__).resolve().parent  # openjury/
    for ancestor in [here.parent, *here.parent.parents]:
        candidate = ancestor / ".env"
        if (ancestor / "pyproject.toml").exists() and candidate.is_file():
            return candidate
        if ancestor == ancestor.parent:
            break

    # 3. cwd
    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file():
        return cwd_env

    return None


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict of key-value pairs."""
    result: dict[str, str] = {}
    for _lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            result[key] = value
    return result


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> bool:
    """Load a ``.env`` file into ``os.environ``.

    Args:
        path: Explicit path to the ``.env`` file.  If ``None``, the file
            is located automatically (see module docstring).
        override: If ``True``, values from the file overwrite existing
            env vars.  Default ``False`` — existing env vars win.

    Returns:
        ``True`` if a file was loaded, ``False`` otherwise.
    """
    if path is not None:
        env_path = Path(path).expanduser()
        if not env_path.is_file():
            return False
    else:
        env_path = _find_env_file()
        if env_path is None:
            return False

    resolved = str(env_path.resolve())
    if resolved in _loaded:
        return True
    _loaded.add(resolved)

    pairs = _parse_env_file(env_path)
    for key, value in pairs.items():
        if override or key not in os.environ:
            os.environ[key] = value

    return True


# ═════════════════════════════════════════════════════════════════════
#  Typed env-var access helpers
# ═════════════════════════════════════════════════════════════════════


def get(name: str, default: str | None = None) -> str | None:
    """Return an env var, or *default* if unset.

    Unlike ``os.environ.get``, this is meant to be called **after**
    ``load_dotenv()`` so ``.env`` values are already in the environment.
    """
    return os.environ.get(name, default)


def require(name: str) -> str:
    """Return an env var or raise with a clear message."""
    val = os.environ.get(name)
    if val is None:
        raise EnvironmentError(
            f"Required environment variable ${name} is not set. "
            f"Set it in your shell or add it to .env."
        )
    return val


def check_required(names: list[str]) -> None:
    """Verify that **all** listed env vars are set.

    Raises a single ``EnvironmentError`` listing every missing variable
    so the user can fix them all at once.
    """
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        bullet_list = "\n".join(f"  - ${name}" for name in missing)
        raise EnvironmentError(
            f"The following required environment variables are not set:\n"
            f"{bullet_list}\n"
            f"Set them in your shell, in .env, or via CLI flags."
        )
