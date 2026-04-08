"""Migrate legacy generate config files to the nested schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openjury.generate_config import GenerateConfig


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML configs. Install with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Generate config must be a JSON/YAML object.")
    return data


def _save_config(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to write YAML configs. Install with: pip install pyyaml"
            ) from exc
        text = yaml.safe_dump(data, sort_keys=False)
    else:
        text = json.dumps(data, indent=2)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="openjury-migrate-generate-config",
        description=(
            "Rewrite a legacy flat openjury-generate config into the new nested "
            "dataset/model/runtime schema."
        ),
    )
    parser.add_argument("input", help="Path to the source generate config file.")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Optional output path. Defaults to --in-place behavior only if requested.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file instead of writing to a separate output path.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if args.in_place and args.output is not None:
        parser.error("Use either an explicit output path or --in-place, not both.")
    if not args.in_place and args.output is None:
        parser.error("Provide an output path or pass --in-place.")

    output_path = input_path if args.in_place else Path(args.output)
    raw = _load_config(input_path)
    upgraded = GenerateConfig.upgrade_legacy_dict(raw)
    normalized = GenerateConfig.from_dict(upgraded).to_dict()
    _save_config(output_path, normalized)

    if output_path == input_path:
        print(f"Migrated generate config in place: {input_path}")
    else:
        print(f"Migrated generate config written to: {output_path}")


if __name__ == "__main__":
    main()
