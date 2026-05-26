from __future__ import annotations

import argparse
from types import SimpleNamespace

from openjury.arena.config import AgreementConfig, JudgeConfig
from openjury.cli import agreement as agreement_cli
from openjury.cli._resolvers.agreement import resolve_agreement_cli
from openjury.cli_args import add_dataset_args, add_dataset_selection_args, add_judge_args


def _agreement_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    add_dataset_args(parser, dataset_required=False)
    add_judge_args(parser)
    parser.add_argument("--output_dir", default="results/agreement")
    parser.add_argument("--ignore_score_cache", action="store_true")
    add_dataset_selection_args(parser)
    parser.add_argument("--truncate_instruction", type=int, default=500)
    parser.add_argument("--stage", choices=("all", "annotate", "analyze"), default="all")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--slurm_output_dir", default="slurm_scripts")
    return parser


def test_agreement_resolver_config_plus_explicit_overrides(tmp_path):
    cfg = AgreementConfig(
        dataset="lmsys",
        judge=JudgeConfig(model="Dummy/J", max_tokens=999),
        seed=7,
    )
    cfg_path = tmp_path / "agreement.json"
    cfg.save(cfg_path)

    parser = _agreement_parser()
    argv = [
        "--config", str(cfg_path),
        "--seed", "42",  # explicit default-valued override
        "--judge_max_tokens", "2048",  # explicit default-valued override
        "--max_model_len", "16384",
        "--enforce_eager",
        "--gen-kwargs", "top_p=0.9",
    ]
    args = parser.parse_args(argv)
    resolved = resolve_agreement_cli(parser, args, argv)

    assert resolved.config.seed == 42
    assert resolved.config.judge.max_tokens == 2048
    assert resolved.config.judge.max_model_len == 16384
    assert resolved.config.judge.enforce_eager is True
    assert resolved.config.judge.generation_kwargs == {"top_p": 0.9}


def test_agreement_resolver_auto_loads_stage_analyze_config(tmp_path):
    out = tmp_path / "agreement_out"
    out.mkdir()
    cfg = AgreementConfig(
        dataset="comparia",
        judge=JudgeConfig(model="Dummy/J"),
        output_dir=str(out),
        language="fr",
    )
    cfg.save(out / "agreement_config.json")

    parser = _agreement_parser()
    argv = ["--stage", "analyze", "--output_dir", str(out)]
    args = parser.parse_args(argv)
    resolved = resolve_agreement_cli(parser, args, argv)

    assert resolved.stage == "analyze"
    assert resolved.config.dataset == "comparia"
    assert resolved.config.language == "fr"


def test_agreement_main_skips_dataset_preflight_for_local_runs(monkeypatch):
    calls: dict[str, object] = {"preflight": 0, "run": None}
    config = AgreementConfig(dataset="lmsys", judge=JudgeConfig(model="Dummy/J"))

    monkeypatch.setattr(
        agreement_cli,
        "resolve_agreement_cli",
        lambda parser, args, argv: SimpleNamespace(config=config, slurm_forward=None),
    )

    def _unexpected_preflight(*args, **kwargs):
        calls["preflight"] = int(calls["preflight"]) + 1
        raise AssertionError("local agreement runs should not preflight datasets")

    monkeypatch.setattr(agreement_cli, "load_dataset", _unexpected_preflight)
    monkeypatch.setattr(
        agreement_cli,
        "run_agreement",
        lambda cfg: calls.__setitem__("run", cfg),
    )

    agreement_cli.main([])

    assert calls["preflight"] == 0
    assert calls["run"] == config


def test_agreement_main_preflights_before_slurm_forward(monkeypatch):
    preflight_calls: list[tuple[str, int]] = []
    forwarded: dict[str, object] = {}
    config = AgreementConfig(dataset="lmsys", judge=JudgeConfig(model="Dummy/J"))
    slurm_forward = SimpleNamespace(warn_output_dir_semantics=False)

    monkeypatch.setattr(
        agreement_cli,
        "resolve_agreement_cli",
        lambda parser, args, argv: SimpleNamespace(
            config=config,
            slurm_forward=slurm_forward,
        ),
    )
    monkeypatch.setattr(
        agreement_cli,
        "load_dataset",
        lambda dataset, n=5: preflight_calls.append((dataset, n)),
    )
    monkeypatch.setattr(
        agreement_cli,
        "forward_task_config_to_slurm",
        lambda task, cfg, forward: forwarded.update(
            {"task": task, "config": cfg, "forward": forward}
        ),
    )
    monkeypatch.setattr(
        agreement_cli,
        "run_agreement",
        lambda cfg: (_ for _ in ()).throw(
            AssertionError("SLURM-forwarded runs should not execute locally")
        ),
    )

    agreement_cli.main(["--slurm"])

    assert preflight_calls == [("lmsys", 5)]
    assert forwarded == {
        "task": "agreement",
        "config": config,
        "forward": slurm_forward,
    }
