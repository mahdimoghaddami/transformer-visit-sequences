"""Command-line interface for cohort preparation and paper experiments."""

import argparse
import json
from pathlib import Path
import tomllib

from .data import prepare
from .features import CATEGORIES
from .forecasts import evaluate_forecasts
from .metrics import score_predictions, summarize
from .report import collect_runs
from .train import run_experiment


DEFAULT_CONFIG = Path(__file__).with_name("paper.toml")


def parser():
    cli = argparse.ArgumentParser(description="Next-visit Alzheimer's diagnosis experiments")
    sub = cli.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare", help="Create a compact cohort and split manifest")
    data.add_argument("--visits", type=Path, required=True)
    data.add_argument("--out", type=Path, required=True)
    data.add_argument("--curated", action="store_true", help="Use an already selected cohort")
    data.add_argument(
        "--balanced", action="store_true", help="Downsample stable subjects per group"
    )
    data.add_argument(
        "--histories", type=Path, help="Imputed history CSV or directory of subject tables"
    )
    data.add_argument("--fold-dir", type=Path, help="Read fixed test subject membership")
    data.add_argument("--converter-rids", type=Path, help="Fixed curated converter membership")
    data.add_argument("--folds", type=int, default=10)
    data.add_argument("--seed", type=int, default=0)
    run = sub.add_parser("run", help="Train and evaluate one model with cross-validation")
    run.add_argument("--cohort", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument(
        "--model", choices=["transformer", "tst", "minrnn", "lstm", "gru"], required=True
    )
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--fold", type=int, help="Run just one zero-based fold")
    run.add_argument("--epochs", type=int, help="Override the configured training budget")
    run.add_argument("--device", default="auto")
    run.add_argument("--history-limit", type=int, choices=[1, 4])
    feature = run.add_mutually_exclusive_group()
    feature.add_argument("--remove", choices=CATEGORIES)
    feature.add_argument("--only", choices=CATEGORIES)
    run.add_argument("--include-time", action="store_true", default=None)
    run.add_argument("--no-time", action="store_false", dest="include_time")
    suite = sub.add_parser("suite", help="Run both cohorts, five models, and eight ablations")
    suite.add_argument("--data", type=Path, default=Path("data"))
    suite.add_argument("--out", type=Path, default=Path("runs/paper"))
    suite.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    suite.add_argument("--device", default="auto")
    suite.add_argument("--epochs", type=int)
    impute = sub.add_parser("impute", help="Build histories with fold-specific minimalRNN models")
    impute.add_argument("--cohort", type=Path, required=True)
    impute.add_argument("--models", type=Path, required=True)
    impute.add_argument("--device", default="auto")
    forecasts = sub.add_parser(
        "forecasts", help="Score monthly forecast tables at target visit dates"
    )
    forecasts.add_argument("--cohort", type=Path, required=True)
    forecasts.add_argument("--forecasts", type=Path, required=True)
    forecasts.add_argument("--out", type=Path, required=True)
    score = sub.add_parser("score", help="Score an out-of-fold prediction table")
    score.add_argument("predictions", type=Path)
    score.add_argument("--out", type=Path, required=True)
    report = sub.add_parser("report", help="Summarize completed runs and statistical comparisons")
    report.add_argument("runs", type=Path)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--plots", action="store_true")
    paper = sub.add_parser("paper", help="Recompute paper tables from recorded fold metrics")
    paper.add_argument("--out", type=Path, required=True)
    paper.add_argument("--plots", action="store_true")
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(
                args.visits,
                args.out,
                curated=args.curated,
                balanced=args.balanced,
                histories=args.histories,
                folds=args.folds,
                seed=args.seed,
                fold_dir=args.fold_dir,
                converter_rids=args.converter_rids,
            )
            print(json.dumps(result, indent=2))
        elif args.command == "run":
            config = tomllib.loads(args.config.read_text())
            result = run_experiment(
                args.cohort,
                args.out,
                args.model,
                config,
                fold=args.fold,
                epochs=args.epochs,
                device=args.device,
                history_limit=args.history_limit,
                remove=args.remove,
                only=args.only,
                include_time=args.include_time,
            )
            print(result[(result.group == 0) & (result.subset == "Overall")].to_string(index=False))
        elif args.command == "suite":
            config = tomllib.loads(args.config.read_text())
            for cohort in ("raw", "balanced"):
                for model in ("transformer", "tst", "minrnn", "lstm", "gru"):
                    run_experiment(
                        args.data / cohort,
                        args.out / cohort / model,
                        model,
                        config,
                        epochs=args.epochs,
                        device=args.device,
                    )
            for category in CATEGORIES:
                for mode in ("remove", "only"):
                    run_experiment(
                        args.data / "raw",
                        args.out / f"{mode}_{category}" / "transformer",
                        "transformer",
                        config,
                        epochs=args.epochs,
                        device=args.device,
                        **{mode: category},
                    )
            for n in (1, 4):
                run_experiment(
                    args.data / "raw",
                    args.out / f"last_{n}" / "transformer",
                    "transformer",
                    config,
                    epochs=args.epochs,
                    device=args.device,
                    history_limit=n,
                )
            collect_runs(args.out, args.out / "report")
        elif args.command == "impute":
            from .impute import export_histories, attach_histories

            target = args.cohort / "imputed.csv"
            export_histories(args.cohort, args.models, target, args.device)
            attach_histories(args.cohort, target)
            target.unlink()
        elif args.command == "forecasts":
            evaluate_forecasts(args.cohort, args.forecasts, args.out)
            print(f"Forecast evaluation saved to {args.out}")
        elif args.command == "score":
            import pandas as pd

            metrics = score_predictions(pd.read_csv(args.predictions))
            args.out.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(args.out / "metrics.csv", index=False)
            summarize(metrics).to_csv(args.out / "summary.csv", index=False)
        elif args.command == "paper":
            from .report import reproduce_paper

            print(reproduce_paper(args.out, args.plots).to_string(index=False))
        elif args.command == "report":
            print(collect_runs(args.runs, args.out, plots=args.plots).to_string(index=False))
    except (ValueError, FileNotFoundError, KeyError) as error:
        cli.exit(2, f"error: {error}\n")
