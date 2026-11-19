"""Fold summaries, Welch comparisons, and publication figures."""

from pathlib import Path

import pandas as pd

from .metrics import METRICS, summarize, welch_tests


def collect_runs(root, output, plots=False):
    output = Path(output)
    files = sorted(Path(root).glob("**/metrics.csv"))
    if not files:
        raise ValueError("No metrics.csv files found under the run directory.")
    frames = []
    for p in files:
        if p.resolve() == (output / "metrics.csv").resolve():
            continue
        frame = pd.read_csv(p)
        frame["experiment"] = p.parent.relative_to(root).as_posix()
        frames.append(frame)
    if not frames:
        raise ValueError("No experiment metrics found.")
    metrics = pd.concat(frames, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(output / "summary.csv", index=False)
    paper = paper_summary(metrics)
    paper.to_csv(output / "paper_summary.csv", index=False)
    ablation_changes(paper).to_csv(output / "ablation_changes.csv", index=False)
    compare_paper(paper).to_csv(output / "paper_comparison.csv", index=False)
    comparisons = metrics.copy()
    comparisons["experiment"] = comparisons.experiment.str.rsplit("/", n=1).str[0]
    welch_tests(comparisons).to_csv(output / "welch.csv", index=False)
    if plots:
        plot_groups(summary, output)
    rows = summary[(summary.group == 0) & (summary.subset == "Overall")]
    lines = [
        "# Experiment results",
        "",
        "Mean ± sample standard deviation across completed folds.",
        "",
        "| Experiment | Folds | " + " | ".join(METRICS) + " |",
        "|---|---:|" + "---:|" * len(METRICS),
    ]
    for _, row in rows.iterrows():
        values = [f"{row[m + '_mean']:.3f} ± {row[m + '_std']:.3f}" for m in METRICS]
        lines.append(f"| {row.experiment} | {row.folds} | " + " | ".join(values) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def plot_groups(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = summary[summary.group > 0].copy()
    data["cohort"] = data.experiment.str.split("/").str[0]
    data = data[data.cohort.isin(["raw", "balanced"])]
    cohorts = [c for c in ("raw", "balanced") if c in set(data.cohort)]
    if not cohorts:
        return
    for subset in ("Overall", "Stable", "Converter"):
        fig, axes = plt.subplots(
            1,
            len(cohorts),
            figsize=(6 * len(cohorts), 4),
            layout="constrained",
            squeeze=False,
            sharey=True,
        )
        for ax, cohort in zip(axes[0], cohorts, strict=True):
            frame = data[(data.subset == subset) & (data.cohort == cohort)]
            for name, curve in frame.groupby("experiment", sort=True):
                curve = curve.sort_values("group")
                ax.errorbar(
                    curve.group,
                    curve.bca_mean,
                    yerr=curve.bca_std.fillna(0),
                    marker="o",
                    capsize=2,
                    label=name.rsplit("/", 1)[-1],
                )
            ax.set(
                xlabel="Number of historical visits",
                ylabel="BCA",
                ylim=(0, 1),
                title=f"{cohort.title()} · {subset}",
            )
            ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(Path(output) / f"{subset.lower()}_groups.png", dpi=200)
        fig.savefig(Path(output) / f"{subset.lower()}_groups.pdf")
        plt.close(fig)


def compare_paper(summary):
    targets = pd.read_csv(Path(__file__).with_name("paper_targets.csv"))
    rows = []
    for _, measured in summary[summary.group == 0].iterrows():
        experiment = measured.experiment.rsplit("/", 1)[0]
        target = targets[
            (targets.experiment == experiment)
            & (targets.model == measured.model)
            & (targets.subset == measured.subset)
        ]
        if target.empty:
            continue
        for metric in METRICS:
            value = target[metric].iloc[0]
            if pd.notna(value):
                rows.append(
                    dict(
                        experiment=experiment,
                        model=measured.model,
                        subset=measured.subset,
                        metric=metric,
                        folds=measured.folds,
                        measured=measured[metric + "_mean"],
                        measured_std=measured[metric + "_std"],
                        reported=value,
                        difference=measured[metric + "_mean"] - value,
                    )
                )
    return pd.DataFrame(rows)


def paper_summary(metrics):
    import numpy as np

    rows = []
    for (experiment, model, subset), data in metrics.groupby(["experiment", "model", "subset"]):
        row = dict(
            experiment=experiment, model=model, subset=subset, group=0, folds=data.fold.nunique()
        )
        if model in ("minrnn", "lstm", "gru") and (data.group > 0).any():
            groups = data[data.group > 0].groupby("group", sort=True)
            means = groups[list(METRICS)].mean()
            weights = (
                groups.group_weight.first()
                if "group_weight" in data and data.group_weight.notna().any()
                else groups.n.sum()
            )
            row["aggregation"] = "weighted_group_means"
            for metric in METRICS:
                valid = means[metric].notna() & weights.gt(0)
                value = float(np.average(means.loc[valid, metric], weights=weights[valid]))
                variance = np.average(
                    (means.loc[valid, metric] - value) ** 2, weights=weights[valid]
                )
                row[metric + "_mean"], row[metric + "_std"] = value, float(np.sqrt(variance))
        else:
            folds = data[data.group == 0]
            row["aggregation"] = "fold_mean"
            for metric in METRICS:
                row[metric + "_mean"] = float(folds[metric].mean())
                row[metric + "_std"] = float(folds[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def reproduce_paper(output, plots=False):
    output = Path(output)
    records = pd.read_csv(Path(__file__).with_name("paper_folds.csv"))
    for experiment, data in records.groupby("experiment", sort=True):
        dest = output / "fold_records" / experiment
        dest.mkdir(parents=True, exist_ok=True)
        data.drop(columns="experiment").to_csv(dest / "metrics.csv", index=False)
    result = collect_runs(output / "fold_records", output, plots=False)
    if plots:
        summary = pd.read_csv(output / "summary.csv")
        groups = pd.read_csv(Path(__file__).with_name("paper_groups.csv"))
        plot_groups(pd.concat([summary, groups], ignore_index=True), output)
    (output / "provenance.txt").write_text(
        "Statistical reanalysis of the paper's recorded fold metrics.\n", encoding="utf-8"
    )
    return result


def ablation_changes(summary):
    columns = ["experiment", "bca", "baseline_bca", "change_percent", "reported_change_percent"]
    baseline = summary[(summary.experiment == "raw/transformer") & (summary.subset == "Overall")]
    if baseline.empty:
        return pd.DataFrame(columns=columns)
    value = baseline.bca_mean.iloc[0]
    targets = pd.read_csv(Path(__file__).with_name("paper_targets.csv")).set_index(
        ["experiment", "model", "subset"]
    )
    rows = []
    for _, row in summary[
        (summary.subset == "Overall") & (summary.model == "transformer")
    ].iterrows():
        name = row.experiment.rsplit("/", 1)[0]
        if not name.startswith(("remove_", "only_", "last_")):
            continue
        target = targets["bca_change_percent"].get((name, "transformer", "Overall"), float("nan"))
        rows.append(
            dict(
                experiment=name,
                bca=row.bca_mean,
                baseline_bca=value,
                change_percent=100 * (row.bca_mean - value) / value,
                reported_change_percent=target,
            )
        )
    return pd.DataFrame(rows, columns=columns)
