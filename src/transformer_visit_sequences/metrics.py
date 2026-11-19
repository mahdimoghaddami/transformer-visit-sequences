"""Three-class clinical status metrics and fold-level comparisons."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_ind

METRICS = ("accuracy", "bca", "mauc", "sensitivity", "specificity", "f1_score")


def classification_metrics(labels, probabilities):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or not len(y) or p.shape != (len(y), 3):
        raise ValueError("Expected nonempty labels [N] and probabilities [N, 3].")
    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError("Diagnosis labels must be 0 (CN), 1 (MCI), or 2 (AD).")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Probabilities must be finite, nonnegative, and sum to one.")
    pred = p.argmax(1)
    sensitivity, specificity, f1 = [], [], []
    for c in range(3):
        tp = int(((pred == c) & (y == c)).sum())
        tn = int(((pred != c) & (y != c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        sensitivity.append(tp / (tp + fn) if tp + fn else 0.5)
        specificity.append(tn / (tn + fp) if tn + fp else 0.5)
        if np.any(y == c) or np.any(pred == c):
            f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    auc = []
    for a, b in combinations(range(3), 2):
        mask = (y == a) | (y == b)
        for positive in (a, b):
            pos = y[mask] == positive
            n1, n0 = int(pos.sum()), int((~pos).sum())
            ranks = rankdata(p[mask, positive], method="average")
            auc.append((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else 0.0)
    sens, spec = float(np.mean(sensitivity)), float(np.mean(specificity))
    result = dict(
        accuracy=float(np.mean(pred == y)),
        bca=(sens + spec) / 2,
        mauc=float(np.mean(auc)),
        sensitivity=sens,
        specificity=spec,
        f1_score=float(np.mean(f1)),
    )
    for c, name in enumerate(("cn", "mci", "ad")):
        result[f"sensitivity_{name}"] = sensitivity[c]
        result[f"specificity_{name}"] = specificity[c]
    return result


def score_predictions(frame):
    required = {"RID", "fold", "group", "label", "converter", "p_cn", "p_mci", "p_ad"}
    if missing := required - set(frame):
        raise ValueError(f"Prediction columns missing: {sorted(missing)}")
    if not frame.converter.isin([0, 1]).all() or (frame.group < 1).any():
        raise ValueError("Groups must be positive and converter must be 0 or 1.")
    if not np.isfinite(frame[["fold", "group"]]).all().all() or (frame.fold < 0).any():
        raise ValueError("Fold and group identifiers must be finite and nonnegative.")
    if (frame[["fold", "group"]] % 1 != 0).any().any():
        raise ValueError("Fold and group identifiers must be integers.")
    if frame.duplicated(["RID"]).any():
        raise ValueError("Each subject must have exactly one out-of-fold prediction.")
    rows = []
    for fold, f in frame.groupby("fold", sort=True):
        for group in [0] + sorted(f.group.unique().tolist()):
            g = f if group == 0 else f[f.group == group]
            subsets = {
                "Overall": g,
                "Stable": g[g.converter == 0],
                "Converter": g[g.converter == 1],
            }
            for subset, data in subsets.items():
                if data.empty:
                    continue
                row = dict(fold=int(fold), group=int(group), subset=subset, n=len(data))
                row.update(classification_metrics(data.label, data[["p_cn", "p_mci", "p_ad"]]))
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(metrics):
    columns = [c for c in metrics if c in METRICS or c.startswith(("sensitivity_", "specificity_"))]
    keys = [c for c in ("experiment", "model", "group", "subset") if c in metrics]
    rows = []
    for key, data in metrics.groupby(keys, sort=True):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        row["folds"] = len(data)
        for col in columns:
            row[col + "_mean"] = float(data[col].mean())
            row[col + "_std"] = float(data[col].std(ddof=1)) if len(data) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def welch_tests(metrics):
    rows = []
    keys = [c for c in ("experiment", "group", "subset") if c in metrics]
    for key, data in metrics.groupby(keys):
        base = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        for a, b in combinations(sorted(data.model.unique()), 2):
            for metric in METRICS:
                x, y = [data[data.model == m][metric].dropna().to_numpy() for m in (a, b)]
                if len(x) < 2 or len(y) < 2:
                    continue
                result = ttest_ind(x, y, equal_var=False)
                rows.append(
                    dict(
                        base,
                        model_a=a,
                        model_b=b,
                        metric=metric,
                        n_a=len(x),
                        n_b=len(y),
                        statistic=float(result.statistic),
                        p_value=float(result.pvalue),
                    )
                )
    return pd.DataFrame(rows)
