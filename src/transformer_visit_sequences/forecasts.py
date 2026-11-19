"""Evaluate monthly diagnosis forecasts at the next observed clinical visit."""

from pathlib import Path

import pandas as pd

from .data import load_cohort
from .metrics import score_predictions, summarize


PROBABILITIES = ("CN relative probability", "MCI relative probability", "AD relative probability")


def evaluate_forecasts(cohort, forecasts, output, models=("minrnn", "lstm", "gru")):
    visits, subjects, _ = load_cohort(cohort)
    targets = visits.groupby("RID", sort=False).EXAMDATE.last()
    names = {"minrnn": "MinRNN", "lstm": "LSTM", "gru": "GRU"}
    output = Path(output)
    for model in models:
        rows = []
        for fold, test in subjects.groupby("fold", sort=True):
            sources = sorted(Path(forecasts).glob(f"{names[model]}_prediction_test.f{fold}_g*.csv"))
            if not sources:
                raise ValueError(f"No forecasts found for model {model}, fold {fold}.")
            frame = pd.concat(
                [pd.read_csv(p, parse_dates=["Forecast Date"]) for p in sources], ignore_index=True
            )
            grouped = dict(tuple(frame.groupby("RID")))
            if set(grouped) != set(test.RID):
                raise ValueError("Forecast subjects disagree with the specified cohort and folds.")
            for _, meta in test.iterrows():
                s = grouped[int(meta.RID)]
                index = (s["Forecast Date"] - targets[int(meta.RID)]).abs().idxmin()
                row = meta[["RID", "fold", "group", "label", "converter"]].to_dict()
                row.update(
                    zip(
                        ("p_cn", "p_mci", "p_ad"),
                        s.loc[index, list(PROBABILITIES)].to_numpy(dtype=float),
                        strict=True,
                    )
                )
                rows.append(row)
        predictions = pd.DataFrame(rows)
        metrics = score_predictions(predictions)
        metrics["model"] = model
        dest = output / model
        dest.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(dest / "predictions.csv", index=False)
        metrics.to_csv(dest / "metrics.csv", index=False)
        summarize(metrics).to_csv(dest / "summary.csv", index=False)
    return output
