"""Create finite visit histories from fold-specific minimalRNN checkpoints."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import classifier_subject_table, load_cohort, subject_table
from .models import ModelFillingRNN
from .train import monthly_arrays, monthly_batch, resolve_device


RNN_KEYS = ("kind", "h_size", "nb_layers", "h_drop", "i_drop")


def export_histories(cohort, checkpoints, output, device="auto"):
    visits, subjects, _ = load_cohort(cohort)
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new history table path.")
    device = resolve_device(device)
    torch.set_num_threads(4)
    frames = []
    for fold, test in subjects.groupby("fold", sort=True):
        checkpoint = torch.load(
            Path(checkpoints) / f"fold_{fold}" / "model.pt", map_location="cpu", weights_only=True
        )
        fields = checkpoint["fields"]
        settings = {key: checkpoint["settings"][key] for key in RNN_KEYS}
        model = ModelFillingRNN(nb_measures=len(fields), **settings).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        mean, std = np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"])
        samples = monthly_arrays(
            visits, test.RID.to_numpy(), fields, mean, std, observed_target=False
        )
        cats, vals, _, _ = monthly_batch(samples, device)
        with torch.no_grad():
            p_cat, p_val = model(cats, vals)
        p_cat, p_val = p_cat.cpu().numpy(), p_val.cpu().numpy()
        for i, sample in enumerate(samples):
            months = sample["months"][:-1]
            observed = sample["val"][months]
            filled = np.where(np.isnan(observed), p_val[i, months], observed)
            s = visits[visits.RID == sample["RID"]].iloc[:-1]
            frame = pd.DataFrame(filled * std + mean, columns=fields)
            frame["RID"] = sample["RID"]
            categories = sample["cat"][months]
            categories = np.where(np.isnan(categories), p_cat[i, months], categories)
            frame["dx_num"] = categories.argmax(-1)
            frame["months_to_pred"] = s.months_to_pred.to_numpy()
            frames.append(frame)
        print(f"Imputed fold {fold}: {len(samples)} histories", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output, index=False)
    return output


def attach_histories(cohort, history_path):
    from .data import read_histories
    import json

    cohort = Path(cohort)
    visits, subjects, _ = load_cohort(cohort)
    histories = read_histories(history_path, visits)
    targets = subject_table(visits).set_index("RID").label.to_dict()
    eligible = []
    for rid, s in histories.groupby("RID", sort=False):
        if np.all(np.diff(np.append(s.dx_num.to_numpy(), targets[rid])) >= 0):
            eligible.append(rid)
    subjects["classifier_eligible"] = subjects.RID.isin(eligible).astype(int)
    classifier_subject_table(histories, subjects).to_csv(
        cohort / "classifier_subjects.csv", index=False
    )
    histories.to_csv(cohort / "histories.csv", index=False)
    subjects.to_csv(cohort / "subjects.csv", index=False)
    metadata_path = cohort / "dataset.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["classifier_excluded"] = int((subjects.classifier_eligible == 0).sum())
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
