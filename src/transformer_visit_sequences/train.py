"""Deterministic training, monthly forecasting, and out-of-fold prediction."""

from copy import deepcopy
import hashlib
import json
import os
import platform
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .data import classifier_arrays, classifier_folds, load_cohort, sha256
from .features import MEASURES, select_features
from .metrics import score_predictions, summarize
from .models import ModelFillingRNN, classifier


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name="auto"):
    if name == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return torch.device(name)


def monthly_arrays(visits, rids, fields, mean, std, *, observed_target=True):
    samples = []
    for rid in rids:
        s = visits[visits.RID == rid].sort_values("EXAMDATE")
        months = s.Month_bl.round().to_numpy(dtype=int)
        months -= months[0]
        length = int(months[-1] + 1)
        cat = np.full((length, 3), np.nan, dtype=np.float32)
        val = np.full((length, len(fields)), np.nan, dtype=np.float32)
        observed = s if observed_target else s.iloc[:-1]
        obs_months = months if observed_target else months[:-1]
        obs_values = ((observed[fields].to_numpy() - mean) / std).astype(np.float32)
        val[obs_months] = obs_values
        diagnoses = observed.rnn_diagnosis.to_numpy()
        good = np.isfinite(diagnoses)
        cat[obs_months[good]] = np.eye(3, dtype=np.float32)[diagnoses[good].astype(int)]
        truth_cat, truth_val = cat.copy(), val.copy()
        for j in range(len(fields)):
            valid = np.flatnonzero(np.isfinite(val[:, j]))
            val[0, j] = val[valid[0], j] if len(valid) else 0.0
        known_dx = np.flatnonzero(np.isfinite(cat).all(1))
        cat[0] = cat[known_dx[0]] if len(known_dx) else np.array([1, 0, 0])
        samples.append(
            dict(
                RID=int(rid),
                cat=cat,
                val=val,
                truth_cat=truth_cat,
                truth_val=truth_val,
                months=months,
            )
        )
    return samples


def monthly_batch(samples, device):
    length = max(len(s["cat"]) for s in samples)
    categories = np.full((len(samples), length, 3), np.nan, np.float32)
    values = np.full((len(samples), length, samples[0]["val"].shape[1]), np.nan, np.float32)
    cat_truth, val_truth = categories.copy(), values.copy()
    for i, sample in enumerate(samples):
        n = len(sample["cat"])
        categories[i, :n], values[i, :n] = sample["cat"], sample["val"]
        cat_truth[i, :n], val_truth[i, :n] = sample["truth_cat"], sample["truth_val"]
    return tuple(
        torch.as_tensor(a, device=device) for a in (categories, values, cat_truth, val_truth)
    )


def fit_rnn(visits, train_rids, settings, fields, device):
    settings = dict(settings)
    epochs = settings.pop("epochs")
    batch_size = settings.pop("batch_size")
    learning_rate = settings.pop("learning_rate")
    weight_decay = settings.pop("weight_decay")
    entropy_weight = settings.pop("entropy_weight", 1.0)
    values = visits[visits.RID.isin(train_rids)][fields]
    mean = values.mean().fillna(0).to_numpy()
    std = values.std(ddof=1).replace(0, 1).fillna(1).to_numpy()
    model = ModelFillingRNN(nb_measures=len(fields), **settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    samples = monthly_arrays(visits, train_rids, fields, mean, std)
    log = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        order = np.random.permutation(len(samples))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            cats, vals, true_cat, true_val = monthly_batch([samples[i] for i in idx], device)
            optimizer.zero_grad()
            p_cat, p_val = model(cats, vals)
            cat_mask = torch.isfinite(true_cat[:, 1:]).all(-1)
            val_mask = torch.isfinite(true_val[:, 1:])
            labels = torch.nan_to_num(true_cat[:, 1:]).argmax(-1)
            ent = (
                F.cross_entropy(p_cat[cat_mask], labels[cat_mask], reduction="sum")
                if cat_mask.any()
                else p_cat.sum() * 0
            )
            mae = (p_val - torch.nan_to_num(true_val[:, 1:])).abs()[val_mask].sum()
            loss = (mae + entropy_weight * ent) / len(idx)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite recurrent training loss.")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        log.append(dict(epoch=epoch + 1, train_loss=total / len(samples)))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}: loss={log[-1]['train_loss']:.5f}", flush=True)
    return model, mean, std, log


def split_validation(indices, fraction, seed):
    if len(indices) < 2:
        raise ValueError("Classifier training requires at least two non-test subjects.")
    order = np.random.RandomState(seed).permutation(len(indices))
    count = max(1, min(len(indices) - 1, int(np.ceil(len(indices) * fraction))))
    return indices[order[count:]], indices[order[:count]]


def fit_classifier(
    name, x, valid, labels, train_indices, settings, model_settings, device, test_indices=None
):
    settings = dict(settings)
    seed = settings["split_seed"]
    if settings.get("selection_split", "validation") == "test":
        train, validation = train_indices, test_indices
    else:
        train, validation = split_validation(train_indices, settings["validation_fraction"], seed)
    model = classifier(name, x.shape[-1], model_settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    best, best_state, waiting = float("inf"), None, 0
    log = []
    batch_size = settings["batch_size"]
    for epoch in range(settings["epochs"]):
        model.train()
        total = 0.0
        order = np.random.permutation(train)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(x[idx], valid[idx])
            loss = F.cross_entropy(logits, labels[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite classifier training loss.")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        model.eval()
        validation_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(validation), batch_size):
                idx = validation[start : start + batch_size]
                loss = F.cross_entropy(model(x[idx], valid[idx]), labels[idx])
                validation_sum += loss.item()
        # Equal-sized validation batches reproduce the experiment's stopping statistic.
        validation_loss = validation_sum / len(validation)
        log.append(dict(epoch=epoch + 1, train_loss=total / len(train), val_loss=validation_loss))
        if validation_loss <= best:
            best, waiting = validation_loss, 0
            if settings["checkpoint_selection"] == "best":
                best_state = deepcopy(model.state_dict())
        else:
            waiting += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"  epoch {epoch + 1}: loss={log[-1]['train_loss']:.5f}, "
                f"validation={validation_loss:.5f}",
                flush=True,
            )
        if waiting >= settings["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, log, train, validation


def run_experiment(
    cohort,
    output,
    name,
    config,
    *,
    fold=None,
    epochs=None,
    device="auto",
    history_limit=None,
    remove=None,
    only=None,
    include_time=None,
):
    output = Path(output)
    visits, subjects, history = load_cohort(cohort)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["runtime"].get("threads", 4))
    device = resolve_device(device)
    recurrent = name in ("minrnn", "lstm", "gru")
    if include_time is None:
        include_time = (
            config["rnn"].get("include_time", False)
            if recurrent
            else config["runtime"].get("include_time", True)
        )
    features = select_features(include_time, remove, only)
    model_settings = dict(config.get(name, {}))
    if history_limit:
        model_settings = dict(config["history_transformer"])
        if history_limit == 4:
            model_settings["max_seq_length"] = 4
    settings = dict(config["rnn" if recurrent else "training"])
    settings.pop("include_time", None)
    if epochs is not None:
        settings["epochs"] = epochs
    if settings["epochs"] < 1:
        raise ValueError("epochs must be positive.")
    if name != "transformer" and (remove or only or history_limit):
        raise ValueError("Feature and history ablations use the Transformer classifier.")
    if recurrent:
        settings["kind"] = name
        meta = subjects.copy()
        fields = [*MEASURES, *(["months_to_pred"] if include_time else [])]
    else:
        class_table = Path(cohort) / "classifier_subjects.csv"
        if class_table.exists():
            subjects = pd.read_csv(class_table)
        if history is None:
            raise ValueError("Prepare imputed histories before training the classifier.")
        x, valid, meta = classifier_arrays(
            history, subjects, features, model_settings["max_seq_length"], history_limit
        )
        split_meta = meta.assign(group=valid.sum(1)) if history_limit else meta
        meta["fold"] = classifier_folds(split_meta, settings["folds"], settings["split_seed"])
        x = torch.as_tensor(x, device=device)
        valid = torch.as_tensor(valid, device=device)
        labels = torch.as_tensor(meta.label.to_numpy(), dtype=torch.long, device=device)
    run_info = dict(
        model=name,
        settings=settings,
        model_settings=model_settings,
        features=features,
        history_limit=history_limit,
        seed=config["runtime"]["seed"],
        device=str(device),
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        pandas_version=pd.__version__,
        code_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        data_sha256={p.name: sha256(p) for p in Path(cohort).glob("*.csv")},
    )
    fingerprint = hashlib.sha256(json.dumps(run_info, sort_keys=True).encode()).hexdigest()
    manifest = output / "run.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Output belongs to a different configuration; choose a new directory.")
    manifest.write_text(json.dumps(dict(run_info, fingerprint=fingerprint), indent=2) + "\n")
    available_folds = sorted(meta.fold.unique())
    if fold is not None and fold not in available_folds:
        raise ValueError(f"Fold must be one of {available_folds}.")
    folds = available_folds if fold is None else [fold]
    for f in folds:
        fold_output = output / f"fold_{f}"
        if (fold_output / "complete.json").exists():
            print(f"{name} fold {f}: using completed fold", flush=True)
            continue
        fold_output.mkdir(exist_ok=True)
        seed_everything(config["runtime"]["seed"] + int(f))
        test = meta[meta.fold == f].copy()
        train = meta[meta.fold != f].copy()
        print(f"{name} fold {f}: {len(train)} train pool, {len(test)} test, {device}", flush=True)
        start = time.monotonic()
        if recurrent:
            model, mean, std, log = fit_rnn(visits, train.RID.to_numpy(), settings, fields, device)
            model.eval()
            samples = monthly_arrays(
                visits, test.RID.to_numpy(), fields, mean, std, observed_target=False
            )
            probabilities, histories = [], []
            with torch.no_grad():
                for sample in samples:
                    cats, vals, _, _ = monthly_batch([sample], device)
                    p_cat, _, filled = model(cats, vals, return_filled=True)
                    probabilities.append(p_cat[0, sample["months"][-1] - 1].cpu().numpy())
                    values = filled[0, sample["months"][:-1]].cpu().numpy() * std + mean
                    s = visits[visits.RID == sample["RID"]].iloc[:-1]
                    h = pd.DataFrame(values, columns=fields)
                    h["RID"] = sample["RID"]
                    h["dx_num"] = s.rnn_diagnosis.to_numpy()
                    h["months_to_pred"] = s.months_to_pred.to_numpy()
                    histories.append(h)
            pd.concat(histories, ignore_index=True).to_csv(
                fold_output / "histories.csv", index=False
            )
            checkpoint = dict(
                state_dict=model.cpu().state_dict(),
                model=name,
                settings=settings,
                mean=mean.tolist(),
                std=std.tolist(),
                fields=fields,
            )
            split = meta[["RID", "fold"]].copy()
            split["role"] = np.where(meta.fold == f, "test", "train")
        else:
            train_idx = np.flatnonzero(meta.fold.to_numpy() != f)
            test_idx = np.flatnonzero(meta.fold.to_numpy() == f)
            model, log, train_idx, val_idx = fit_classifier(
                name, x, valid, labels, train_idx, settings, model_settings, device, test_idx
            )
            model.eval()
            with torch.no_grad():
                probabilities = np.concatenate(
                    [
                        model(x[idx], valid[idx]).softmax(-1).cpu().numpy()
                        for idx in np.array_split(
                            test_idx, max(1, int(np.ceil(len(test_idx) / settings["batch_size"])))
                        )
                    ]
                )
            checkpoint = dict(
                state_dict=model.cpu().state_dict(),
                model=name,
                model_settings=model_settings,
                features=features,
            )
            split = meta[["RID", "fold"]].copy()
            split["role"] = "test"
            split.loc[train_idx, "role"] = "train"
            if settings.get("selection_split", "validation") != "test":
                split.loc[val_idx, "role"] = "validation"
            split["used_for_selection"] = split.index.isin(val_idx)
        predictions = test[["RID", "fold", "group", "label", "converter"]].copy()
        predictions[["p_cn", "p_mci", "p_ad"]] = np.asarray(probabilities)
        score_predictions(predictions)
        predictions.to_csv(fold_output / "predictions.csv", index=False)
        split.to_csv(fold_output / "split.csv", index=False)
        pd.DataFrame(log).to_csv(fold_output / "loss.csv", index=False)
        torch.save(checkpoint, fold_output / "model.pt")
        (fold_output / "complete.json").write_text(
            json.dumps(
                dict(epochs=len(log), seconds=time.monotonic() - start, subjects=len(test)),
                indent=2,
            )
        )
    predictions = pd.concat(
        [pd.read_csv(p) for p in sorted(output.glob("fold_*/predictions.csv"))], ignore_index=True
    )
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics = score_predictions(predictions)
    metrics["model"] = name
    metrics.to_csv(output / "metrics.csv", index=False)
    summarize(metrics).to_csv(output / "summary.csv", index=False)
    if recurrent:
        pd.concat(
            [pd.read_csv(p) for p in sorted(output.glob("fold_*/histories.csv"))], ignore_index=True
        ).to_csv(output / "histories.csv", index=False)
    return summarize(metrics)
