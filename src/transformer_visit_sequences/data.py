"""Cohort preparation, visit histories, and reproducible subject splits."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import DXCHANGE, MEASURES


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(series):
    return pd.to_numeric(series.astype(str).str.strip().str.lstrip("<>"), errors="coerce")


def read_visits(source, curated=False):
    wanted = {"RID", "EXAMDATE", "Month_bl", "DX", "DXCHANGE", "group_no", *MEASURES}
    data = pd.read_csv(source, usecols=lambda c: c in wanted, low_memory=False)
    required = {"RID", "EXAMDATE", "DXCHANGE", *MEASURES}
    if missing := required - set(data):
        raise ValueError(f"Visit columns missing: {sorted(missing)}")
    data["EXAMDATE"] = pd.to_datetime(data.EXAMDATE, errors="raise")
    data["RID"] = numeric(data.RID)
    if data.RID.isna().any() or (data.RID % 1 != 0).any():
        raise ValueError("RID must contain integer subject identifiers.")
    data["RID"] = data.RID.astype(int)
    for name in MEASURES:
        data[name] = numeric(data[name]).replace([np.inf, -np.inf], np.nan)
    data["DXCHANGE"] = numeric(data.DXCHANGE)
    data["diagnosis"] = data.DXCHANGE.map(DXCHANGE)
    if "DX" in data:
        data["rnn_diagnosis"] = (
            data.DX.astype(str)
            .str.split(" to ")
            .str[-1]
            .map({"NL": 0, "CN": 0, "MCI": 1, "Dementia": 2, "AD": 2})
        )
    else:
        data["rnn_diagnosis"] = data.diagnosis
    frames = []
    for _, subject in data.groupby("RID", sort=False):
        subject = subject.sort_values("EXAMDATE", kind="stable").copy()
        if not curated and (
            subject.DXCHANGE.isin([7, 8, 9]).any() or (subject.diagnosis.dropna().diff() < 0).any()
        ):
            continue
        subject = subject.dropna(subset=["diagnosis"])
        if not curated:
            conversion = np.flatnonzero(subject.diagnosis.diff().to_numpy() > 0)
            if len(conversion):
                subject = subject.iloc[: conversion[0] + 1].copy()
        if len(subject) < 2:
            continue
        if subject.EXAMDATE.duplicated().any():
            raise ValueError("A subject has duplicate examination dates.")
        if "Month_bl" not in subject:
            days = (subject.EXAMDATE - subject.EXAMDATE.iloc[0]).dt.days
            subject["Month_bl"] = days / (365.25 / 12)
        subject["Month_bl"] = numeric(subject.Month_bl)
        months = subject.Month_bl.round().to_numpy()
        if not np.isfinite(months).all() or np.any(np.diff(months) <= 0):
            raise ValueError("Visit months must be distinct, finite, and increasing per subject.")
        if curated and "group_no" in subject:
            if not (numeric(subject.group_no) == len(subject) - 1).all():
                raise ValueError("group_no must equal the number of historical visits.")
        subject["group_no"] = len(subject) - 1
        date = subject.EXAMDATE
        end = date.iloc[-1]
        subject["months_to_pred"] = (end.year - date.dt.year) * 12 + end.month - date.dt.month
        frames.append(subject)
    if not frames:
        raise ValueError("No eligible subjects remain after cohort selection.")
    return pd.concat(frames, ignore_index=True)


def subject_table(visits):
    rows = []
    for rid, s in visits.groupby("RID", sort=False):
        dx = s.diagnosis.to_numpy(dtype=int)
        rows.append(
            dict(
                RID=int(rid),
                group=len(s) - 1,
                label=int(dx[-1]),
                converter=int(s.DXCHANGE.iloc[-1] in (4, 5, 6)),
                previous=int(dx[-2]),
            )
        )
    return pd.DataFrame(rows)


def balance_subjects(subjects, seed=0):
    rng = np.random.RandomState(seed)
    selected = []
    for _, group in subjects.groupby("group", sort=True):
        stable = group[group.converter == 0].RID.to_numpy()
        converters = group[group.converter == 1].RID.to_numpy()
        if len(stable) < len(converters):
            raise ValueError("Balancing requires at least as many stable subjects as converters.")
        selected.extend(converters.tolist())
        selected.extend(rng.choice(stable, len(converters), replace=False).tolist())
    if not selected:
        raise ValueError("No converter sequences are available for balancing.")
    return subjects[subjects.RID.isin(selected)].copy()


def assign_folds(subjects, folds=10, seed=0, grouped=True):
    if folds < 2 or len(subjects) < folds:
        raise ValueError("At least two folds and one subject per fold are required.")
    rng = np.random.RandomState(seed)
    assignment = {}
    groups = subjects.groupby("group", sort=True) if grouped else [(0, subjects)]
    for _, group in groups:
        rids = np.sort(group.RID.to_numpy()) if grouped else group.RID.to_numpy()
        permuted = rng.permutation(rids)
        for fold, chunk in enumerate(np.array_split(permuted, folds)):
            assignment.update({int(rid): fold for rid in chunk})
    return subjects.RID.map(assignment).to_numpy(dtype=int)


def read_histories(source, visits):
    """Read finite, imputed historical visits from one table or per-subject tables."""
    source = Path(source)
    if source.is_file():
        histories = pd.read_csv(source)
    else:
        files = sorted(source.glob("fold_*/not_normalized/*/*.csv"))
        if not files:
            raise ValueError("No history files matched fold_*/not_normalized/*/*.csv.")
        histories = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    required = {"RID", "dx_num", *MEASURES}
    if missing := required - set(histories):
        raise ValueError(f"History columns missing: {sorted(missing)}")
    histories["RID"] = numeric(histories.RID).astype(int)
    counts = visits.groupby("RID", sort=False).size() - 1
    if "Future_DX" not in histories:
        histories = histories[histories.RID.isin(counts.index)].copy()
        if set(histories.RID) != set(counts.index):
            raise ValueError("Every subject requires a history or an explicit Future_DX label.")
    if "months_to_pred" not in histories:
        timing = {
            int(rid): s.months_to_pred.iloc[:-1].to_numpy()
            for rid, s in visits.groupby("RID", sort=False)
        }
        histories["months_to_pred"] = np.concatenate(
            [timing[int(rid)] for rid in histories.RID.drop_duplicates()]
        )
    features = [*MEASURES, "months_to_pred", "dx_num"]
    for name in features:
        histories[name] = numeric(histories[name])
    if not np.isfinite(histories[features].to_numpy()).all():
        raise ValueError("Imputed history features must all be finite.")
    if not histories.dx_num.isin([0, 1, 2]).all():
        raise ValueError("Historical diagnoses must be 0, 1, or 2.")
    if "Future_DX" in histories:
        targets = visits.groupby("RID", sort=False).diagnosis.last()
        mapped = histories.RID.map(targets)
        if not (
            histories.loc[mapped.notna(), "Future_DX"].to_numpy() == mapped.dropna().to_numpy()
        ).all():
            raise ValueError("Future_DX disagrees with the final visit diagnosis.")
    histories = pd.concat(
        [
            s.sort_values("months_to_pred", ascending=False, kind="stable")
            for _, s in histories.groupby("RID", sort=False)
        ],
        ignore_index=True,
    )
    return histories[["RID", *features, *(["Future_DX"] if "Future_DX" in histories else [])]]


def classifier_subject_table(history, subjects, converter_rids=None):
    known = subjects.set_index("RID")
    rows = []
    for rid, s in history.groupby("RID", sort=False):
        if "Future_DX" in s:
            if s.Future_DX.nunique() != 1 or not s.Future_DX.isin([0, 1, 2]).all():
                raise ValueError("Future_DX must be a constant diagnosis for each subject.")
            label = int(s.Future_DX.iloc[0])
        else:
            label = int(known.loc[rid, "label"])
        previous = int(s.dx_num.iloc[-1])
        converter = (
            int(rid in converter_rids)
            if converter_rids is not None
            else (int(known.loc[rid, "converter"]) if rid in known.index else int(label > previous))
        )
        eligible = int(np.all(np.diff(np.append(s.dx_num.to_numpy(), label)) >= 0))
        rows.append(
            dict(
                RID=int(rid),
                group=len(s),
                label=label,
                previous=previous,
                converter=converter,
                classifier_eligible=eligible,
            )
        )
    return pd.DataFrame(rows)


def prepare(
    source,
    destination,
    *,
    curated=False,
    balanced=False,
    histories=None,
    folds=10,
    seed=0,
    fold_dir=None,
    converter_rids=None,
):
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose an empty preparation directory.")
    visits = read_visits(source, curated=curated)
    subjects = subject_table(visits)
    if balanced:
        subjects = balance_subjects(subjects, seed)
        visits = visits[visits.RID.isin(subjects.RID)].copy()
    subjects["fold"] = assign_folds(subjects, folds, seed)
    if fold_dir:
        assignment = {}
        for fold in range(folds):
            f = pd.read_csv(Path(fold_dir) / f"fold{fold}_group_0_test.csv", usecols=["RID"])
            for rid in f.RID:
                if int(rid) in assignment:
                    raise ValueError("A subject appears in multiple test folds.")
                assignment[int(rid)] = fold
        if set(assignment) != set(subjects.RID):
            raise ValueError("Fold membership must cover exactly the prepared subjects.")
        subjects["fold"] = subjects.RID.map(assignment)
    history = read_histories(histories, visits) if histories else None
    class_subjects = None
    excluded = 0
    if history is not None:
        converter_ids = None
        if converter_rids:
            converter_ids = {int(x) for x in Path(converter_rids).read_text().split()}
        class_subjects = classifier_subject_table(history, subjects, converter_ids)
        excluded = int((class_subjects.classifier_eligible == 0).sum())
        eligible = class_subjects.set_index("RID").classifier_eligible
        subjects["classifier_eligible"] = subjects.RID.map(eligible).fillna(0).astype(int)
    else:
        subjects["classifier_eligible"] = 1
    destination.mkdir(parents=True, exist_ok=True)
    visits.to_csv(destination / "visits.csv", index=False)
    subjects.to_csv(destination / "subjects.csv", index=False)
    if history is not None:
        history.to_csv(destination / "histories.csv", index=False)
        class_subjects.to_csv(destination / "classifier_subjects.csv", index=False)
    metadata = dict(
        source_sha256=sha256(source),
        curated=curated,
        balanced=balanced,
        folds=folds,
        seed=seed,
        subjects=len(subjects),
        visits=len(visits),
        stable=int((subjects.converter == 0).sum()),
        converters=int(subjects.converter.sum()),
        classifier_excluded=excluded,
        classifier_subjects=len(class_subjects) if class_subjects is not None else 0,
        group_counts={str(k): int(v) for k, v in subjects.groupby("group").size().items()},
    )
    (destination / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_cohort(directory):
    directory = Path(directory)
    visits = pd.read_csv(directory / "visits.csv", parse_dates=["EXAMDATE"])
    subjects = pd.read_csv(directory / "subjects.csv")
    history = (
        pd.read_csv(directory / "histories.csv") if (directory / "histories.csv").exists() else None
    )
    return visits, subjects, history


def classifier_folds(subjects, folds=10, seed=42):
    rng = np.random.RandomState(seed)
    assignment = {}
    for _, group in subjects.groupby("group", sort=False):
        rids = group.RID.to_numpy().copy()
        rng.shuffle(rids)
        assignment.update({int(rid): i % folds for i, rid in enumerate(rids)})
    return subjects.RID.map(assignment).to_numpy(dtype=int)


def classifier_arrays(history, subjects, features, max_length=8, history_limit=None):
    grouped = dict(tuple(history.groupby("RID", sort=False)))
    x, masks, rows = [], [], []
    for rid in history.RID.drop_duplicates():
        meta = subjects[subjects.RID == rid]
        if meta.empty:
            continue
        frame = grouped[rid]
        if history_limit:
            frame = frame.iloc[-history_limit:]
        diagnoses = np.append(frame.dx_num.to_numpy(), meta.label.iloc[0])
        if np.any(np.diff(diagnoses) < 0):
            continue
        seq = frame[features].to_numpy(dtype=np.float32)
        if len(seq) > max_length:
            raise ValueError(f"History exceeds max_length={max_length}.")
        padded = np.zeros((max_length, len(features)), dtype=np.float32)
        padded[: len(seq)] = seq
        lo, hi = padded.min(0), padded.max(0)
        span = hi - lo
        padded = (padded - lo) / np.where(span == 0, 1, span)
        x.append(padded)
        masks.append(np.arange(max_length) < len(seq))
        rows.append(meta.iloc[0].to_dict())
    if not rows:
        raise ValueError("No eligible classifier sequences.")
    return np.stack(x), np.stack(masks), pd.DataFrame(rows)
