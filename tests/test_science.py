import numpy as np
import pandas as pd
import pytest
import torch

from transformer_visit_sequences.data import (
    assign_folds,
    balance_subjects,
    classifier_arrays,
    load_cohort,
    prepare,
    read_visits,
)
from transformer_visit_sequences.features import MEASURES, select_features
from transformer_visit_sequences.metrics import classification_metrics, score_predictions
from transformer_visit_sequences.models import ModelFillingRNN, SequenceClassifierTransformer
from transformer_visit_sequences.train import monthly_arrays


def test_metrics_perfect_and_chance():
    y = np.array([0, 1, 2, 0, 1, 2])
    perfect = classification_metrics(y, np.eye(3)[y])
    assert all(value == 1 for value in perfect.values())
    chance = classification_metrics(y, np.full((6, 3), 1 / 3))
    assert chance["mauc"] == 0.5
    assert chance["bca"] == 0.5


def test_metrics_absent_class_and_invalid_probabilities():
    result = classification_metrics([1, 2], [[0, 1, 0], [0, 0, 1]])
    assert result["sensitivity_cn"] == 0.5
    assert result["specificity_cn"] == 1
    assert result["bca"] == pytest.approx(11 / 12)
    assert result["mauc"] == pytest.approx(1 / 3)
    for p in ([[0, 0, 0]], [[np.nan, 0, 1]], [[-1, 1, 1]]):
        with pytest.raises(ValueError):
            classification_metrics([0], p)


def test_duplicate_predictions_rejected():
    frame = pd.DataFrame(
        dict(
            RID=[1, 1],
            fold=[0, 1],
            group=[1, 1],
            label=[0, 0],
            converter=[0, 0],
            p_cn=[1, 1],
            p_mci=[0, 0],
            p_ad=[0, 0],
        )
    )
    with pytest.raises(ValueError, match="exactly one"):
        score_predictions(frame)


def test_paper_parameter_count():
    assert sum(p.numel() for p in SequenceClassifierTransformer().parameters()) == 6_603_267


def test_subject_splits_and_balancing(small_cohort):
    _, subjects, _ = load_cohort(small_cohort)
    assert len(subjects) == subjects.RID.nunique()
    folds = assign_folds(subjects, 3, 0)
    np.testing.assert_array_equal(folds, assign_folds(subjects, 3, 0))
    for _, group in subjects.assign(test_fold=folds).groupby("group"):
        assert group.test_fold.value_counts().max() - group.test_fold.value_counts().min() <= 1
    balanced = balance_subjects(subjects)
    for _, group in balanced.groupby("group"):
        assert (group.converter == 0).sum() == (group.converter == 1).sum()


def test_target_is_excluded_from_monthly_inputs(small_cohort):
    visits, subjects, _ = load_cohort(small_cohort)
    rid = subjects.RID.iloc[0]
    fields = list(MEASURES)
    original = monthly_arrays(
        visits, [rid], fields, np.zeros(22), np.ones(22), observed_target=False
    )
    changed = visits.copy()
    target = changed[changed.RID == rid].index[-1]
    changed.loc[target, fields] = 100000
    changed.loc[target, "rnn_diagnosis"] = 2
    altered = monthly_arrays(
        changed, [rid], fields, np.zeros(22), np.ones(22), observed_target=False
    )
    np.testing.assert_array_equal(original[0]["val"], altered[0]["val"])
    np.testing.assert_array_equal(original[0]["cat"], altered[0]["cat"])
    assert np.isnan(original[0]["val"][-1]).all()


def test_classifier_ablation_preserves_labels_and_original_groups(small_cohort):
    _, subjects, histories = load_cohort(small_cohort)
    x, mask, meta = classifier_arrays(histories, subjects, select_features(), history_limit=1)
    assert x.shape == (len(subjects), 8, 23)
    assert (mask.sum(1) == 1).all()
    assert set(meta.group) == set(subjects.group)
    assert len(select_features(remove="cognitive")) == 11
    assert len(select_features(only="cognitive")) == 11


def test_filtering_discards_reverters_and_stops_at_first_conversion(small_cohort, tmp_path):
    visits, _, _ = load_cohort(small_cohort)
    source = visits[visits.RID.isin([1, 2, 3])].drop(columns=["diagnosis", "rnn_diagnosis"])
    indices = source[source.RID == 1].index
    source.loc[indices[-1], "DXCHANGE"] = 7
    indices = source[source.RID == 2].index
    source.loc[indices, "DXCHANGE"] = [1, 4, 2, 5]
    csv = tmp_path / "visits.csv"
    source.to_csv(csv, index=False)
    filtered = read_visits(csv)
    assert 1 not in set(filtered.RID)
    assert len(filtered[filtered.RID == 2]) == 2
    with pytest.raises(ValueError, match="empty"):
        prepare(csv, small_cohort)


@pytest.mark.parametrize("kind", ["minrnn", "lstm", "gru"])
def test_monthly_feedback_is_finite_and_differentiable(kind):
    model = ModelFillingRNN(kind=kind, nb_measures=2, h_size=4, nb_layers=1, h_drop=0, i_drop=0)
    cat = torch.full((2, 5, 3), float("nan"))
    val = torch.full((2, 5, 2), float("nan"))
    cat[:, 0] = torch.tensor([1.0, 0.0, 0.0])
    val[:, 0] = 0
    p, measurements = model(cat, val)
    assert torch.isfinite(p).all() and torch.isfinite(measurements).all()
    (p.sum() + measurements.sum()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_final_architecture_and_recorded_paper_statistics():
    from pathlib import Path
    import transformer_visit_sequences
    from transformer_visit_sequences.models import TimeSeriesTransformer
    from transformer_visit_sequences.report import paper_summary

    assert (
        sum(p.numel() for p in SequenceClassifierTransformer(input_size=24).parameters())
        == 6_603_523
    )
    assert sum(p.numel() for p in TimeSeriesTransformer().parameters()) == 1_330_947
    source = Path(transformer_visit_sequences.__file__).with_name("paper_folds.csv")
    summary = paper_summary(pd.read_csv(source))
    for model, expected in [
        ("transformer", 0.87362902138612),
        ("tst", 0.8664336303431973),
    ]:
        row = summary[(summary.experiment == f"raw/{model}") & (summary.subset == "Overall")]
        assert row.bca_mean.iloc[0] == pytest.approx(expected, abs=1e-12)

    row = summary[(summary.experiment == "raw/gru") & (summary.subset == "Overall")]
    assert row.bca_mean.iloc[0] == pytest.approx(0.830, abs=0.001)
