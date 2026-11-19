import json

import numpy as np
import pandas as pd
import pytest
import torch

from transformer_visit_sequences.data import load_cohort, classifier_arrays
from transformer_visit_sequences.impute import export_histories, attach_histories
from transformer_visit_sequences.models import classifier
from transformer_visit_sequences.report import collect_runs
from transformer_visit_sequences.train import run_experiment


@pytest.mark.parametrize("model", ["transformer", "tst", "minrnn", "lstm", "gru"])
def test_training_checkpoint_predictions_and_resume(small_cohort, small_config, tmp_path, model):
    output = tmp_path / "runs" / model
    result = run_experiment(small_cohort, output, model, small_config, fold=0, device="cpu")
    assert len(result[result.group == 0]) > 0
    pred = pd.read_csv(output / "predictions.csv")
    assert not pred.RID.duplicated().any()
    assert np.isfinite(pred[["p_cn", "p_mci", "p_ad"]]).all().all()
    losses = pd.read_csv(output / "fold_0/loss.csv")
    assert np.isfinite(losses.train_loss).all()
    split = pd.read_csv(output / "fold_0/split.csv")
    assert set(split[split.role == "test"].RID).isdisjoint(split[split.role == "train"].RID)
    if model in ("transformer", "tst"):
        checkpoint = torch.load(output / "fold_0/model.pt", weights_only=True)
        restored = classifier(model, len(checkpoint["features"]), checkpoint["model_settings"])
        restored.load_state_dict(checkpoint["state_dict"])
        restored.eval()
        _, subjects, histories = load_cohort(small_cohort)
        x, masks, meta = classifier_arrays(histories, subjects, checkpoint["features"])
        idx = [int(meta.index[meta.RID == rid][0]) for rid in pred.RID]
        with torch.no_grad():
            probabilities = restored(torch.tensor(x[idx]), torch.tensor(masks[idx])).softmax(-1)
        np.testing.assert_allclose(
            probabilities.numpy(), pred[["p_cn", "p_mci", "p_ad"]], atol=1e-7
        )
    before = (output / "fold_0/model.pt").stat().st_mtime_ns
    run_experiment(small_cohort, output, model, small_config, fold=0, device="cpu")
    assert before == (output / "fold_0/model.pt").stat().st_mtime_ns
    with pytest.raises(ValueError, match="different configuration"):
        run_experiment(small_cohort, output, model, small_config, fold=0, epochs=9, device="cpu")


def test_full_imputation_and_report(small_cohort, small_config, tmp_path):
    output = tmp_path / "runs/raw/minrnn"
    run_experiment(small_cohort, output, "minrnn", small_config, device="cpu")
    target = tmp_path / "imputed.csv"
    export_histories(small_cohort, output, target, "cpu")
    attach_histories(small_cohort, target)
    _, subjects, histories = load_cohort(small_cohort)
    assert len(histories) == subjects.group.sum()
    assert np.isfinite(histories.to_numpy()).all()
    report = tmp_path / "report"
    collect_runs(tmp_path / "runs", report, plots=False)
    assert (report / "report.md").exists()
    assert json.loads((output / "run.json").read_text())["model"] == "minrnn"


def test_publication_plots(tmp_path):
    pytest.importorskip("matplotlib")
    from transformer_visit_sequences.report import plot_groups

    summary = pd.DataFrame(
        dict(
            group=[1, 2, 1, 2, 1, 2],
            subset=["Overall"] * 2 + ["Stable"] * 2 + ["Converter"] * 2,
            experiment=["raw/transformer"] * 6,
            bca_mean=[0.6, 0.7] * 3,
            bca_std=[0.02, 0.03] * 3,
        )
    )
    plot_groups(summary, tmp_path)
    assert (tmp_path / "overall_groups.pdf").stat().st_size > 1000
