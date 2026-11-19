# Transformer Model for Alzheimer's Disease Progression Prediction Using Longitudinal Visit Sequences

Next-visit CN/MCI/AD prediction from longitudinal TADPOLE/ADNI visits. Includes Transformer, TST, LSTM, GRU, and minimalRNN models, raw and balanced cohorts, ten-fold evaluation, and the paper's feature/history ablations.

Paper: [arXiv:2507.03899](https://arxiv.org/abs/2507.03899).

Python 3.13 and uv are required:

~~~sh
uv sync --extra plots
uv run pytest -q
uv run visit-sequences --help
~~~

Recompute the paper's tables, statistical comparisons, and figures from the packaged fold-level results:

~~~sh
uv run visit-sequences paper --out results/paper --plots
~~~

This statistical reanalysis uses aggregate results and requires no participant data. New model training writes its own predictions and measured results.

The prepared participant cohorts are available locally in data/paper/raw and data/paper/balanced. Participant data, checkpoints, and generated runs are excluded from Git.

~~~sh
uv run visit-sequences run --cohort data/paper/raw --model transformer --out runs/raw/transformer
uv run visit-sequences suite --data data/paper --out runs/paper
uv run visit-sequences report runs/paper --out results/training --plots
~~~

Use --fold 0 --epochs 1 for a short execution check. Completed folds resume when the configuration, source, and data fingerprints agree. The [paper configuration](src/transformer_visit_sequences/paper.toml) monitors the held-out fold during early stopping; selection_split = "validation" creates a separate validation split.

For NVIDIA CUDA acceleration, install the PyTorch backend after syncing and preserve it with --no-sync:

~~~sh
uv pip install torch==2.9.1 --torch-backend cu128 --reinstall-package torch
uv run --no-sync visit-sequences suite --data data/paper --out runs/paper --device cuda
~~~

Runs contain subject splits, losses, state-dictionary checkpoints, out-of-fold probabilities, group/subset metrics, and measured comparisons with the published tables. Reports distinguish ordinary fold summaries from the paper's group-weighted RNN summaries.

Read the [experiment protocol](docs/protocol.md) for preparation and evaluation details and the [verification record](docs/verification.md) for executed checks and numerical results.
