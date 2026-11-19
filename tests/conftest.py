from copy import deepcopy
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import pytest

from transformer_visit_sequences.data import prepare
from transformer_visit_sequences.features import MEASURES


@pytest.fixture
def small_cohort(tmp_path):
    visits, histories = [], []
    rng = np.random.default_rng(11)
    for rid in range(1, 49):
        n = 2 + rid % 3
        baseline = rid % 2
        converter = rid % 4 < 2
        for i in range(n):
            target = baseline + int(converter and i == n - 1)
            code = {0: 1, 1: 2, 2: 3}[target]
            if converter and i == n - 1:
                code = 4 if baseline == 0 else 5
            values = rng.uniform(1, 10, len(MEASURES))
            row = dict(
                RID=rid,
                EXAMDATE=str(pd.Timestamp("2020-01-01") + pd.DateOffset(months=6 * i))[:10],
                Month_bl=6 * i,
                DXCHANGE=code,
                DX=["NL", "MCI", "Dementia"][target],
                group_no=n - 1,
            )
            row.update(zip(MEASURES, values, strict=True))
            visits.append(row)
            if i < n - 1:
                histories.append(
                    dict(
                        RID=rid,
                        dx_num=baseline,
                        months_to_pred=6 * (n - 1 - i),
                        **dict(zip(MEASURES, values, strict=True)),
                    )
                )
    source = tmp_path / "source.csv"
    pd.DataFrame(visits).to_csv(source, index=False)
    history = tmp_path / "histories.csv"
    pd.DataFrame(histories).to_csv(history, index=False)
    destination = tmp_path / "cohort"
    prepare(source, destination, curated=True, histories=history, folds=3)
    return destination


@pytest.fixture
def small_config():
    source = Path(__file__).parents[1] / "src/transformer_visit_sequences/paper.toml"
    config = tomllib.loads(source.read_text())
    config = deepcopy(config)
    config["runtime"]["threads"] = 1
    config["runtime"]["include_time"] = False
    config["training"].update(folds=3, epochs=2, batch_size=16, selection_split="validation")
    config["transformer"].update(
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        hidden_size=16,
        dropout=0,
        max_seq_length=8,
    )
    config["tst"].update(d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=16, dropout=0)
    config["history_transformer"] = dict(config["transformer"])
    config["rnn"].update(epochs=1, batch_size=16, h_size=8, nb_layers=1, h_drop=0, i_drop=0)
    return config
