"""Attention classifiers and monthly model-filling recurrent networks."""

import math

import torch
from torch import nn


class SequenceClassifierTransformer(nn.Module):
    def __init__(
        self,
        input_size=23,
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=8,
        dim_feedforward=128,
        dropout=0.5,
        hidden_size=256,
        max_seq_length=8,
        mask_mode="additive",
    ):
        super().__init__()
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=False,
        )
        self.fc1 = nn.Linear(d_model * max_seq_length, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 3)

    def forward(self, src, valid):
        src = self.input_fc(src).transpose(0, 1)
        if self.mask_mode == "additive":
            output = self.transformer(src, src, src_key_padding_mask=valid.to(src.dtype))
        elif self.mask_mode == "padding":
            padding = ~valid.bool()
            output = self.transformer(
                src,
                src,
                src_key_padding_mask=padding,
                tgt_key_padding_mask=padding,
                memory_key_padding_mask=padding,
            )
        else:
            raise ValueError("mask_mode must be additive or padding.")
        output = output.transpose(0, 1).contiguous().flatten(1)
        return self.fc2(self.relu(self.fc1(output)))


class MinimalRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, hidden_size)
        self.weight_uh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_uz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.ones(hidden_size))
        bound = 1 / math.sqrt(hidden_size)
        nn.init.uniform_(self.weight_uh, -bound, bound)
        nn.init.uniform_(self.weight_uz, -bound, bound)

    def forward(self, x, h):
        z = torch.tanh(self.W(x))
        u = torch.sigmoid(h @ self.weight_uh + z @ self.weight_uz + self.bias_hh)
        return u * h + (1 - u) * z


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 3 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size)
        bound = 1 / math.sqrt(hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, x, h):
        xr, xu, xn = self.x2h(x).chunk(3, -1)
        hr, hu, hn = self.h2h(h).chunk(3, -1)
        candidate = torch.tanh(xn + torch.sigmoid(xr + hr) * hn)
        update = torch.sigmoid(xu + hu)
        return candidate + update * (h - candidate)


class ModelFillingRNN(nn.Module):
    def __init__(
        self, kind="minrnn", nb_measures=22, h_size=128, nb_layers=2, h_drop=0.4, i_drop=0.1
    ):
        super().__init__()
        if kind not in ("minrnn", "gru", "lstm"):
            raise ValueError("Recurrent model must be minrnn, gru, or lstm.")
        self.kind, self.h_size = kind, h_size
        self.h_drop, self.i_drop = h_drop, i_drop
        self.hid2category = nn.Linear(h_size, 3)
        self.hid2measures = nn.Linear(h_size, nb_measures)
        cell = {"minrnn": MinimalRNNCell, "gru": GRUCell, "lstm": nn.LSTMCell}[kind]
        self.cells = nn.ModuleList(
            [cell(3 + nb_measures if i == 0 else h_size, h_size) for i in range(nb_layers)]
        )
        if kind == "lstm":
            for layer in self.cells:
                nn.init.ones_(layer.bias_hh[h_size : 2 * h_size])

    def forward(self, categories, values, *, return_filled=False):
        """Predict month t+1 and feed detached predictions into missing observations."""
        batch, length, measures = values.shape
        hidden = [values.new_zeros(batch, self.h_size) for _ in self.cells]
        memory = [torch.zeros_like(h) for h in hidden]
        input_mask = values.new_ones(batch, measures)
        hidden_masks = [torch.ones_like(h) for h in hidden]
        if self.training:
            input_mask.bernoulli_(1 - self.i_drop)
            for mask in hidden_masks:
                mask.bernoulli_(1 - self.h_drop)
        if not torch.isfinite(values[:, 0]).all() or not torch.isfinite(categories[:, 0]).all():
            raise ValueError("The first monthly input must be initialized.")
        cat, val = categories[:, 0], values[:, 0]
        out_cat, out_val, filled = [], [], [val]
        for t in range(length - 1):
            h = torch.cat((cat, val * input_mask), dim=-1)
            next_hidden = []
            for i, cell in enumerate(self.cells):
                previous = hidden[i] * hidden_masks[i]
                if self.kind == "lstm":
                    h, memory[i] = cell(h, (previous, memory[i]))
                else:
                    h = cell(h, previous)
                next_hidden.append(h)
            hidden = next_hidden
            p_cat = torch.softmax(self.hid2category(h), dim=-1)
            p_val = self.hid2measures(h) + val
            out_cat.append(p_cat)
            out_val.append(p_val)
            cat = torch.where(
                torch.isnan(categories[:, t + 1]), p_cat.detach(), categories[:, t + 1]
            )
            val = torch.where(torch.isnan(values[:, t + 1]), p_val.detach(), values[:, t + 1])
            filled.append(val)
        result = torch.stack(out_cat, 1), torch.stack(out_val, 1)
        return (*result, torch.stack(filled, 1)) if return_filled else result


class TimeSeriesBlock(nn.Module):
    def __init__(self, width, heads, feedforward, dropout):
        super().__init__()
        self.heads, self.depth = heads, width // heads
        if width % heads:
            raise ValueError("TST width must be divisible by attention heads.")
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, width),
        )
        self.attention_norm = nn.BatchNorm1d(width)
        self.feedforward_norm = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = [
            layer(x).reshape(batch, length, self.heads, self.depth).transpose(1, 2)
            for layer in (self.query, self.key, self.value)
        ]
        attention = (q @ k.transpose(-2, -1) / math.sqrt(self.depth)).softmax(-1)
        context = (attention @ v).transpose(1, 2).reshape(batch, length, width)
        x = x + self.dropout(self.projection(context))
        x = self.attention_norm(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.dropout(self.feedforward(x))
        return self.feedforward_norm(x.transpose(1, 2)).transpose(1, 2)


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_size=24,
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        dim_feedforward=128,
        dropout=0.5,
        fc_dropout=0.1,
        max_seq_length=8,
    ):
        super().__init__()
        self.input = nn.Linear(input_size, d_model)
        self.position = nn.Parameter(torch.empty(max_seq_length, d_model))
        nn.init.uniform_(self.position, -0.02, 0.02)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TimeSeriesBlock(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.GELU(), nn.Flatten(), nn.Dropout(fc_dropout), nn.Linear(d_model * max_seq_length, 3)
        )

    def forward(self, x, valid=None):
        x = self.dropout(self.input(x) + self.position[: x.shape[1]])
        for layer in self.layers:
            x = layer(x)
        return self.head(x.transpose(1, 2).contiguous())


def classifier(name, input_size, settings):
    if name == "transformer":
        return SequenceClassifierTransformer(input_size=input_size, **settings)
    if name == "tst":
        return TimeSeriesTransformer(input_size=input_size, **settings)
    raise ValueError(f"Unknown classifier: {name}")
