"""R6: pronóstico de juegos hasta el siguiente cruce observado."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import GAMES_DATASET, MOMENTUM_DATASET, PROCESSED_DIR, PROJECT_ROOT, RANDOM_SEED
from .crossings import momentum_crossings

from .crossing_model import HISTORY, INPUTS, MODEL_PATH


class CrossingGRU(nn.Module):
    """Regresión a partir de la secuencia de los últimos cinco juegos."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(len(INPUTS), 32, batch_first=True)
        self.output = nn.Linear(32, 1)

    def forward(self, history):
        sequence, _ = self.gru(history)
        return 1.0 + nn.functional.softplus(self.output(sequence[:, -1]).squeeze(-1))


def build_examples():
    momentum = pd.read_parquet(MOMENTUM_DATASET)
    games = pd.read_parquet(GAMES_DATASET, columns=["match_id", "game_num", "date"])
    data = momentum.merge(games, on=["match_id", "game_num"], validate="one_to_one")
    data = data.sort_values(["match_id", "game_num"])
    inputs, targets, years = [], [], []
    for _, match in data.groupby("match_id", sort=False):
        match = match.set_index("game_num")
        crossings = momentum_crossings(match["N"]).index.to_numpy()
        values = match[INPUTS].to_numpy(dtype=np.float32)
        year = pd.Timestamp(match["date"].iloc[0]).year
        for end in range(HISTORY - 1, len(match) - 1):
            game = match.index[end]
            next_position = np.searchsorted(crossings, game, side="right")
            target = crossings[next_position] - game if next_position < len(crossings) else np.nan
            inputs.append(values[end - HISTORY + 1:end + 1])
            targets.append(target)
            years.append(year)
    return np.asarray(inputs), np.asarray(targets, dtype=np.float32), np.asarray(years)


def main():
    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(2)
    inputs, targets, years = build_examples()
    observed = np.isfinite(targets)
    periods = {"train": years < 2022, "validation": years == 2022, "test": years >= 2023}
    partitions = {
        name: (torch.from_numpy(inputs[mask & observed]), torch.from_numpy(targets[mask & observed]))
        for name, mask in periods.items()
    }
    train_x, train_y = partitions["train"]
    validation_x, validation_y = partitions["validation"]
    test_x, test_y = partitions["test"]
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=256, shuffle=True)
    model = CrossingGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss = nn.L1Loss()
    losses = {"train": [], "validation": []}
    best_validation = float("inf")
    for epoch in range(20):
        model.train()
        for history, remaining in loader:
            optimizer.zero_grad()
            loss(model(history), remaining).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_mae = loss(model(train_x), train_y).item()
            validation_mae = loss(model(validation_x), validation_y).item()
        losses["train"].append(train_mae)
        losses["validation"].append(validation_mae)
        if validation_mae < best_validation:
            best_validation = validation_mae
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        test_mae = loss(model(test_x), test_y).item()
        baseline = float(np.median(train_y.numpy()))
        baseline_mae = loss(torch.full_like(test_y, baseline), test_y).item()
    results = {
        "history_games": HISTORY,
        "seed": RANDOM_SEED,
        "splits": {
            name: {"eligible": int(mask.sum()), "with_observed_crossing": len(partitions[name][1])}
            for name, mask in periods.items()
        },
        "selected_epoch": best_epoch + 1,
        "losses": losses,
        "test_mae": test_mae,
        "baseline_train_median": baseline,
        "baseline_test_mae": baseline_mae,
        "relative_loss_gap": abs(best_validation - losses["train"][best_epoch])
        / losses["train"][best_epoch],
    }
    torch.save(best_state, MODEL_PATH)
    (PROCESSED_DIR / "r6_results.json").write_text(json.dumps(results, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in losses.items():
        ax.plot(range(1, len(values) + 1), values, label=name)
    ax.set(xlabel="Época", ylabel="MAE (juegos)", title="R6 · Pérdida de la GRU")
    ax.legend()
    fig.tight_layout()
    path = PROJECT_ROOT / "docs_tesis" / "figures" / "14_r6_training.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return results


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
