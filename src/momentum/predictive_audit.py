"""Evaluate the deployed R4/R5 models without changing their predictions."""

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from backend.paths import MODEL_BUNDLE
from config import PROCESSED_DIR, PROJECT_ROOT, RANDOM_SEED
from . import classification, regression


def persistence_comparison(test, prediction):
    observed = test["N_next"].to_numpy()
    model_errors = np.abs(prediction - observed)
    persistence_errors = np.abs(test["N_t"].to_numpy() - observed)
    matches = test.assign(model=model_errors, persistence=persistence_errors).groupby("match_id")
    sums = matches[["model", "persistence"]].sum()
    sizes = matches.size().to_numpy()
    delta = (sums["model"] - sums["persistence"]).to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    boot, null = [], []
    for _ in range(20):
        draw = rng.integers(0, len(delta), size=(500, len(delta)))
        boot.extend(delta[draw].sum(axis=1) / sizes[draw].sum(axis=1))
        signs = rng.choice([-1, 1], size=(500, len(delta)))
        null.extend(signs @ delta / len(test))
    difference = float(model_errors.mean() - persistence_errors.mean())
    return {
        "matches": len(sums), "games": len(test),
        "mae_model": float(model_errors.mean()), "mae_persistence": float(persistence_errors.mean()),
        "delta_mae": difference, "delta_ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
        "paired_sign_permutation_p": float((1 + np.count_nonzero(np.abs(null) >= abs(difference))) / (1 + len(null))),
        "skill_mae": float(1 - model_errors.mean() / persistence_errors.mean()),
        "skill_mse": float(1 - np.mean((prediction - observed) ** 2) / np.mean(persistence_errors ** 2)),
    }


def equal_budget(train, test):
    columns = regression.get_feature_cols(train)
    candidates = {**regression.MODELS, "SVR": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1))}
    return {
        name: regression.evaluate_model(
            model, train[columns].to_numpy(), train["N_next"].to_numpy(),
            test[columns].to_numpy(), test["N_next"].to_numpy(), train["date"], train_limit=5000,
        )
        for name, model in candidates.items()
    }


def calibration(bundle):
    train, test = regression.temporal_split(classification.build_dataset())
    columns = bundle["clf_features"]
    probabilities = bundle["clf_model"].predict_proba(test[columns].fillna(bundle["clf_medians"]))[:, 1]
    target = test["next_dir_up"].to_numpy()
    reference = np.full(len(test), train["next_dir_up"].mean())
    observed, predicted = calibration_curve(target, probabilities, n_bins=10, strategy="quantile")
    _, edges = np.histogram(probabilities, bins=np.quantile(probabilities, np.linspace(0, 1, 11)))
    bins = np.searchsorted(np.unique(edges)[1:-1], probabilities)
    counts = np.bincount(bins)
    if len(counts) != len(observed):
        raise ValueError("Los grupos de calibración y su recuento no coinciden.")
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Calibración ideal")
    axes[0].plot(predicted, observed, "o-", color="#246760", label="XGBoost")
    axes[0].set(ylabel="Frecuencia observada", title="Fiabilidad de las probabilidades de R5", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend()
    axes[1].hist(probabilities, bins=np.linspace(0, 1, 21), color="#246760")
    axes[1].set(xlabel="Probabilidad estimada de aumento del índice", ylabel="Juegos")
    fig.tight_layout()
    path = PROJECT_ROOT / "docs_tesis" / "figures" / "r5_reliability.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "games": len(test), "brier": brier_score_loss(target, probabilities),
        "log_loss": log_loss(target, probabilities), "reference_probability": float(reference[0]),
        "reference_brier": brier_score_loss(target, reference), "reference_log_loss": log_loss(target, reference),
        "bins": [{"mean_probability": p, "observed_frequency": o, "games": int(n)}
                 for p, o, n in zip(predicted, observed, counts)],
    }


def main():
    bundle = joblib.load(MODEL_BUNDLE)
    train, test = regression.temporal_split(regression.build_dataset()[0])
    prediction = bundle["reg_model"].predict(test[bundle["reg_features"]].fillna(bundle["reg_medians"]))
    outputs = {
        "r4_persistence_audit.json": persistence_comparison(test, prediction),
        "r5_calibration.json": calibration(bundle),
        "r4_equal_budget.json": equal_budget(train, test),
    }
    for name, values in outputs.items():
        (PROCESSED_DIR / name).write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
