"""Regenerate docs/confusion_matrix.png, the figure the README shows.

Run from the repo root:

    python examples/make_confusion_matrix.py

Predictions come from cross_val_predict on the same folds cross_val_score
reports, so the panel and the headline accuracy describe the same run.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write a file without needing a display

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict, cross_val_score

from spt_classify import build_pipeline, make_dataset
from spt_classify.features import extract_many

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "confusion_matrix.png"


def main(n_per_class: int = 250, n_steps: int = 100, cv: int = 5, seed: int = 0) -> None:
    tracks, labels = make_dataset(n_per_class=n_per_class, n_steps=n_steps, seed=seed)
    X = extract_many(tracks)
    y = np.asarray(labels)

    model = build_pipeline(seed)
    accuracy = cross_val_score(model, X, y, cv=cv).mean()
    predictions = cross_val_predict(model, X, y, cv=cv)

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    ConfusionMatrixDisplay.from_predictions(
        y,
        predictions,
        normalize="true",
        cmap="Blues",
        values_format=".2f",
        colorbar=False,
        xticks_rotation=45,
        ax=ax,
    )
    ax.set_title(f"{cv}-fold CV, {len(tracks)} simulated tracks ({accuracy:.0%} accuracy)")
    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    print(f"wrote {OUTPUT}  ({accuracy:.3f} accuracy)")


if __name__ == "__main__":
    main()
