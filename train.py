"""Train the xG model and save it to models.

Run once from the project roo    python train.py

Downloading the six competitions takes a few minutes. The application never
runs this file, it only loads the artefact it produces.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

import pipeline

MODELS_DIR = Path("models")
ARTEFACT = MODELS_DIR / "xg_model.joblib"

TEST_SIZE = 0.25
RANDOM_STATE = 42


def main():
    ids = pipeline.all_match_ids()
    print(f"{len(ids)} matches to download")

    raw = pipeline.get_shots_many(ids)
    print(f"{len(raw)} raw shots")

    data = pipeline.prepare(raw, drop_penalties=True)
    keeper_median = float(data["gardien_dist_but"].median())
    print(f"{len(data)} shots kept, goal rate {data['is_goal'].mean():.4f}")

    X = pipeline.build_features(data, keeper_median=keeper_median)
    y = data["is_goal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    predicted = model.predict_proba(X_test)[:, 1]
    baseline = pd.Series(y_test.mean(), index=y_test.index)
    statsbomb = data.loc[X_test.index, "xg_statsbomb"]

    ll_baseline = log_loss(y_test, baseline)
    ll_model = log_loss(y_test, predicted)
    ll_statsbomb = log_loss(y_test, statsbomb)
    share = (ll_baseline - ll_model) / (ll_baseline - ll_statsbomb)

    print()
    print(f"log loss, baseline   {ll_baseline:.5f}")
    print(f"log loss, this model {ll_model:.5f}")
    print(f"log loss, StatsBomb  {ll_statsbomb:.5f}")
    print(f"brier,    this model {brier_score_loss(y_test, predicted):.5f}")
    print(f"share of StatsBomb's gain: {share:.1%}")
    print()
    print(f"train log loss {log_loss(y_train, model.predict_proba(X_train)[:, 1]):.5f}")
    print(f"test  log loss {ll_model:.5f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "columns": list(X.columns),
            "keeper_median": keeper_median,
        },
        ARTEFACT,
    )
    print()
    print(f"saved to {ARTEFACT}")


if __name__ == "__main__":
    main()
