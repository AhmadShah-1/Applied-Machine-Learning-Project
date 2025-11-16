'''
This file tries Logitic Regressiob with scaling and random forest, and saves the best model
'''

from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import List, Tuple

import pandas as pd 


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Error: dataset not found at {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    # Use all feature columns except IDs and target
    excluded = {"Subject_ID", "Trial_ID", "Target"}
    features = [c for c in df.columns if c not in excluded]
    # Sanity check: expect 45 features
    if len(features) != 45:
        print(
            f"Warning: expected 45 features, found {len(features)}. Proceeding with detected columns.",
            file=sys.stderr,
        )
    return features


def evaluate_and_choose_model(X: pd.DataFrame, y: pd.Series):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import make_scorer, f1_score

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    macro_f1 = make_scorer(f1_score, average="macro")

    candidates: List[Tuple[str, object]] = [
        (
            "logreg_scaled",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            multi_class="multinomial",
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=400,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
    ]

    results = []
    for name, model in candidates:
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring=macro_f1)
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        results.append(
            {
                "name": name,
                "f1_macro_mean": float(f1_scores.mean()),
                "f1_macro_std": float(f1_scores.std()),
                "acc_mean": float(acc_scores.mean()),
                "acc_std": float(acc_scores.std()),
            }
        )

    # Choose by best macro-F1, tie-breaker accuracy
    results.sort(key=lambda r: (r["f1_macro_mean"], r["acc_mean"]), reverse=True)
    best_name = results[0]["name"]

    # Recreate best model instance to return unfitted
    if best_name == "logreg_scaled":
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression

        best_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        multi_class="multinomial",
                        class_weight="balanced",
                    ),
                ),
            ]
        )
    else:
        from sklearn.ensemble import RandomForestClassifier

        best_model = RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

    return best_model, results


def train_and_save(
    train_csv: Path,
    model_dir: Path,
) -> None:
    df = load_dataset(train_csv)
    feature_cols = select_feature_columns(df)

    X = df[feature_cols]
    y = df["Target"].astype(int)

    model, cv_results = evaluate_and_choose_model(X, y)

    # Fit on all training data
    model.fit(X, y)

    # Prepare output directory
    model_dir.mkdir(parents=True, exist_ok=True)

    # Persist model and metadata
    import joblib  # type: ignore

    model_path = model_dir / "eeg_intent_model.joblib"
    meta_path = model_dir / "eeg_intent_model.meta.json"
    joblib.dump(model, model_path)

    metadata = {
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "target_column": "Target",
        "cv_results": cv_results,
        "model_path": str(model_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    # Simple holdout evaluation on 20% of the same training file (reporting only)
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix

        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model_holdout = joblib.load(model_path)
        model_holdout.fit(X_tr, y_tr)
        y_pred = model_holdout.predict(X_va)
        report = classification_report(y_va, y_pred, digits=3)
        cm = confusion_matrix(y_va, y_pred).tolist()
        print("Model selection (5-fold CV) results:")
        for r in cv_results:
            print(
                f" - {r['name']}: F1_macro={r['f1_macro_mean']:.3f}±{r['f1_macro_std']:.3f}, "
                f"Acc={r['acc_mean']:.3f}±{r['acc_std']:.3f}"
            )
        print("\nHoldout validation on 20% of training file:")
        print(report)
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)
    except Exception as e:
        print(f"Warning: holdout evaluation skipped due to: {e}", file=sys.stderr)


if __name__ == "__main__":
    root = get_project_root()
    train_csv = root / "Dataset" / "Modified" / "eeg_neuroprosthetic_dataset_first_80.csv"
    model_out_dir = root / "Model_Training" / "Version1" / "models"

    train_and_save(train_csv, model_out_dir)
    print("\nSaved model and metadata to:")
    print(f" - {model_out_dir / 'eeg_intent_model.joblib'}")
    print(f" - {model_out_dir / 'eeg_intent_model.meta.json'}")
    