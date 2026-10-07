import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODEL_VERSION = "v1"
MODEL_DIR = Path("models") / MODEL_VERSION


def main():
    # ---------------------------------------------------------
    # 1. Generate synthetic loan data
    # ---------------------------------------------------------

    rng = np.random.default_rng(42)
    n = 1000

    income = rng.integers(20000, 150000, n)
    debt = rng.integers(1000, 80000, n)
    credit_score = rng.integers(300, 900, n)
    missed_payments = rng.integers(0, 6, n)

    debt_ratio = debt / income

    risk_score = (
        2.5 * debt_ratio
        + 0.7 * missed_payments
        - 0.006 * credit_score
    )

    default_probability = 1 / (1 + np.exp(-risk_score))

    defaulted = rng.binomial(1, default_probability)

    df = pd.DataFrame(
        {
            "income": income,
            "debt": debt,
            "credit_score": credit_score,
            "missed_payments": missed_payments,
            "defaulted": defaulted,
        }
    )

    # ---------------------------------------------------------
    # 2. Save complete synthetic dataset
    # ---------------------------------------------------------

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    dataset_path = data_dir / "loans.csv"
    df.to_csv(dataset_path, index=False)

    print(f"Dataset saved to {dataset_path}")

    # ---------------------------------------------------------
    # 3. Separate features and target
    # ---------------------------------------------------------

    X = df.drop(columns=["defaulted"])
    y = df["defaulted"]

    # ---------------------------------------------------------
    # 4. Split training and testing datasets
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ---------------------------------------------------------
    # 5. Save held-out test dataset
    # ---------------------------------------------------------

    test_df = X_test.copy()
    test_df["defaulted"] = y_test

    test_data_path = data_dir / "test.csv"
    test_df.to_csv(test_data_path, index=False)

    print(f"Test dataset saved to {test_data_path}")

    # ---------------------------------------------------------
    # 6. Build ML pipeline
    # ---------------------------------------------------------

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )

    # ---------------------------------------------------------
    # 7. Train model
    # ---------------------------------------------------------

    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 8. Evaluate during training
    # ---------------------------------------------------------

    predictions = model.predict(X_test)

    print("\nTraining Evaluation")
    print("-------------------")
    print("Accuracy:", accuracy_score(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # ---------------------------------------------------------
    # 9. Create version-specific model directory
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 10. Save trained model artifact
    # ---------------------------------------------------------

    model_path = MODEL_DIR / "loan_model.joblib"

    joblib.dump(model, model_path)

    print(f"\nModel saved to {model_path}")

    # ---------------------------------------------------------
    # 11. Create model metadata / lineage
    # ---------------------------------------------------------

    metadata = {
        "model_name": "loan-default-model",
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "LogisticRegression",
        "training": {
            "random_state": 42,
            "test_size": 0.2,
            "training_records": len(X_train),
            "test_records": len(X_test),
        },
        "features": list(X.columns),
        "target": "defaulted",
    }

    # ---------------------------------------------------------
    # 12. Save model metadata
    # ---------------------------------------------------------

    metadata_path = MODEL_DIR / "metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
