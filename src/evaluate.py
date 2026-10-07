import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


MODEL_VERSION = "v1"

MODEL_DIR = Path("models") / MODEL_VERSION

MODEL_PATH = MODEL_DIR / "loan_model.joblib"
TEST_DATA_PATH = Path("data/test.csv")
EVALUATION_PATH = MODEL_DIR / "evaluation.json"


# These are learning/demo thresholds only.
# They are NOT production lending-risk thresholds.
MIN_ACCURACY = 0.70
MIN_DEFAULT_RECALL = 0.65


def main():
    # ---------------------------------------------------------
    # 1. Verify required artifacts exist
    # ---------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run training first."
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_PATH}. "
            "Run training first."
        )

    # ---------------------------------------------------------
    # 2. Load trained model
    # ---------------------------------------------------------

    model = joblib.load(MODEL_PATH)

    # ---------------------------------------------------------
    # 3. Load held-out test dataset
    # ---------------------------------------------------------

    df = pd.read_csv(TEST_DATA_PATH)

    X_test = df.drop(columns=["defaulted"])
    y_test = df["defaulted"]

    # ---------------------------------------------------------
    # 4. Generate predictions
    # ---------------------------------------------------------

    predictions = model.predict(X_test)

    # ---------------------------------------------------------
    # 5. Calculate model metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    # ---------------------------------------------------------
    # 6. Calculate confusion matrix
    # ---------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    # ---------------------------------------------------------
    # 7. Apply automated quality gate
    # ---------------------------------------------------------

    passed = (
        accuracy >= MIN_ACCURACY
        and recall >= MIN_DEFAULT_RECALL
    )

    # ---------------------------------------------------------
    # 8. Create evaluation report
    # ---------------------------------------------------------

    evaluation = {
        "model_version": MODEL_VERSION,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "quality_gate": {
            "min_accuracy": MIN_ACCURACY,
            "min_default_recall": MIN_DEFAULT_RECALL,
            "passed": bool(passed),
        },
    }

    # ---------------------------------------------------------
    # 9. Save machine-readable evaluation artifact
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVALUATION_PATH, "w") as f:
        json.dump(evaluation, f, indent=4)

    print(json.dumps(evaluation, indent=4))

    # ---------------------------------------------------------
    # 10. Return pipeline-compatible result
    # ---------------------------------------------------------

    if not passed:
        print("\nQUALITY GATE: FAILED")
        raise SystemExit(1)

    print("\nQUALITY GATE: PASSED")


if __name__ == "__main__":
    main()
