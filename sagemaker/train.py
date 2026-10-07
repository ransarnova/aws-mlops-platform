import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # SageMaker injects these paths into the training container.
    training_dir = os.environ["SM_CHANNEL_TRAINING"]
    model_dir = os.environ["SM_MODEL_DIR"]

    print(f"Training data directory: {training_dir}")
    print(f"Model output directory: {model_dir}")

    # ---------------------------------------------------------
    # Load training dataset
    # ---------------------------------------------------------

    training_path = Path(training_dir) / "loans.csv"

    df = pd.read_csv(training_path)

    print(f"Loaded {len(df)} training records")

    # ---------------------------------------------------------
    # Separate features and target
    # ---------------------------------------------------------

    X = df.drop(columns=["defaulted"])
    y = df["defaulted"]

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # ---------------------------------------------------------
    # Build and train model
    # ---------------------------------------------------------

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print(f"validation_accuracy={accuracy}")

    # ---------------------------------------------------------
    # Save model into SageMaker model directory
    # ---------------------------------------------------------

    model_path = Path(model_dir) / "loan_model.joblib"

    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

    # Save basic training metadata with the model artifact.
    metadata = {
        "algorithm": "LogisticRegression",
        "validation_accuracy": accuracy,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "training_records": len(X_train),
        "test_records": len(X_test),
        "features": list(X.columns),
        "target": "defaulted",
    }

    metadata_path = Path(model_dir) / "metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main()
