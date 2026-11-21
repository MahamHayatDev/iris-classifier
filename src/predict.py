"""Phase 7B: Predict the species of a new flower."""
import sys
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("outputs") / "knn_model.joblib"


def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    """Return (species, probabilities) for one flower. Measurements are in cm."""
    saved = joblib.load(MODEL_PATH)
    model, scaler, classes = saved["model"], saved["scaler"], saved["classes"]

    flower = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=scaler.feature_names_in_,
    )
    scaled = scaler.transform(flower)

    index = int(model.predict(scaled)[0])
    probabilities = model.predict_proba(scaled)[0]
    return str(classes[index]), {str(c): float(p) for c, p in zip(classes, probabilities)}


def show(values):
    species, probabilities = predict_flower(*values)
    print(f"Flower (cm): {values}")
    print(f"Prediction : {species}")
    for name, p in probabilities.items():
        print(f"   {name:<12}{p:.0%}")
    print()


def main():
    # Usage: python -m src.predict 5.9 3.0 5.1 1.8
    if len(sys.argv) == 5:
        show([float(v) for v in sys.argv[1:]])
    else:
        print("=== NEW FLOWERS (the model has never seen these) ===\n")
        show([5.0, 3.4, 1.5, 0.2])
        show([5.9, 2.8, 4.3, 1.3])
        show([6.7, 3.0, 5.2, 2.3])


if __name__ == "__main__":
    main()