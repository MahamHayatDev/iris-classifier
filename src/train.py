"""Phase 4: Train a K-Nearest Neighbors model."""
from pathlib import Path

import joblib
from sklearn.neighbors import KNeighborsClassifier

from src.preprocess import prepare_data

OUTPUT_DIR = Path("outputs")
DEFAULT_K = 5


def train_model(k=DEFAULT_K):
    """Train KNN and predict on the test set."""
    X_train, X_test, y_train, y_test, scaler, features, classes = prepare_data()

    model = KNeighborsClassifier(n_neighbors=k)  # INSTANTIATE: build the frame
    model.fit(X_train, y_train)                  # FIT: memorize the training data
    predictions = model.predict(X_test)          # PREDICT: classify unseen flowers

    return model, predictions, X_test, y_test, scaler, classes


def main():
    model, predictions, X_test, y_test, scaler, classes = train_model()
    actual = y_test.to_numpy()

    print(f"=== KNN TRAINED (K = {DEFAULT_K}) ===\n")
    print(f"{'#':<4}{'Actual':<14}{'Predicted':<14}{'Result'}")
    for i in range(10):
        result = "correct" if actual[i] == predictions[i] else "WRONG"
        print(f"{i + 1:<4}{classes[actual[i]]:<14}{classes[predictions[i]]:<14}{result}")

    correct = int((actual == predictions).sum())
    print(f"\nCorrect predictions: {correct} out of {len(actual)}")
    print(f"Quick accuracy     : {model.score(X_test, y_test):.2%}")

    # Save the trained model and scaler for later phases
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "classes": classes},
                OUTPUT_DIR / "knn_model.joblib")
    print("\nSaved model: outputs/knn_model.joblib")


if __name__ == "__main__":
    main()