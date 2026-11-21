"""Phase 5: Find the best K using cross-validation on the training set."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_iris_data
from src.preprocess import RANDOM_STATE, TEST_SIZE
from src.train import train_model

OUTPUT_DIR = Path("outputs")
K_VALUES = list(range(1, 31))


def find_best_k():
    """Try each K with 5-fold cross-validation. Returns best K, all K values, errors."""
    df, features, classes = load_iris_data()

    # Same split as Phase 3 (same test_size and random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        df[features],
        df["target"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df["target"],
    )

    errors = []
    for k in K_VALUES:
        # The pipeline re-fits the scaler inside each fold (no leakage)
        pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        scores = cross_val_score(pipe, X_train, y_train, cv=5)
        errors.append(round(1 - scores.mean(), 6))

    # If several K tie for the lowest error, pick the largest (smoother, less noisy)
    lowest = min(errors)
    best_k = max(k for k, e in zip(K_VALUES, errors) if e == lowest)
    return best_k, K_VALUES, errors


def plot_errors(best_k, k_values, errors):
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(k_values, errors, marker="o", color="#1f4e8c")
    plt.axvline(best_k, color="#e8590c", linestyle="--", label=f"Best K = {best_k}")
    plt.title("Choosing K: Cross-Validated Error Rate")
    plt.xlabel("K (number of neighbors)")
    plt.ylabel("Error rate")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "k_tuning.png", dpi=120)
    plt.close()


def main():
    best_k, k_values, errors = find_best_k()

    print("=== CROSS-VALIDATED ERROR FOR EACH K ===")
    for k, e in zip(k_values, errors):
        marker = "  <-- BEST" if k == best_k else ""
        print(f"K = {k:<3} error = {e:.4f}{marker}")

    plot_errors(best_k, k_values, errors)
    print(f"\nBest K: {best_k}")
    print("Saved plot: outputs/k_tuning.png")

    # One final check on the locked test set
    model, predictions, X_test, y_test, scaler, classes = train_model(k=best_k)
    print(f"\nTest accuracy with K = {best_k}: {model.score(X_test, y_test):.2%}")


if __name__ == "__main__":
    main()