"""Phase 6: Evaluate the final model (confusion matrix, precision, recall, F1)."""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.train import train_model
from src.tune import find_best_k

OUTPUT_DIR = Path("outputs")


def main():
    best_k, _, _ = find_best_k()
    model, predictions, X_test, y_test, scaler, classes = train_model(k=best_k)

    class_names = [str(c) for c in classes]
    y_true = y_test.to_numpy()

    accuracy = accuracy_score(y_true, predictions)
    macro_f1 = f1_score(y_true, predictions, average="macro")
    cm = confusion_matrix(y_true, predictions)
    report = classification_report(y_true, predictions, target_names=class_names, digits=2)

    print(f"=== FINAL MODEL: KNN with K = {best_k} ===\n")
    print(f"Accuracy : {accuracy:.2%}")
    print(f"Macro F1 : {macro_f1:.2f}")

    print("\n=== CONFUSION MATRIX (rows = actual, columns = predicted) ===")
    print(cm)

    print("\n=== PRECISION / RECALL / F1 PER CLASS ===")
    print(report)

    print("=== MISTAKES ===")
    wrong = [(class_names[a], class_names[p]) for a, p in zip(y_true, predictions) if a != p]
    if wrong:
        for actual, predicted in wrong:
            print(f"Actual {actual}  ->  predicted {predicted}")
    else:
        print("No mistakes on the test set.")

    # Save the confusion matrix picture
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix (K = {best_k})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=120)
    plt.close()

    # Save the text report
    with open(OUTPUT_DIR / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(f"KNN with K = {best_k}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Macro F1: {macro_f1:.2f}\n\n")
        f.write(f"Confusion matrix:\n{cm}\n\n")
        f.write(report)

    # Save the final model (replaces the earlier K=5 model)
    joblib.dump({"model": model, "scaler": scaler, "classes": classes},
                OUTPUT_DIR / "knn_model.joblib")

    print("\nSaved: outputs/confusion_matrix.png")
    print("Saved: outputs/evaluation_report.txt")
    print("Saved: outputs/knn_model.joblib (final model)")


if __name__ == "__main__":
    main()