"""Phase 7A: Compare different algorithms on the same data."""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import load_iris_data
from src.preprocess import RANDOM_STATE, TEST_SIZE
from src.tune import find_best_k

OUTPUT_DIR = Path("outputs")


def main():
    best_k, _, _ = find_best_k()
    df, features, classes = load_iris_data()

    # Same split as every other phase
    X_train, X_test, y_train, y_test = train_test_split(
        df[features],
        df["target"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df["target"],
    )

    models = {
        f"KNN (K={best_k})": KNeighborsClassifier(n_neighbors=best_k),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(),
    }

    rows = []
    for name, classifier in models.items():
        pipe = make_pipeline(StandardScaler(), classifier)
        cv_accuracy = cross_val_score(pipe, X_train, y_train, cv=5).mean()
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
        rows.append({
            "Algorithm": name,
            "CV accuracy (train)": round(cv_accuracy, 4),
            "Test accuracy": round(accuracy_score(y_test, predictions), 4),
            "Test macro F1": round(f1_score(y_test, predictions, average="macro"), 4),
        })

    results = pd.DataFrame(rows)
    print("=== ALGORITHM COMPARISON ===\n")
    print(results.to_string(index=False))

    OUTPUT_DIR.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_DIR / "algorithm_comparison.csv", index=False)
    print("\nSaved: outputs/algorithm_comparison.csv")


if __name__ == "__main__":
    main()