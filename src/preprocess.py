"""Phase 3: Shuffle, split and scale the data."""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_iris_data

TEST_SIZE = 0.2      # 80% train / 20% test
RANDOM_STATE = 42    # makes the shuffle repeatable


def prepare_data(test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Shuffle, split (80/20) and scale the Iris data."""
    df, features, classes = load_iris_data()
    X = df[features]
    y = df["target"]

    # 1. Shuffle and split. stratify=y keeps the 3 classes balanced in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y,
    )

    # 2. Scale: learn the mean/std from TRAINING data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # test data only transformed

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features, classes


def main():
    X_train, X_test, y_train, y_test, scaler, features, classes = prepare_data()

    print("=== SPLIT SIZES ===")
    print("Training set:", X_train.shape)
    print("Test set    :", X_test.shape)

    print("\n=== CLASSES IN TRAINING SET ===")
    print(y_train.value_counts().sort_index())

    print("\n=== CLASSES IN TEST SET ===")
    print(y_test.value_counts().sort_index())

    print("\n=== SCALED TRAINING DATA (should be mean ~0, std ~1) ===")
    print("Mean:", X_train.mean(axis=0).round(2))
    print("Std :", X_train.std(axis=0).round(2))

    print("\n=== SCALED TEST DATA (close to 0 and 1, not exact) ===")
    print("Mean:", X_test.mean(axis=0).round(2))
    print("Std :", X_test.std(axis=0).round(2))


if __name__ == "__main__":
    main()