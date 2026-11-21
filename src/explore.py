"""Phase 2: Explore the Iris dataset."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files, no pop-up window
import seaborn as sns

from src.data_loader import load_iris_data

OUTPUT_DIR = Path("outputs")


def main():
    df, features, classes = load_iris_data()

    print("=== SHAPE (rows, columns) ===")
    print(df.shape)

    print("\n=== FEATURES ===")
    print(features)

    print("\n=== CLASSES ===")
    print(classes)

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== SAMPLES PER CLASS (is it balanced?) ===")
    print(df["species"].value_counts())

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    print("\n=== SUMMARY STATISTICS ===")
    print(df[features].describe().round(2))

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot = sns.pairplot(df, vars=features, hue="species", palette="Set2")
    plot.savefig(OUTPUT_DIR / "pairplot.png", dpi=120)
    print("\nSaved plot: outputs/pairplot.png")


if __name__ == "__main__":
    main()