"""Phase 2: Load the Iris dataset."""
from sklearn.datasets import load_iris


def load_iris_data():
    """Return the Iris data as a DataFrame, plus feature and class names."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    return df, list(iris.feature_names), list(iris.target_names)