"""Phase 7C: Automated tests for the classification pipeline."""
import unittest

import numpy as np

from src.data_loader import load_iris_data
from src.predict import predict_flower
from src.preprocess import prepare_data
from src.train import train_model


class TestData(unittest.TestCase):
    def test_dataset_shape_and_balance(self):
        df, features, classes = load_iris_data()
        self.assertEqual(df.shape[0], 150)
        self.assertEqual(len(features), 4)
        self.assertEqual(len(classes), 3)
        self.assertTrue((df["species"].value_counts() == 50).all())

    def test_no_missing_values(self):
        df, _, _ = load_iris_data()
        self.assertEqual(int(df.isnull().sum().sum()), 0)


class TestPreprocessing(unittest.TestCase):
    def test_split_sizes(self):
        X_train, X_test, y_train, y_test, *_ = prepare_data()
        self.assertEqual(X_train.shape, (120, 4))
        self.assertEqual(X_test.shape, (30, 4))

    def test_split_is_stratified(self):
        _, _, y_train, y_test, *_ = prepare_data()
        self.assertTrue((y_train.value_counts() == 40).all())
        self.assertTrue((y_test.value_counts() == 10).all())

    def test_training_data_is_scaled(self):
        X_train, *_ = prepare_data()
        self.assertTrue(np.allclose(X_train.mean(axis=0), 0, atol=1e-6))
        self.assertTrue(np.allclose(X_train.std(axis=0), 1, atol=1e-6))


class TestModel(unittest.TestCase):
    def test_accuracy_is_reasonable(self):
        model, predictions, X_test, y_test, *_ = train_model()
        self.assertGreaterEqual(model.score(X_test, y_test), 0.90)


class TestPrediction(unittest.TestCase):
    def test_typical_setosa(self):
        species, _ = predict_flower(5.0, 3.4, 1.5, 0.2)
        self.assertEqual(species, "setosa")

    def test_typical_versicolor(self):
        species, _ = predict_flower(5.9, 2.8, 4.3, 1.3)
        self.assertEqual(species, "versicolor")

    def test_typical_virginica(self):
        species, _ = predict_flower(6.7, 3.0, 5.2, 2.3)
        self.assertEqual(species, "virginica")

    def test_probabilities_sum_to_one(self):
        _, probabilities = predict_flower(5.9, 3.0, 5.1, 1.8)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)


if __name__ == "__main__":
    unittest.main()