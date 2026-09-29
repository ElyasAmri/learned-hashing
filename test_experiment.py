import unittest

import numpy as np
import torch

from experiment import (Autoencoder, code_statistics, encode, evaluate,
                        hamming_neighbors, quantize, train_autoencoder)


class HashingTests(unittest.TestCase):
    def test_training_medians_define_thresholds(self):
        train = np.array([[0, 8], [2, 6], [4, 4], [6, 2]])
        thresholds = np.median(train, axis=0)
        np.testing.assert_array_equal(quantize(train, thresholds).mean(axis=0), [0.5, 0.5])
        np.testing.assert_array_equal(quantize(np.array([[100, 0]]), thresholds), [[True, False]])

    def test_packed_hamming_matches_brute_force_with_ties(self):
        rng = np.random.default_rng(3)
        database = rng.integers(0, 2, (27, 13)).astype(bool)
        queries = rng.integers(0, 2, (7, 13)).astype(bool)
        distances = (queries[:, None] != database[None]).sum(axis=2)
        expected = np.argsort(distances, axis=1, kind="stable")[:, :10]
        np.testing.assert_array_equal(hamming_neighbors(database, queries, 10, batch_size=3), expected)

    def test_statistics_count_collisions_and_constant_bits(self):
        codes = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
        stats = code_statistics(codes)
        self.assertEqual(stats["unique_codes"], 2)
        self.assertEqual(stats["duplicate_item_fraction"], 0.5)
        self.assertEqual(stats["constant_bits"], 1)
        self.assertEqual(stats["mean_absolute_nonconstant_bit_correlation"], 1.0)
        self.assertEqual(stats["packed_database_bytes"], 4)
        self.assertEqual(stats["occupied_bucket_size_percentiles"]["max"], 2)

    def test_all_constant_codes(self):
        stats = code_statistics(np.zeros((5, 16), dtype=bool))
        self.assertEqual(stats["unique_codes"], 1)
        self.assertIsNone(stats["mean_absolute_nonconstant_bit_correlation"])

    def test_precision_uses_query_labels(self):
        database = np.array([[0, 0], [0, 1], [1, 1]], dtype=bool)
        result = evaluate(database, database[[0, 2]], np.array([0, 0, 1]), np.array([0, 1]), 1)
        self.assertEqual(result["same_class_precision_at_1"], 1.0)

    def test_autoencoder_training_and_encoding(self):
        rng = np.random.default_rng(42)
        train = rng.random((16, 784), dtype=np.float32)
        validation = rng.random((8, 784), dtype=np.float32)
        model, history = train_autoencoder(train, validation, 4, 2, 8, "cpu", 42)
        self.assertEqual(len(history), 2)
        self.assertTrue(np.isfinite(history).all())
        latent = encode(model, validation, "cpu", batch_size=3)
        self.assertEqual(latent.shape, (8, 4))
        self.assertTrue(np.isfinite(latent).all())
        self.assertEqual(model(torch.from_numpy(validation)).shape, (8, 784))
        self.assertIsInstance(model, Autoencoder)


if __name__ == "__main__":
    unittest.main()
