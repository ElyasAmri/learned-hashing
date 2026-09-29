import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiment import (Autoencoder, code_statistics, encode, evaluate, load_data,
                        hamming_neighbors, pair_distances, quantize, separation_penalty,
                        train_autoencoder)
from penalty_experiment import PENALTIES, select_weights


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

    def test_distribution_distances_match_known_values(self):
        p = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
        q = torch.tensor([[0.2, 0.8]], dtype=torch.float64)
        kl = pair_distances(p.log(), q.log(), "inverse_kl")
        self.assertAlmostEqual(kl.item(), 0.6 * np.log(4))
        self.assertAlmostEqual(pair_distances(p.log(), q.log(), "inverse_wasserstein").item(), 0.6)
        for penalty in PENALTIES:
            torch.testing.assert_close(pair_distances(p.log(), q.log(), penalty),
                                       pair_distances(q.log(), p.log(), penalty))
            torch.testing.assert_close(pair_distances(p.log(), p.log(), penalty), torch.zeros(1, dtype=p.dtype))
        left = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float64)
        right = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float64)
        self.assertAlmostEqual(pair_distances(left.log(), right.log(), "inverse_wasserstein").item(), 0.7)

    def test_penalty_gradients_and_reciprocal_direction(self):
        for penalty in PENALTIES:
            with self.subTest(penalty=penalty):
                z = torch.tensor([[0.2, -0.2], [-0.2, 0.2]], requires_grad=True)
                loss = separation_penalty(z, penalty, torch.Generator().manual_seed(1))
                loss.backward()
                self.assertTrue(torch.isfinite(z.grad).all())
                self.assertGreater(z.grad.abs().sum().item(), 0)
                distant_loss = separation_penalty(z.detach() * 5, penalty, torch.Generator().manual_seed(1))
                self.assertLess(distant_loss.item(), loss.item())
                identical = torch.zeros(3, 4, requires_grad=True)
                identical_loss = separation_penalty(identical, penalty, torch.Generator().manual_seed(1))
                identical_loss.backward()
                self.assertAlmostEqual(identical_loss.item(), 1000, places=3)
                self.assertTrue(torch.isfinite(identical.grad).all())

    def test_pairing_excludes_self_pairs(self):
        latent = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        with patch("experiment.pair_distances", return_value=torch.ones(5)) as distance:
            separation_penalty(latent, "inverse_kl", torch.Generator().manual_seed(42))
            left, right, _ = distance.call_args.args
            self.assertTrue((left[:, 0] != right[:, 0]).all())
        self.assertEqual(separation_penalty(latent[:1], "inverse_kl", torch.Generator()).item(), 0)

    def test_validation_queries_do_not_load_test_data(self):
        class FakeDataset:
            data = torch.arange(60000).reshape(-1, 1).expand(-1, 784)
            targets = torch.arange(60000)

            def __len__(self):
                return 60000

        with patch("experiment.FashionMNIST", return_value=FakeDataset()) as dataset:
            train, validation, queries, labels, query_labels = load_data("unused", 42, 5, 3, 2, "validation")
            dataset.assert_called_once_with("unused", train=True, download=True)
        np.testing.assert_array_equal(queries, validation[:2])
        self.assertFalse(set(labels) & set(query_labels))
        self.assertEqual(train.shape, (5, 784))

    def test_selection_uses_validation_and_breaks_ties_by_weight(self):
        results = []
        for penalty in PENALTIES:
            for weight, score in [(1e-3, 0.6), (1e-4, 0.6), (1e-5, 0.5)]:
                results.append({"method": penalty, "bits": 16, "penalty_weight": weight,
                                "validation_precision_at_10": score})
        self.assertEqual(select_weights(results, [16]), {"16": {p: 1e-4 for p in PENALTIES}})

    def test_penalized_training_and_zero_weight_control(self):
        rng = np.random.default_rng(42)
        train = rng.random((16, 784), dtype=np.float32)
        validation = rng.random((8, 784), dtype=np.float32)
        baseline, history = train_autoencoder(train, validation, 4, 1, 8, "cpu", 42)
        zero, zero_history = train_autoencoder(train, validation, 4, 1, 8, "cpu", 42, "inverse_kl", 0)
        self.assertEqual(history, zero_history)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, zero.state_dict()[name], rtol=0, atol=0)
        for penalty in PENALTIES:
            model, history = train_autoencoder(train, validation, 4, 2, 8, "cpu", 42, penalty, 1e-4)
            self.assertTrue(np.isfinite(history).all())
            self.assertTrue(np.isfinite(encode(model, validation, "cpu")).all())


if __name__ == "__main__":
    unittest.main()
