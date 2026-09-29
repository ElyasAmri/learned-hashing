import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from sklearn.decomposition import PCA

from diagnose_quantization import continuous_neighbors, fit_itq, load_artifact, save_artifact
from experiment import (Autoencoder, code_statistics, encode, evaluate, load_data,
                        hamming_neighbors, pair_distances, quantize, quantization_penalty, separation_penalty,
                        train_autoencoder)
from penalty_experiment import PENALTIES, run_variant, select_weights


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
        for penalty in PENALTIES + ["quantization"]:
            model, history = train_autoencoder(train, validation, 4, 2, 8, "cpu", 42, penalty, 1e-4)
            self.assertTrue(np.isfinite(history).all())
            self.assertTrue(np.isfinite(encode(model, validation, "cpu")).all())


class QuantizationTests(unittest.TestCase):
    def test_quantization_penalty_targets_binary_values(self):
        self.assertEqual(quantization_penalty(torch.tensor([[-1., 1.]])).item(), 0)
        self.assertEqual(quantization_penalty(torch.zeros(2, 3)).item(), 1)
        z = torch.tensor([[0.25, 2., -0.25, -2.]], requires_grad=True)
        loss = quantization_penalty(z)
        loss.backward()
        np.testing.assert_array_equal(torch.sign(z.grad).numpy(), [[-1, 1, 1, -1]])
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_quantization_weight_selection(self):
        candidates = [{"bits": 16, "method": "quantization", "penalty_weight": w,
                       "validation_precision_at_10": score,
                       "median_binary": {"validation_precision_at_10": 1 - score}}
                      for w, score in [(0.001, 0.2), (0.01, 0.6), (0.1, 0.4)]]
        self.assertEqual(select_weights(candidates, [16], ["quantization"]), {"16": {"quantization": 0.01}})

    def test_quantization_variant_saves_zero_threshold_model(self):
        rng = np.random.default_rng(9)
        train = rng.random((16, 784), dtype=np.float32)
        validation = rng.random((8, 784), dtype=np.float32)
        data = (train, validation, validation, np.arange(16) % 2, np.arange(8) % 2)
        args = SimpleNamespace(epochs=1, batch_size=8, device="cpu")
        with tempfile.TemporaryDirectory() as root:
            result = run_variant(data, 4, "quantization", 0.01, args, 42, "validation", Path(root) / "model")
            self.assertEqual(result["artifact_roundtrip"], "passed")
            self.assertEqual(result["threshold_rule"], "zero")
            self.assertIn("validation_precision_at_10", result)
            self.assertNotIn("test_precision_at_10", result)
            self.assertIn("validation_precision_at_10", result["median_binary"])
            _, arrays = load_artifact(Path(root) / "model")
            np.testing.assert_array_equal(arrays["binary_thresholds"], np.zeros(4))

    def test_continuous_search_matches_direct_distances(self):
        rng = np.random.default_rng(12)
        database = rng.normal(size=(31, 5))
        database[1] = database[0]
        queries = np.concatenate([database[:2], rng.normal(size=(5, 5))])
        for metric in ["euclidean", "cosine"]:
            if metric == "euclidean":
                distances = np.square(queries[:, None] - database[None]).sum(axis=2)
            else:
                d = database / np.linalg.norm(database, axis=1, keepdims=True)
                q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
                distances = 1 - q @ d.T
            expected = np.argsort(distances, axis=1, kind="stable")[:, :10]
            np.testing.assert_array_equal(continuous_neighbors(database, queries, metric, 10, 3), expected)

    def test_continuous_ties_and_zero_vectors(self):
        database = np.zeros((20, 4))
        queries = np.zeros((3, 4))
        for metric in ["euclidean", "cosine"]:
            np.testing.assert_array_equal(continuous_neighbors(database, queries, metric),
                                          np.tile(np.arange(10), (3, 1)))

    def test_itq_is_orthogonal_and_reduces_objective(self):
        rng = np.random.default_rng(7)
        latent = rng.normal(size=(200, 8)) * np.arange(1, 9) + 5
        mean, rotation, history = fit_itq(latent, 15, 42)
        np.testing.assert_allclose(mean, latent.mean(axis=0))
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(8), atol=1e-12)
        self.assertTrue((np.diff(history) <= 1e-10).all())
        self.assertLess(history[-1], history[0])
        centered = latent - mean
        np.testing.assert_allclose(np.linalg.norm(centered[:5] - centered[5:10], axis=1),
                                   np.linalg.norm((centered[:5] - centered[5:10]) @ rotation, axis=1))
        repeated = fit_itq(latent, 15, 42)
        np.testing.assert_array_equal(rotation, repeated[1])

    def test_pca_and_autoencoder_artifacts_reproduce_codes(self):
        rng = np.random.default_rng(42)
        train = rng.random((24, 784), dtype=np.float32)
        queries = rng.random((9, 784), dtype=np.float32)
        with tempfile.TemporaryDirectory() as root:
            for method in ["pca", "autoencoder"]:
                if method == "pca":
                    model = PCA(n_components=4, svd_solver="randomized", random_state=42).fit(train)
                    model.components_ = np.ascontiguousarray(model.components_)
                    transform = model.transform
                else:
                    model = Autoencoder(4)
                    transform = lambda x: encode(model, x, "cpu")
                latent = transform(train)
                original = transform(queries)
                thresholds = np.median(latent, axis=0)
                mean, rotation, _ = fit_itq(latent, 3, 42)
                path = Path(root) / method
                save_artifact(path, method, model, mean, thresholds, rotation)
                restored, saved = load_artifact(path)
                actual = restored(queries)
                np.testing.assert_allclose(actual, original, rtol=1e-5, atol=1e-5)
                if method == "pca":
                    np.testing.assert_array_equal(actual, original)
                np.testing.assert_array_equal(actual > saved["median_thresholds"], original > thresholds)
                np.testing.assert_array_equal((actual - saved["latent_mean"]) @ saved["rotation"] > 0,
                                              (original - mean) @ rotation > 0)
                with self.assertRaises(FileExistsError):
                    save_artifact(path, method, model, mean, thresholds, rotation)


if __name__ == "__main__":
    unittest.main()
