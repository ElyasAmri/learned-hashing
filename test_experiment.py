import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import sklearn
import torch
import torchvision
from sklearn.decomposition import PCA

from benchmark_transfer import (evaluate_saved_databases, fit_itq_artifact, frozen_settings,
                                measure_cpu_encoding, run_pca, saved_codes, unseen_partitions)
from contrastive_experiment import (augment_views, contrastive_loss, diagnose_selected, latent_diagnostics,
                                    select_temperatures, train_paired)
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
        args = SimpleNamespace(epochs=1, batch_size=8, device="cpu", quantization_threshold="zero")
        with tempfile.TemporaryDirectory() as root:
            result = run_variant(data, 4, "quantization", 0.01, args, 42, "validation", Path(root) / "model")
            self.assertEqual(result["artifact_roundtrip"], "passed")
            self.assertEqual(result["threshold_rule"], "zero")
            self.assertIn("validation_precision_at_10", result)
            self.assertNotIn("test_precision_at_10", result)
            self.assertIn("validation_precision_at_10", result["median_binary"])
            _, arrays = load_artifact(Path(root) / "model")
            np.testing.assert_array_equal(arrays["binary_thresholds"], np.zeros(4))

    def test_quantization_median_rule_uses_training_thresholds(self):
        rng = np.random.default_rng(9)
        train = rng.random((16, 784), dtype=np.float32)
        validation = rng.random((8, 784), dtype=np.float32)
        data = (train, validation, validation, np.arange(16) % 2, np.arange(8) % 2)
        args = SimpleNamespace(epochs=1, batch_size=8, device="cpu", quantization_threshold="median")
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model"
            with patch("penalty_experiment.evaluate", wraps=evaluate) as evaluator:
                result = run_variant(data, 4, "quantization", 0.0001, args, 42, "validation", path)
                self.assertEqual(evaluator.call_count, 1)
            self.assertEqual(result["threshold_rule"], "median")
            self.assertEqual(result["validation_precision_at_10"], result["median_binary"]["validation_precision_at_10"])
            self.assertEqual(result["artifact_roundtrip"], "passed")
            transform, saved = load_artifact(path)
            np.testing.assert_array_equal(saved["binary_thresholds"], np.median(transform(train), axis=0))
            np.testing.assert_array_equal(saved["binary_thresholds"], saved["median_thresholds"])

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


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.source = {
            "config": {"penalties": ["quantization"], "quantization_threshold": "median",
                       "query_split": "validation", "epochs": 20, "batch_size": 256,
                       "train_size": 50000, "validation_size": 10000, "query_size": 10000,
                       "bits": [16, 32, 64], "seeds": [42, 7, 21, 84, 123],
                       "checkpoint_selection": "minimum_validation_reconstruction_mse"},
            "selected_weights": {"16": {"quantization": 0.001}, "32": {"quantization": 0.0003},
                                 "64": {"quantization": 0.0003}},
        }

    def test_frozen_settings_preserve_training_and_weights(self):
        config, weights = frozen_settings(self.source)
        self.assertEqual(config["dataset"], "kmnist")
        self.assertEqual(config["query_split"], "test")
        for key in ["epochs", "batch_size", "train_size", "validation_size", "query_size", "bits", "seeds"]:
            self.assertEqual(config[key], self.source["config"][key])
        self.assertEqual(weights, {"16": 0.001, "32": 0.0003, "64": 0.0003})
        self.assertEqual(self.source["config"]["query_split"], "validation")

    def test_smoke_settings_never_use_test_queries(self):
        config, weights = frozen_settings(self.source, smoke=True)
        self.assertEqual(config["query_split"], "validation")
        self.assertEqual(config["epochs"], 1)
        self.assertEqual(config["bits"], [16])
        self.assertEqual(weights, {"16": 0.001})
        self.assertEqual(self.source["config"]["epochs"], 20)

    def test_reject_nonmedian_source(self):
        self.source["config"]["quantization_threshold"] = "zero"
        with self.assertRaisesRegex(ValueError, "median-threshold"):
            frozen_settings(self.source)

    def test_kmnist_loading_uses_separate_official_splits(self):
        class FakeDataset:
            def __init__(self, size, offset):
                self.targets = torch.arange(size) + offset
                self.data = self.targets.reshape(-1, 1).expand(-1, 784)

            def __len__(self):
                return len(self.targets)

        def factory(root, train, download):
            return FakeDataset(60000, 0) if train else FakeDataset(10000, 60000)

        with patch("experiment.FashionMNIST") as fashion, patch("experiment.KMNIST", side_effect=factory) as kmnist:
            train, validation, queries, labels, query_labels, ids = load_data(
                "unused", 42, 5, 3, 2, dataset="kmnist", return_ids=True)
            self.assertEqual(kmnist.call_args_list, [call("unused", train=True, download=True),
                                                   call("unused", train=False, download=True)])
            fashion.assert_not_called()
        self.assertTrue((query_labels >= 60000).all())
        np.testing.assert_array_equal(ids["train"], labels)
        np.testing.assert_array_equal(ids["queries"] + 60000, query_labels)
        self.assertTrue((labels < 60000).all())
        self.assertEqual(queries.shape, (2, 784))
        with patch("experiment.KMNIST", side_effect=factory) as kmnist:
            data = load_data("unused", 42, 5, 3, 2, query_split="validation", dataset="kmnist")
            kmnist.assert_called_once_with("unused", train=True, download=True)
            np.testing.assert_array_equal(data[2], validation[:2])

    def test_transfer_pca_saves_and_reloads_codes(self):
        rng = np.random.default_rng(9)
        train = rng.random((24, 784), dtype=np.float32)
        queries = rng.random((8, 784), dtype=np.float32)
        data = (train, queries, queries, np.arange(24) % 2, np.arange(8) % 2)
        with tempfile.TemporaryDirectory() as root:
            result = run_pca(data, 4, 42, "test", Path(root) / "pca")
            self.assertEqual(result["artifact_roundtrip"], "passed")
            self.assertIn("test_precision_at_10", result)
            self.assertNotIn("validation_precision_at_10", result)
            transform, saved = load_artifact(Path(root) / "pca")
            np.testing.assert_array_equal(saved["binary_thresholds"], np.median(transform(train), axis=0))


class UnseenDatabaseTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(24, 4)).astype(np.float32)
        test = rng.normal(loc=10, size=(20, 4)).astype(np.float32)
        self.data = (train, train[:4], test, np.arange(24) % 2, np.arange(20) % 2,
                     {"train": np.arange(24), "validation": np.arange(50, 54), "queries": np.arange(100, 120)})

    def test_unseen_database_and_queries_are_disjoint_and_matched(self):
        partitions = unseen_partitions(self.data)
        seen, unseen, queries = [partitions[name] for name in ["training_database", "unseen_database", "queries"]]
        self.assertEqual(len(seen["images"]), 10)
        self.assertEqual(len(unseen["images"]), 10)
        self.assertEqual(len(queries["images"]), 10)
        self.assertEqual(seen["official_split"], "train")
        self.assertEqual(unseen["official_split"], "test")
        self.assertEqual(queries["official_split"], "test")
        self.assertFalse(set(unseen["ids"]) & set(queries["ids"]))
        np.testing.assert_array_equal(unseen["images"], self.data[2][:10])
        np.testing.assert_array_equal(queries["images"], self.data[2][10:])
        np.testing.assert_array_equal(seen["images"], self.data[0][:10])
        np.testing.assert_array_equal(queries["labels"], self.data[4][10:])

    def test_reject_overlapping_query_ids(self):
        self.data[5]["queries"][-1] = self.data[5]["queries"][0]
        with self.assertRaisesRegex(ValueError, "IDs overlap"):
            unseen_partitions(self.data)

    def test_saved_itq_codes_apply_centering_and_rotation(self):
        images = np.array([[1., 2.], [3., 4.]])
        arrays = {"binary_thresholds": np.zeros(2), "latent_mean": np.ones(2),
                  "rotation": np.array([[0., -1.], [1., 0.]])}
        np.testing.assert_array_equal(saved_codes(lambda x: x, arrays, images, itq=True),
                                      [[True, False], [True, False]])
        np.testing.assert_array_equal(saved_codes(lambda x: x, arrays, images), images > 0)

    def test_itq_artifact_fits_training_latents_and_keeps_source(self):
        train, validation, queries, labels, query_labels, _ = self.data
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "pca"
            destination = Path(root) / "itq"
            run_pca((train, validation, queries, labels, query_labels), 2, 42, "test", source)
            original_bytes = (source / "transform.npz").read_bytes()
            transform, _ = load_artifact(source)
            with patch("benchmark_transfer.fit_itq", wraps=fit_itq) as fitter:
                result = fit_itq_artifact(train, source, destination, 42)
                fitter.assert_called_once()
                np.testing.assert_array_equal(fitter.call_args.args[0], transform(train))
                self.assertEqual(fitter.call_args.kwargs, {"iterations": 50, "seed": 42})
            restored, arrays = load_artifact(destination)
            np.testing.assert_allclose(arrays["latent_mean"], transform(train).mean(axis=0, dtype=np.float64))
            np.testing.assert_array_equal(saved_codes(restored, arrays, queries, itq=True),
                                          (transform(queries) - arrays["latent_mean"]) @ arrays["rotation"] > 0)
            self.assertEqual((source / "transform.npz").read_bytes(), original_bytes)
            self.assertEqual(result["artifact_roundtrip"], "passed")
            self.assertEqual(len(result["itq_quantization_mse_by_iteration"]), 51)
            self.assertEqual(result["inference_parameter_bytes"], sum(arrays[key].nbytes for key in
                ["components", "input_mean", "latent_mean", "rotation", "binary_thresholds"]))

    def test_cpu_encoding_costs_include_packing_and_restore_threads(self):
        images = self.data[0]
        transform = Mock(side_effect=lambda x: x)
        arrays = {"binary_thresholds": np.zeros(4)}
        threads = torch.get_num_threads()
        with patch("benchmark_transfer.load_artifact", return_value=(transform, arrays)) as loader:
            result = measure_cpu_encoding(Path("unused"), images)
            loader.assert_called_once_with(Path("unused"), "cpu")
        self.assertEqual(transform.call_count, 9)
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertEqual(len(result["cpu_encode_seconds_by_repeat"]), 7)
        self.assertGreater(result["cpu_encode_median_seconds"], 0)
        self.assertEqual(result["packed_query_bytes"], len(images))

    def test_saved_thresholds_and_query_codes_are_reused(self):
        partitions = unseen_partitions(self.data)
        thresholds = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        arrays = {"binary_thresholds": thresholds.copy(), "median_thresholds": thresholds.copy()}
        with patch("benchmark_transfer.load_artifact", return_value=(lambda x: x, arrays)) as loader:
            with patch("benchmark_transfer.evaluate", wraps=evaluate) as evaluator:
                results = evaluate_saved_databases(partitions, Path("unused"), "cpu")
                self.assertEqual(evaluator.call_count, 2)
                first, second = evaluator.call_args_list
                self.assertIs(first.args[1], second.args[1])
                np.testing.assert_array_equal(first.args[1], self.data[2][10:] > thresholds)
                np.testing.assert_array_equal(first.args[0], self.data[0][:10] > thresholds)
                np.testing.assert_array_equal(second.args[0], self.data[2][:10] > thresholds)
            loader.assert_called_once_with(Path("unused"), "cpu")
        np.testing.assert_array_equal(arrays["binary_thresholds"], thresholds)
        self.assertEqual(results["unseen_database"]["unique_codes"], 1)
        self.assertEqual(results["unseen_database"]["bit_one_fractions"], [1.0] * 4)
        self.assertIn("test_precision_at_10", results["training_database"])


class ContrastiveTests(unittest.TestCase):
    def test_augmentations_are_seeded_bounded_and_nonmutating(self):
        images = torch.linspace(0, 1, 4 * 784).reshape(4, 784)
        original = images.clone()
        first = augment_views(images, torch.Generator().manual_seed(42))
        repeated = augment_views(images, torch.Generator().manual_seed(42))
        second = augment_views(images, torch.Generator().manual_seed(43))
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        torch.testing.assert_close(images, original, rtol=0, atol=0)
        self.assertEqual(first.shape, images.shape)
        self.assertTrue((first >= 0).all() and (first <= 1).all())
        self.assertFalse(torch.equal(first, second))

    def test_contrastive_loss_matches_uniform_case_and_pairing(self):
        identical = torch.ones(3, 4)
        self.assertAlmostEqual(contrastive_loss(identical, identical, 0.2).item(), np.log(5), places=6)
        orthogonal = torch.eye(4)
        matched = contrastive_loss(orthogonal, orthogonal, 0.1)
        mismatched = contrastive_loss(orthogonal, orthogonal.roll(1, 0), 0.1)
        self.assertLess(matched.item(), mismatched.item())
        with self.assertRaises(ValueError):
            contrastive_loss(orthogonal[:1], orthogonal[:1], 0.1)

    def test_contrastive_gradients_are_finite(self):
        first = torch.randn(4, 8, requires_grad=True)
        second = torch.randn(4, 8, requires_grad=True)
        loss = contrastive_loss(first, second, 0.2)
        loss.backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertTrue(torch.isfinite(second.grad).all())
        self.assertGreater(first.grad.abs().sum().item(), 0)

    def test_temperature_selection_uses_validation_and_lower_tie(self):
        rows = [{"bits": 16, "temperature": temperature, "validation_precision_at_10": score,
                 "test_precision_at_10": 1 - score}
                for temperature, score in [(0.1, 0.5), (0.2, 0.6), (0.5, 0.6)]]
        self.assertEqual(select_temperatures(rows, [16]), {"16": 0.2})

    def test_paired_training_and_frozen_contrastive_decoder(self):
        rng = np.random.default_rng(42)
        train = rng.random((16, 784), dtype=np.float32)
        validation = rng.random((8, 784), dtype=np.float32)
        torch.manual_seed(42)
        initial = Autoencoder(4)
        for objective in ["contrastive", "augmented_reconstruction"]:
            model, history = train_paired(train, validation, 4, 42, objective, 0.2, 2, 8, "cpu")
            self.assertEqual(len(history), 2)
            self.assertTrue(np.isfinite(history).all())
            self.assertTrue(np.isfinite(encode(model, validation, "cpu")).all())
            self.assertTrue(any(not torch.equal(p, q) for p, q in zip(model.encoder.parameters(), initial.encoder.parameters())))
            if objective == "contrastive":
                for p, q in zip(model.decoder.parameters(), initial.decoder.parameters()):
                    torch.testing.assert_close(p, q, rtol=0, atol=0)
                    self.assertFalse(p.requires_grad)

    def test_neural_itq_keeps_encoder_bytes_and_counts_parameters(self):
        rng = np.random.default_rng(42)
        train = rng.random((24, 784), dtype=np.float32)
        model = Autoencoder(4)
        latent = encode(model, train, "cpu")
        thresholds = np.median(latent, axis=0)
        with tempfile.TemporaryDirectory() as root:
            source, destination = Path(root) / "source", Path(root) / "rotated"
            save_artifact(source, "autoencoder", model, latent.mean(axis=0), thresholds, np.eye(4), thresholds)
            original = (source / "autoencoder.pt").read_bytes()
            result = fit_itq_artifact(train, source, destination, 42)
            self.assertEqual((source / "autoencoder.pt").read_bytes(), original)
            self.assertEqual((destination / "autoencoder.pt").read_bytes(), original)
            transform, arrays = load_artifact(destination)
            np.testing.assert_array_equal(saved_codes(transform, arrays, train, itq=True),
                                          (latent - arrays["latent_mean"]) @ arrays["rotation"] > 0)
            encoder_bytes = sum(p.numel() * p.element_size() for p in model.encoder.parameters())
            expected_bytes = encoder_bytes + sum(arrays[key].nbytes for key in
                                                 ["latent_mean", "rotation", "binary_thresholds"])
            self.assertEqual(result["inference_parameter_bytes"], expected_bytes)

    def test_latent_diagnostics_keep_threshold_controls_separate(self):
        rng = np.random.default_rng(42)
        database = rng.normal(size=(30, 4)) + 3
        queries = rng.normal(size=(8, 4)) + 3
        labels, query_labels = np.arange(30) % 3, np.arange(8) % 3
        mean, rotation, _ = fit_itq(database, 5, 42)
        thresholds = np.median(database, axis=0)
        result = latent_diagnostics(database, queries, labels, query_labels, thresholds, mean, rotation)
        self.assertEqual(len(result), 6)
        for name, db, q in [("median_binary", database > thresholds, queries > thresholds),
                            ("centered_sign", database > mean, queries > mean),
                            ("itq_binary", (database - mean) @ rotation > 0, (queries - mean) @ rotation > 0)]:
            expected = evaluate(db, q, labels, query_labels, 10)["same_class_precision_at_10"]
            self.assertEqual(result[name]["validation_precision_at_10"], expected)
        for metrics in result.values():
            self.assertGreaterEqual(metrics["validation_precision_at_10"], 0)
            self.assertLessEqual(metrics["validation_precision_at_10"], 1)
            self.assertNotIn("test_precision_at_10", metrics)

    def test_diagnostic_never_loads_test_queries_or_retrains_encoder(self):
        rng = np.random.default_rng(42)
        train = rng.random((16, 784), dtype=np.float32)
        validation = rng.random((8, 784), dtype=np.float32)
        queries = validation[:4]
        labels, query_labels = np.arange(16) % 2, np.arange(4) % 2
        model = Autoencoder(4)
        latent = encode(model, train, "cpu")
        thresholds = np.median(latent, axis=0)
        score = evaluate(latent > thresholds, encode(model, queries, "cpu") > thresholds,
                         labels, query_labels, 10)["same_class_precision_at_10"]
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source"
            directory = source / "model"
            save_artifact(directory, "autoencoder", model, latent.mean(axis=0), thresholds, np.eye(4), thresholds)
            report = {"config": {"tuning_dataset": "kmnist", "threshold_rule": "training_median",
                                  "tuning_seed": 42, "bits": [4], "train_size": 16, "validation_size": 8,
                                  "tuning_queries": 4, "smoke": True},
                      "versions": {"numpy": np.__version__, "torch": torch.__version__,
                                   "torchvision": torchvision.__version__, "sklearn": sklearn.__version__},
                      "selected_temperatures": {"4": 0.2},
                      "results": [{"bits": 4, "temperature": 0.2, "artifact_dir": str(directory),
                                   "validation_precision_at_10": score}]}
            (source / "tuning.json").write_text(json.dumps(report))
            args = SimpleNamespace(diagnose_from=source, reference_itq_from=None, device="cpu",
                                   data_dir=Path("unused"), output_dir=Path(root) / "diagnostic")
            ids = {"train": np.arange(16), "validation": np.arange(50, 58), "queries": np.arange(50, 54)}
            with patch("contrastive_experiment.load_data", return_value=(train, validation, queries, labels, query_labels, ids)) as loader:
                with patch("contrastive_experiment.train_paired", side_effect=AssertionError("Unexpected retraining")):
                    diagnose_selected(args)
                loader.assert_called_once_with(args.data_dir, 42, 16, 8, 4,
                                               query_split="validation", dataset="kmnist", return_ids=True)
            result = json.loads((args.output_dir / "diagnostics.json").read_text())
            self.assertTrue(result["source_files_unchanged"])
            self.assertTrue(result["results"][0]["median_score_matches_tuning"])

    def test_replication_uses_fixed_temperature_and_seed_without_retuning(self):
        rng = np.random.default_rng(7)
        train = rng.random((16, 784), dtype=np.float32)
        validation = rng.random((8, 784), dtype=np.float32)
        data = (train, validation, validation[:4], np.arange(16) % 2, np.arange(4) % 2,
                {"train": np.arange(16), "validation": np.arange(50, 58), "queries": np.arange(50, 54)})
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source"
            source.mkdir()
            report = {"config": {"tuning_dataset": "kmnist", "threshold_rule": "training_median",
                                  "tuning_seed": 42, "seeds": [42, 7], "bits": [4], "train_size": 16,
                                  "validation_size": 8, "tuning_queries": 4, "smoke": True,
                                  "quantization_weight": 0.001, "epochs": 1, "batch_size": 8},
                      "versions": {"numpy": np.__version__, "torch": torch.__version__,
                                   "torchvision": torchvision.__version__, "sklearn": sklearn.__version__},
                      "selected_temperatures": {"4": 0.2},
                      "results": [{"bits": 4, "temperature": 0.2, "artifact_dir": "unused-seed-42-model",
                                   "validation_precision_at_10": 0.5}]}
            (source / "tuning.json").write_text(json.dumps(report))
            args = SimpleNamespace(diagnose_from=source, diagnostic_seed=7, reference_itq_from=None,
                                   device="cpu", data_dir=Path("unused"), output_dir=Path(root) / "diagnostic")
            with patch("contrastive_experiment.load_data", return_value=data) as loader:
                with patch("contrastive_experiment.train_paired", wraps=train_paired) as trainer:
                    with patch("contrastive_experiment.select_temperatures", side_effect=AssertionError("Unexpected retuning")):
                        diagnose_selected(args)
                    trainer.assert_called_once()
                    self.assertEqual(trainer.call_args.args[2:], (4, 7, "contrastive", 0.2, 1, 8, "cpu"))
                loader.assert_called_once_with(args.data_dir, 7, 16, 8, 4,
                                               query_split="validation", dataset="kmnist", return_ids=True)
            result = json.loads((args.output_dir / "diagnostics.json").read_text())
            self.assertTrue(result["source_files_unchanged"])
            self.assertTrue(result["config"]["encoders_retrained"])
            self.assertEqual(result["selected_temperatures"], {"4": 0.2})
            self.assertIsNone(result["results"][0]["median_score_matches_tuning"])
            self.assertEqual(len(result["results"][0]["encoder_fit"]["validation_objective_by_epoch"]), 1)
            args.diagnostic_seed = 99
            args.output_dir = Path(root) / "invalid-seed"
            with self.assertRaisesRegex(ValueError, "frozen experiment's seed list"):
                diagnose_selected(args)

    def test_mnist_loader_uses_requested_official_dataset(self):
        class FakeDataset:
            data = torch.arange(60000).reshape(-1, 1).expand(-1, 784)
            targets = torch.arange(60000)

            def __len__(self):
                return 60000

        with patch("experiment.MNIST", return_value=FakeDataset()) as mnist:
            with patch("experiment.KMNIST") as kmnist, patch("experiment.FashionMNIST") as fashion:
                data = load_data("unused", 42, 16, 8, 4, query_split="validation", dataset="mnist")
                mnist.assert_called_once_with("unused", train=True, download=True)
                kmnist.assert_not_called()
                fashion.assert_not_called()
                np.testing.assert_array_equal(data[2], data[1][:4])


if __name__ == "__main__":
    unittest.main()
