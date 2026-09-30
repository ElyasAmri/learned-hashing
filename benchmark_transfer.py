"""Evaluate a frozen Fashion-MNIST configuration on KMNIST without retuning."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sklearn
import torch
import torchvision
from sklearn.decomposition import PCA

from diagnose_quantization import load_artifact, save_artifact
from experiment import evaluate, load_data, quantize
from penalty_experiment import run_variant, save_report


def frozen_settings(source, smoke=False):
    source_config = source["config"]
    if (source_config["penalties"] != ["quantization"]
            or source_config["quantization_threshold"] != "median"
            or source_config["query_split"] != "validation"
            or source_config["checkpoint_selection"] != "minimum_validation_reconstruction_mse"
            or source_config.get("dataset", "fashion_mnist") != "fashion_mnist"):
        raise ValueError("Source must be a median-threshold Fashion-MNIST validation tuning report")
    config = {key: source_config[key] for key in ["epochs", "batch_size", "train_size", "validation_size",
                                                "query_size", "bits", "seeds"]}
    config.update({"dataset": "kmnist", "query_split": "test", "quantization_threshold": "median",
                   "checkpoint_selection": source_config["checkpoint_selection"], "smoke": smoke})
    if smoke:
        config.update({"epochs": 1, "train_size": 1000, "validation_size": 200,
                       "query_size": 100, "bits": [16], "seeds": [42], "query_split": "validation"})
    weights = {str(bits): source["selected_weights"][str(bits)]["quantization"] for bits in config["bits"]}
    if any(not np.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("Frozen weights must be finite and positive")
    return config, weights


def run_pca(data, bits, seed, split, artifact_dir):
    train, _, queries, labels, query_labels = data
    start = time.perf_counter()
    model = PCA(n_components=bits, svd_solver="randomized", random_state=seed).fit(train)
    model.components_ = np.ascontiguousarray(model.components_)
    latent = model.transform(train)
    thresholds = np.median(latent, axis=0)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    query_latent = model.transform(queries)
    query_codes = quantize(query_latent, thresholds)
    query_seconds = time.perf_counter() - start
    save_artifact(artifact_dir, "pca", model, latent.mean(axis=0), thresholds,
                  np.eye(bits), binary_thresholds=thresholds)
    transform, saved = load_artifact(artifact_dir)
    restored = transform(queries)
    np.testing.assert_array_equal(restored, query_latent)
    np.testing.assert_array_equal(quantize(restored, saved["binary_thresholds"]), query_codes)
    metrics = evaluate(quantize(latent, thresholds), query_codes, labels, query_labels, 10)
    metrics[f"{split}_precision_at_10"] = metrics.pop("same_class_precision_at_10")
    return {"method": "pca", "bits": bits, "seed": seed, "threshold_rule": "median",
            "fit_seconds": fit_seconds, "query_encode_seconds": query_seconds,
            "inference_parameter_bytes": model.components_.nbytes + model.mean_.nbytes + thresholds.nbytes,
            "artifact_dir": str(artifact_dir), "artifact_roundtrip": "passed", **metrics}


def unseen_partitions(data):
    train, _, test, train_labels, test_labels, ids = data
    size = len(test) // 2
    if len(test) % 2 or size < 10 or len(train) < size:
        raise ValueError("Need equal test halves and a matched training database of at least 10 items")
    if np.intersect1d(ids["queries"][:size], ids["queries"][size:]).size:
        raise ValueError("Unseen database and query IDs overlap")
    return {
        "training_database": {"images": train[:size], "labels": train_labels[:size],
                              "ids": ids["train"][:size], "official_split": "train"},
        "unseen_database": {"images": test[:size], "labels": test_labels[:size],
                            "ids": ids["queries"][:size], "official_split": "test"},
        "queries": {"images": test[size:], "labels": test_labels[size:],
                    "ids": ids["queries"][size:], "official_split": "test"},
    }


def evaluate_saved_databases(partitions, directory, device):
    transform, saved = load_artifact(directory, device)
    # Thresholds are reused from training, never fitted to either database or queries.
    thresholds = saved["binary_thresholds"]
    np.testing.assert_array_equal(thresholds, saved["median_thresholds"])
    queries = partitions["queries"]
    query_codes = quantize(transform(queries["images"]), thresholds)
    results = {}
    for name in ["training_database", "unseen_database"]:
        database = partitions[name]
        database_codes = quantize(transform(database["images"]), thresholds)
        metrics = evaluate(database_codes, query_codes, database["labels"], queries["labels"], 10)
        metrics["test_precision_at_10"] = metrics.pop("same_class_precision_at_10")
        results[name] = metrics
    return results


def evaluate_unseen_database(args, versions):
    source_path = args.unseen_database_from / "protocol.json"
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    config = source["config"]
    if (config["dataset"] != "kmnist" or config["smoke"] or config["query_split"] != "test"
            or config["query_size"] != 10000 or config["train_size"] != 50000):
        raise ValueError("Unseen evaluation requires a full frozen KMNIST benchmark")
    if versions != source["versions"]:
        raise ValueError("Library versions differ from saved models")
    methods = ["pca", "autoencoder", "quantization"]
    artifact_hashes = {}
    for seed in config["seeds"]:
        for bits in config["bits"]:
            for method in methods:
                relative = Path("models") / f"seed-{seed}" / f"{method}-{bits}"
                files = ["transform.npz"] if method == "pca" else ["transform.npz", "autoencoder.pt"]
                for filename in files:
                    path = relative / filename
                    artifact_hashes[str(path)] = hashlib.sha256((args.unseen_database_from / path).read_bytes()).hexdigest()
    manifest = {"config": {"dataset": "kmnist", "seeds": config["seeds"], "bits": config["bits"],
                           "database_size": 5000, "query_size": 5000, "device": args.device,
                           "data_dir": str(args.data_dir), "threshold_rule": "saved_training_median",
                           "test_partition": "first_half_database_second_half_queries",
                           "training_database": "first_5000_training_rows", "retrained": False},
                "versions": versions, "fixed_weights": source["fixed_weights"],
                "source_protocol": str(source_path), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "source_config": config, "artifact_sha256": artifact_hashes}
    save_report(args.output_dir / "protocol.json", manifest)
    start = time.perf_counter()
    for seed in config["seeds"]:
        data = load_data(args.data_dir, seed, config["train_size"], config["validation_size"],
                         config["query_size"], dataset="kmnist", return_ids=True)
        partitions = unseen_partitions(data)
        results = []
        for bits in config["bits"]:
            for method in methods:
                directory = args.unseen_database_from / "models" / f"seed-{seed}" / f"{method}-{bits}"
                metrics = evaluate_saved_databases(partitions, directory, args.device)
                results.append({"method": method, "bits": bits, "seed": seed, "artifact_dir": str(directory),
                                "retrieval": metrics})
                print(f"seed={seed} {method} {bits} bits: " + ", ".join(
                    f"{name}={values['test_precision_at_10']:.5f}" for name, values in metrics.items()), flush=True)
        partition_records = {name: {"official_split": values["official_split"], "ids": values["ids"].tolist(),
                                    "class_counts": np.bincount(values["labels"], minlength=10).tolist()}
                             for name, values in partitions.items()}
        save_report(args.output_dir / f"seed-{seed}.json",
                    {"config": {**manifest["config"], "seed": seed}, "partitions": partition_records,
                     "results": results})
    for relative, expected in artifact_hashes.items():
        actual = hashlib.sha256((args.unseen_database_from / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Source artifact changed during evaluation: {relative}")
    save_report(args.output_dir / "completion.json",
                {"source_artifacts_unchanged": True, "artifact_files_checked": len(artifact_hashes),
                 "model_evaluations": len(config["seeds"]) * len(config["bits"]) * len(methods),
                 "elapsed_seconds": time.perf_counter() - start})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/qat-median/tuning.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/kmnist-frozen"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Reduced training and validation queries only")
    parser.add_argument("--unseen-database-from", type=Path, help="Evaluate existing frozen KMNIST models without training")
    args = parser.parse_args()
    versions = {"numpy": np.__version__, "torch": torch.__version__,
                "torchvision": torchvision.__version__, "sklearn": sklearn.__version__}
    if args.output_dir.exists():
        parser.error("Output directory exists; choose a fresh path")
    if args.unseen_database_from is not None:
        if args.smoke:
            parser.error("Unseen-database evaluation cannot use smoke models or splits")
        evaluate_unseen_database(args, versions)
        return
    source_bytes = args.source.read_bytes()
    source = json.loads(source_bytes)
    config, weights = frozen_settings(source, args.smoke)
    config.update({"data_dir": str(args.data_dir), "device": args.device})
    if versions != source["versions"]:
        parser.error("Library versions differ from the frozen source report")
    manifest = {"config": config, "fixed_weights": weights, "versions": versions,
                "source_report": str(args.source), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "source_config": source["config"]}
    # Record the fixed protocol before loading or evaluating any new benchmark data.
    save_report(args.output_dir / "protocol.json", manifest)
    train_args = SimpleNamespace(**config)
    metric_key = f"{config['query_split']}_precision_at_10"
    start = time.perf_counter()
    for seed in config["seeds"]:
        data = load_data(args.data_dir, seed, config["train_size"], config["validation_size"],
                         config["query_size"], query_split=config["query_split"], dataset="kmnist")
        results = []
        for bits in config["bits"]:
            for method in ["pca", "autoencoder", "quantization"]:
                print(f"Evaluating KMNIST {config['query_split']} seed={seed}, {method}, {bits} bits", flush=True)
                directory = args.output_dir / "models" / f"seed-{seed}" / f"{method}-{bits}"
                if method == "pca":
                    result = run_pca(data, bits, seed, config["query_split"], directory)
                else:
                    penalty = "quantization" if method == "quantization" else None
                    weight = weights[str(bits)] if penalty else 0.0
                    result = run_variant(data, bits, penalty, weight, train_args, seed, config["query_split"], directory)
                    result.update({"method": method, "threshold_rule": "median"})
                results.append(result)
                print(f"  precision@10={result[metric_key]:.5f}", flush=True)
        save_report(args.output_dir / f"seed-{seed}.json",
                    {**manifest, "config": {**config, "seed": seed}, "results": results})
    print(f"Elapsed seconds: {time.perf_counter() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
