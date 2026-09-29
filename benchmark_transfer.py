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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/qat-median/tuning.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/kmnist-frozen"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Reduced training and validation queries only")
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    source = json.loads(source_bytes)
    config, weights = frozen_settings(source, args.smoke)
    config.update({"data_dir": str(args.data_dir), "device": args.device})
    versions = {"numpy": np.__version__, "torch": torch.__version__,
                "torchvision": torchvision.__version__, "sklearn": sklearn.__version__}
    if versions != source["versions"]:
        parser.error("Library versions differ from the frozen source report")
    if args.output_dir.exists():
        parser.error("Output directory exists; choose a fresh path")
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
