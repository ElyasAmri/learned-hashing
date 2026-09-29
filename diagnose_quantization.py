"""Diagnose continuous versus binary retrieval and ITQ using validation only."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision
from sklearn.decomposition import PCA

from experiment import Autoencoder, encode, evaluate, load_data, quantize, train_autoencoder


def fit_itq(latent, iterations=50, seed=42):
    mean = latent.mean(axis=0, dtype=np.float64)
    centered = latent.astype(np.float64) - mean
    rotation, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((latent.shape[1], latent.shape[1])))
    history = [float(np.square(np.abs(centered @ rotation) - 1).mean())]
    for _ in range(iterations):
        binary = np.where(centered @ rotation > 0, 1.0, -1.0)
        u, _, vt = np.linalg.svd(centered.T @ binary, full_matrices=False)
        rotation = u @ vt
        history.append(float(np.square(np.abs(centered @ rotation) - 1).mean()))
    return mean, rotation, history


def continuous_neighbors(database, queries, metric, k=10, batch_size=64):
    database = database.astype(np.float64)
    queries = queries.astype(np.float64)
    if metric == "cosine":
        database /= np.maximum(np.linalg.norm(database, axis=1, keepdims=True), 1e-12)
        queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    elif metric != "euclidean":
        raise ValueError(f"Unknown metric: {metric}")
    database_norm = np.square(database).sum(axis=1)
    neighbors = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        product = batch @ database.T
        if metric == "euclidean":
            distances = np.maximum(np.square(batch).sum(axis=1)[:, None] + database_norm - 2 * product, 0)
        else:
            distances = 1 - product
        cutoffs = np.partition(distances, k - 1, axis=1)[:, k - 1]
        for row, cutoff in zip(distances, cutoffs):
            # Include boundary ties, then order by distance and database row index.
            candidates = np.flatnonzero(row <= cutoff)
            order = np.lexsort((candidates, row[candidates]))[:k]
            neighbors.append(candidates[order])
    return np.stack(neighbors)


def save_artifact(directory, method, model, mean, thresholds, rotation):
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {"method": method, "latent_mean": mean, "median_thresholds": thresholds, "rotation": rotation}
    if method == "pca":
        arrays.update({"input_mean": model.mean_, "components": model.components_})
    else:
        with (directory / "autoencoder.pt").open("xb") as output:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output)
    with (directory / "transform.npz").open("xb") as output:
        np.savez(output, **arrays)


def load_artifact(directory, device="cpu"):
    with np.load(Path(directory) / "transform.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if arrays["method"].item() == "pca":
        components = arrays["components"]
        offset = arrays["input_mean"].reshape(1, -1) @ components.T
        transform = lambda x: x @ components.T - offset
    else:
        model = Autoencoder(len(arrays["latent_mean"]))
        model.load_state_dict(torch.load(Path(directory) / "autoencoder.pt", map_location="cpu", weights_only=True))
        model.to(device)
        transform = lambda x: encode(model, x, device)
    return transform, arrays


def diagnose(train, validation, queries, labels, query_labels, method, bits, seed, args):
    start = time.perf_counter()
    extra = {}
    if method == "pca":
        model = PCA(n_components=bits, svd_solver="randomized", random_state=seed).fit(train)
        # Match the saved array layout so rounding near hash thresholds is reproducible.
        model.components_ = np.ascontiguousarray(model.components_)
        transform = model.transform
    else:
        model, history = train_autoencoder(train, validation, bits, args.epochs, args.batch_size, args.device, seed)
        transform = lambda x: encode(model, x, args.device)
        extra = {"validation_mse_by_epoch": history, "best_validation_mse": min(history)}
    database_latent = transform(train)
    query_latent = transform(queries)
    fit_seconds = time.perf_counter() - start
    thresholds = np.median(database_latent, axis=0)
    start = time.perf_counter()
    mean, rotation, errors = fit_itq(database_latent, args.itq_iterations, seed)
    itq_seconds = time.perf_counter() - start
    artifact_dir = args.output_dir / f"seed-{seed}" / f"{method}-{bits}"
    save_artifact(artifact_dir, method, model, mean, thresholds, rotation)
    restored, saved = load_artifact(artifact_dir, args.device)
    restored_queries = restored(queries)
    np.testing.assert_allclose(restored_queries, query_latent, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(quantize(restored_queries, saved["median_thresholds"]),
                                  quantize(query_latent, thresholds))
    np.testing.assert_array_equal((restored_queries - saved["latent_mean"]) @ saved["rotation"] > 0,
                                  (query_latent - mean) @ rotation > 0)
    centered_database, centered_queries = database_latent - mean, query_latent - mean
    result = {"method": method, "bits": bits, "seed": seed, "fit_seconds": fit_seconds,
              "itq_seconds": itq_seconds, "itq_quantization_mse_by_iteration": errors,
              "artifact_dir": str(artifact_dir), "artifact_roundtrip": "passed", **extra, "retrieval": {}}
    for metric in ["euclidean", "cosine"]:
        start = time.perf_counter()
        neighbors = continuous_neighbors(centered_database, centered_queries, metric)
        result["retrieval"][f"continuous_{metric}"] = {
            "validation_precision_at_10": float((labels[neighbors] == query_labels[:, None]).mean()),
            "search_seconds": time.perf_counter() - start,
            "database_representation_bytes": database_latent.nbytes,
        }
    for name, database_codes, query_codes in [
        ("median_binary", quantize(database_latent, thresholds), quantize(query_latent, thresholds)),
        ("centered_sign", centered_database > 0, centered_queries > 0),
        ("itq_binary", centered_database @ rotation > 0, centered_queries @ rotation > 0),
    ]:
        metrics = evaluate(database_codes, query_codes, labels, query_labels, 10)
        metrics["validation_precision_at_10"] = metrics.pop("same_class_precision_at_10")
        result["retrieval"][name] = metrics
    print(f"  {method} {bits} bits: " + ", ".join(
        f"{name}={metrics['validation_precision_at_10']:.5f}" for name, metrics in result["retrieval"].items()), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/quantization"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--bits", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 21, 84, 123])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--itq-iterations", type=int, default=50)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--validation-size", type=int, default=10000)
    parser.add_argument("--query-size", type=int, default=10000)
    args = parser.parse_args()
    if not (10 <= args.train_size <= 50000 and 1 <= args.query_size <= args.validation_size <= 10000
            and args.epochs > 0 and args.batch_size > 0 and args.itq_iterations > 0):
        parser.error("Invalid split sizes, epoch count, batch size, or iteration count")
    if any(b < 2 or b > min(args.train_size, 784) for b in args.bits):
        parser.error("bits must be between 2 and min(train-size, 784)")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.bits)) != len(args.bits):
        parser.error("seeds and bits must be unique")
    if any((args.output_dir / f"seed-{seed}").exists() or (args.output_dir / f"seed-{seed}.json").exists()
           for seed in args.seeds):
        parser.error("Seed output already exists; choose a fresh directory or unrun seeds")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"query_split": "validation", "cosine_centering": "training_latent_mean",
                   "itq_preprocessing": "training_mean_centering_only", "k": 10})
    versions = {"numpy": np.__version__, "torch": torch.__version__,
                "torchvision": torchvision.__version__, "sklearn": sklearn.__version__}
    start = time.perf_counter()
    for seed in args.seeds:
        data = load_data(args.data_dir, seed, args.train_size, args.validation_size,
                         args.query_size, query_split="validation")
        results = []
        for bits in args.bits:
            for method in ["pca", "autoencoder"]:
                print(f"Diagnosing seed={seed}, {method}, {bits} bits", flush=True)
                results.append(diagnose(*data, method, bits, seed, args))
        path = args.output_dir / f"seed-{seed}.json"
        with path.open("x", encoding="utf-8") as output:
            json.dump({"config": {**config, "seed": seed}, "versions": versions, "results": results},
                      output, indent=2, allow_nan=False)
        print(f"Wrote {path}", flush=True)
    print(f"Elapsed seconds: {time.perf_counter() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
