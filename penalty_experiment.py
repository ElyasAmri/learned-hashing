"""Tune autoencoder penalties on validation, then evaluate fixed weights."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision

from diagnose_quantization import load_artifact, save_artifact
from experiment import encode, evaluate, load_data, quantize, train_autoencoder


PENALTIES = ["inverse_kl", "inverse_wasserstein"]


def select_weights(results, bits, penalties=PENALTIES):
    selected = {}
    for bit_count in bits:
        selected[str(bit_count)] = {}
        for penalty in penalties:
            candidates = [r for r in results if r["bits"] == bit_count and r["method"] == penalty]
            best = max(candidates, key=lambda r: (r["validation_precision_at_10"], -r["penalty_weight"]))
            selected[str(bit_count)][penalty] = best["penalty_weight"]
    return selected


def run_variant(data, bits, penalty, weight, args, seed, split, artifact_dir=None):
    train, validation, queries, labels, query_labels = data
    start = time.perf_counter()
    model, history = train_autoencoder(train, validation, bits, args.epochs, args.batch_size,
                                       args.device, seed, penalty, weight)
    latent = encode(model, train, args.device)
    median_thresholds = np.median(latent, axis=0)
    thresholds = np.zeros(bits, dtype=np.float32) if penalty == "quantization" else median_thresholds
    fit_seconds = time.perf_counter() - start
    database_codes = quantize(latent, thresholds)
    start = time.perf_counter()
    query_latent = encode(model, queries, args.device)
    query_codes = quantize(query_latent, thresholds)
    query_seconds = time.perf_counter() - start
    metrics = evaluate(database_codes, query_codes, labels, query_labels, 10)
    metrics[f"{split}_precision_at_10"] = metrics.pop("same_class_precision_at_10")
    extra = {}
    if penalty == "quantization":
        median_metrics = evaluate(quantize(latent, median_thresholds), quantize(query_latent, median_thresholds),
                                  labels, query_labels, 10)
        median_metrics[f"{split}_precision_at_10"] = median_metrics.pop("same_class_precision_at_10")
        extra = {"threshold_rule": "zero", "median_binary": median_metrics,
                 "training_quantization_mse": float(np.square(np.abs(latent) - 1).mean()),
                 "query_quantization_mse": float(np.square(np.abs(query_latent) - 1).mean())}
    if artifact_dir is not None:
        # Identity rotation is a placeholder: these models do not use ITQ.
        save_artifact(artifact_dir, "autoencoder", model, latent.mean(axis=0), median_thresholds,
                      np.eye(bits), binary_thresholds=thresholds)
        transform, saved = load_artifact(artifact_dir, args.device)
        restored = transform(queries)
        np.testing.assert_allclose(restored, query_latent, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(quantize(restored, saved["binary_thresholds"]), query_codes)
        np.testing.assert_array_equal(quantize(restored, saved["median_thresholds"]),
                                      quantize(query_latent, median_thresholds))
        extra.update({"artifact_dir": str(artifact_dir), "artifact_roundtrip": "passed"})
    return {"method": penalty, "bits": bits, "seed": seed, "penalty_weight": weight,
            "validation_mse_by_epoch": history, "best_validation_mse": min(history),
            "fit_seconds": fit_seconds, "query_encode_seconds": query_seconds,
            "inference_parameter_bytes": sum(p.numel() * p.element_size() for p in model.encoder.parameters())
                                         + thresholds.nbytes,
            **extra, **metrics}


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(f"Wrote {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "tune", "test"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("results/penalties"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--bits", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 21, 84, 123])
    parser.add_argument("--weights", type=float, nargs="+", default=[1e-6, 1e-5, 1e-4, 1e-3])
    parser.add_argument("--penalties", nargs="+", choices=PENALTIES + ["quantization"], default=PENALTIES)
    parser.add_argument("--query-split", choices=["test", "validation"], default="test")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--validation-size", type=int, default=10000)
    parser.add_argument("--tuning-queries", type=int, default=2000)
    parser.add_argument("--query-size", type=int, default=10000)
    args = parser.parse_args()
    if not (10 <= args.train_size <= 50000 and 1 <= args.tuning_queries <= args.validation_size <= 10000
            and 1 <= args.query_size <= 10000 and args.epochs > 0 and args.batch_size >= 2):
        parser.error("Invalid split sizes, epoch count, or batch size")
    if any(b < 2 or b > min(args.train_size, 784) for b in args.bits):
        parser.error("bits must be between 2 and min(train-size, 784)")
    if any(not np.isfinite(w) or w <= 0 for w in args.weights):
        parser.error("weights must be finite and positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.bits)) != len(args.bits):
        parser.error("seeds and bits must be unique")
    if args.query_split == "validation" and args.query_size > args.validation_size:
        parser.error("Validation queries exceed validation size")
    if len(set(args.penalties)) != len(args.penalties):
        parser.error("penalties must be unique")
    tuning_path = args.output_dir / "tuning.json"
    outputs = ([] if args.stage == "test" else [tuning_path])
    if args.stage != "tune":
        outputs += [args.output_dir / f"seed-{s}.json" for s in args.seeds]
        if args.save_models:
            outputs += [args.output_dir / "models" / f"seed-{s}" for s in args.seeds]
    if any(p.exists() for p in outputs):
        parser.error("An output already exists; choose a fresh output directory or unrun test seeds")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"temperature": 1.0, "epsilon": 1e-3, "tuning_seed": 42,
                   "checkpoint_selection": "minimum_validation_reconstruction_mse"})
    versions = {"numpy": np.__version__, "torch": torch.__version__,
                "torchvision": torchvision.__version__, "sklearn": sklearn.__version__}
    start = time.perf_counter()
    if args.stage != "test":
        data = load_data(args.data_dir, 42, args.train_size, args.validation_size,
                         args.tuning_queries, query_split="validation")
        results = []
        for bits in args.bits:
            for penalty in args.penalties:
                for weight in args.weights:
                    print(f"Tuning {penalty}, {bits} bits, weight={weight}", flush=True)
                    result = run_variant(data, bits, penalty, weight, args, 42, "validation")
                    results.append(result)
                    print(f"  validation precision@10={result['validation_precision_at_10']:.5f}", flush=True)
        tuning = {"config": config, "versions": versions, "results": results,
                  "selected_weights": select_weights(results, args.bits, args.penalties)}
        save_report(tuning_path, tuning)
    else:
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        for key in ["data_dir", "bits", "epochs", "batch_size", "train_size", "validation_size",
                    "temperature", "epsilon", "checkpoint_selection"]:
            if tuning["config"][key] != config[key]:
                parser.error(f"Test configuration differs from tuning: {key}")
        if tuning["config"].get("penalties", PENALTIES) != args.penalties:
            parser.error("Penalty selection differs from tuning")
        if tuning["versions"] != versions:
            parser.error("Library versions differ from tuning")
    if args.stage != "tune":
        print(f"Fixed weights: {tuning['selected_weights']}", flush=True)
        for seed in args.seeds:
            data = load_data(args.data_dir, seed, args.train_size, args.validation_size, args.query_size,
                             query_split=args.query_split)
            results = []
            for bits in args.bits:
                for penalty in args.penalties:
                    weight = tuning["selected_weights"][str(bits)][penalty]
                    print(f"Evaluating {args.query_split} seed={seed} {penalty}, {bits} bits, weight={weight}", flush=True)
                    artifact_dir = (args.output_dir / "models" / f"seed-{seed}" / f"{penalty}-{bits}"
                                    if args.save_models else None)
                    result = run_variant(data, bits, penalty, weight, args, seed, args.query_split, artifact_dir)
                    results.append(result)
                    print(f"  {args.query_split} precision@10={result[f'{args.query_split}_precision_at_10']:.5f}", flush=True)
            save_report(args.output_dir / f"seed-{seed}.json",
                        {"config": {**config, "seed": seed}, "versions": versions,
                         "selected_weights": tuning["selected_weights"], "results": results})
    print(f"Elapsed seconds: {time.perf_counter() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
