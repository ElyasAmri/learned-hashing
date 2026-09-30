"""Tune contrastive hashing on KMNIST validation and freeze it for MNIST."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision
from sklearn.decomposition import PCA
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from benchmark_transfer import evaluate_saved_databases, fit_itq_artifact, measure_cpu_encoding, unseen_partitions
from diagnose_quantization import load_artifact, save_artifact
from experiment import Autoencoder, encode, evaluate, load_data, quantization_penalty
from penalty_experiment import save_report


QUANTIZATION_WEIGHT = 0.001


def augment_views(images, generator):
    draws = torch.rand((len(images), 4), device=images.device, generator=generator)
    angle = (draws[:, 0] * 2 - 1) * (10 * torch.pi / 180)
    scale = 0.9 + 0.2 * draws[:, 1]
    theta = torch.zeros((len(images), 2, 3), device=images.device, dtype=images.dtype)
    theta[:, 0, 0] = angle.cos() / scale
    theta[:, 0, 1] = -angle.sin() / scale
    theta[:, 1, 0] = angle.sin() / scale
    theta[:, 1, 1] = angle.cos() / scale
    theta[:, :, 2] = (draws[:, 2:] * 2 - 1) * (4 / 28)
    image_grid = images.reshape(-1, 1, 28, 28)
    grid = F.affine_grid(theta, image_grid.shape, align_corners=False)
    return F.grid_sample(image_grid, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False).flatten(1)


def contrastive_loss(first, second, temperature):
    if len(first) != len(second) or len(first) < 2 or temperature <= 0:
        raise ValueError("Contrastive loss needs matching batches of at least two pairs and positive temperature")
    representations = F.normalize(torch.cat([first, second]), dim=1)
    logits = representations @ representations.T / temperature
    logits = logits.masked_fill(torch.eye(len(logits), dtype=torch.bool, device=logits.device), -torch.inf)
    targets = (torch.arange(len(logits), device=logits.device) + len(first)) % len(logits)
    return F.cross_entropy(logits, targets)


def paired_objective(model, batch, generator, objective, temperature):
    views = torch.cat([augment_views(batch, generator), augment_views(batch, generator)])
    latent = model.encoder(views)
    if objective == "contrastive":
        base = contrastive_loss(latent[:len(batch)], latent[len(batch):], temperature)
    elif objective == "augmented_reconstruction":
        base = F.mse_loss(model.decoder(latent), views)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return base, quantization_penalty(latent)


def train_paired(train, validation, bits, seed, objective, temperature, epochs, batch_size, device):
    torch.manual_seed(seed)
    model = Autoencoder(bits).to(device)
    if objective == "contrastive":
        model.decoder.requires_grad_(False)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.from_numpy(train)), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    training_rng = torch.Generator(device=device).manual_seed(seed + 17)
    history = []
    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            base, quantization = paired_objective(model, batch, training_rng, objective, temperature)
            loss = base + QUANTIZATION_WEIGHT * quantization
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite paired training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        validation_rng = torch.Generator(device=device).manual_seed(seed + 1000000)
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(validation), batch_size):
                batch = torch.from_numpy(validation[start:start + batch_size]).to(device)
                base, _ = paired_objective(model, batch, validation_rng, objective, temperature)
                total += base.item() * len(batch)
        value = total / len(validation)
        history.append(value)
        print(f"  {objective} bits={bits} epoch={epoch + 1} validation_objective={value:.6f}", flush=True)
        if value < best_loss:
            best_loss = value
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def fit_paired_artifact(train, validation, bits, seed, objective, temperature, config, device, directory):
    start = time.perf_counter()
    model, history = train_paired(train, validation, bits, seed, objective, temperature,
                                  config["epochs"], config["batch_size"], device)
    latent = encode(model, train, device)
    thresholds = np.median(latent, axis=0)
    fit_seconds = time.perf_counter() - start
    save_artifact(directory, "autoencoder", model, latent.mean(axis=0), thresholds, np.eye(bits),
                  binary_thresholds=thresholds)
    transform, saved = load_artifact(directory, device)
    restored = transform(validation)
    expected = encode(model, validation, device)
    np.testing.assert_allclose(restored, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(restored > saved["binary_thresholds"], expected > thresholds)
    return {"fit_seconds": fit_seconds, "validation_objective_by_epoch": history,
            "best_validation_objective": min(history), "artifact_roundtrip": "passed",
            "training_quantization_mse": float(np.square(np.abs(latent) - 1).mean()),
            "inference_parameter_bytes": sum(p.numel() * p.element_size() for p in model.encoder.parameters())
                                         + thresholds.nbytes}


def fit_pca_artifact(train, bits, seed, directory):
    start = time.perf_counter()
    model = PCA(n_components=bits, svd_solver="randomized", random_state=seed).fit(train)
    model.components_ = np.ascontiguousarray(model.components_)
    latent = model.transform(train)
    thresholds = np.median(latent, axis=0)
    fit_seconds = time.perf_counter() - start
    save_artifact(directory, "pca", model, latent.mean(axis=0), thresholds, np.eye(bits), binary_thresholds=thresholds)
    transform, saved = load_artifact(directory)
    np.testing.assert_array_equal(transform(train) > saved["binary_thresholds"], latent > thresholds)
    return {"fit_seconds": fit_seconds, "artifact_roundtrip": "passed",
            "inference_parameter_bytes": model.components_.nbytes + model.mean_.nbytes + thresholds.nbytes}


def select_temperatures(results, bits):
    return {str(b): max((r for r in results if r["bits"] == b),
                       key=lambda r: (r["validation_precision_at_10"], -r["temperature"]))["temperature"]
            for b in bits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "tune", "evaluate"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("results/contrastive"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Reduced KMNIST-only validation run; never loads MNIST")
    args = parser.parse_args()
    config = {"bits": [16, 32, 64], "seeds": [42, 7, 21, 84, 123], "epochs": 20, "batch_size": 256,
              "train_size": 50000, "validation_size": 10000, "tuning_queries": 2000, "query_pool_size": 10000,
              "temperatures": [0.1, 0.2, 0.5], "quantization_weight": QUANTIZATION_WEIGHT,
              "tuning_dataset": "kmnist", "evaluation_dataset": "mnist", "evaluation_query_split": "test",
              "threshold_rule": "training_median", "tuning_seed": 42, "smoke": args.smoke,
              "data_dir": str(args.data_dir), "device": args.device,
              "augmentation": {"rotation_degrees": 10, "scale": [0.9, 1.1], "grid_translation_pixels": 2,
                               "interpolation": "bilinear", "padding": "zeros", "flips": False},
              "checkpoint_selection": "minimum_fixed_view_validation_base_objective",
              "projection_head": False, "itq_iterations": 50}
    if args.smoke:
        config.update({"bits": [16], "seeds": [42], "epochs": 1, "train_size": 1000, "validation_size": 200,
                       "tuning_queries": 100, "query_pool_size": 200, "temperatures": [0.1, 0.2],
                       "evaluation_dataset": "kmnist", "evaluation_query_split": "validation"})
    versions = {"numpy": np.__version__, "torch": torch.__version__,
                "torchvision": torchvision.__version__, "sklearn": sklearn.__version__}
    protocol = {"config": config, "versions": versions}
    if args.stage != "evaluate":
        if args.output_dir.exists():
            parser.error("Output directory exists; choose a fresh path")
        save_report(args.output_dir / "protocol.json", protocol)
    else:
        recorded = json.loads((args.output_dir / "protocol.json").read_text())
        if recorded != protocol:
            parser.error("Configuration or versions differ from tuning")
    if args.stage != "tune":
        if any((args.output_dir / f"seed-{s}.json").exists() or (args.output_dir / "evaluation" / f"seed-{s}").exists()
               for s in config["seeds"]):
            parser.error("Evaluation outputs already exist")
    start = time.perf_counter()
    tuning_path = args.output_dir / "tuning.json"
    if args.stage != "evaluate":
        train, validation, queries, labels, query_labels = load_data(
            args.data_dir, 42, config["train_size"], config["validation_size"], config["tuning_queries"],
            query_split="validation", dataset="kmnist")
        results = []
        for bits in config["bits"]:
            for temperature in config["temperatures"]:
                print(f"Tuning KMNIST bits={bits}, temperature={temperature}", flush=True)
                directory = args.output_dir / "tuning-models" / f"bits-{bits}-temperature-{temperature}"
                extra = fit_paired_artifact(train, validation, bits, 42, "contrastive", temperature,
                                            config, args.device, directory)
                transform, arrays = load_artifact(directory, args.device)
                metrics = evaluate(transform(train) > arrays["binary_thresholds"],
                                   transform(queries) > arrays["binary_thresholds"], labels, query_labels, 10)
                score = metrics.pop("same_class_precision_at_10")
                results.append({"bits": bits, "temperature": temperature, "validation_precision_at_10": score,
                                "artifact_dir": str(directory), **extra, **metrics})
                print(f"  validation precision@10={score:.5f}", flush=True)
        save_report(tuning_path, {**protocol, "results": results,
                                  "selected_temperatures": select_temperatures(results, config["bits"])})
    tuning_bytes = tuning_path.read_bytes()
    tuning = json.loads(tuning_bytes)
    if tuning["config"] != config or tuning["versions"] != versions:
        parser.error("Tuning report differs from the reserved configuration")
    if args.stage != "tune":
        print(f"Frozen temperatures: {tuning['selected_temperatures']}", flush=True)
        for seed in config["seeds"]:
            data = load_data(args.data_dir, seed, config["train_size"], config["validation_size"],
                             config["query_pool_size"], query_split=config["evaluation_query_split"],
                             dataset=config["evaluation_dataset"], return_ids=True)
            train, validation = data[:2]
            partitions = unseen_partitions(data)
            if args.smoke:
                for name in ["unseen_database", "queries"]:
                    partitions[name]["official_split"] = "train"
            results = []
            for bits in config["bits"]:
                pca_directory = args.output_dir / "evaluation" / f"seed-{seed}" / f"pca-{bits}"
                pca_extra = fit_pca_artifact(train, bits, seed, pca_directory)
                for method in ["pca", "pca_itq", "augmented_reconstruction", "contrastive"]:
                    print(f"Evaluating {config['evaluation_dataset']} seed={seed}, {method}, bits={bits}", flush=True)
                    directory = args.output_dir / "evaluation" / f"seed-{seed}" / f"{method}-{bits}"
                    temperature = tuning["selected_temperatures"][str(bits)]
                    if method == "pca":
                        extra = pca_extra
                    elif method == "pca_itq":
                        extra = fit_itq_artifact(train, pca_directory, directory, seed)
                        extra["base_pca_fit_seconds"] = pca_extra["fit_seconds"]
                    else:
                        extra = fit_paired_artifact(train, validation, bits, seed, method, temperature,
                                                    config, args.device, directory)
                    metrics = evaluate_saved_databases(partitions, directory, args.device, itq=method == "pca_itq")
                    costs = measure_cpu_encoding(directory, partitions["queries"]["images"], itq=method == "pca_itq")
                    results.append({"method": method, "bits": bits, "seed": seed,
                                    "temperature": temperature if method == "contrastive" else None,
                                    "artifact_dir": str(directory), "retrieval": metrics, "costs": costs, **extra})
                    print(f"  unseen precision@10={metrics['unseen_database']['test_precision_at_10']:.5f}", flush=True)
            partition_records = {name: {"official_split": values["official_split"], "ids": values["ids"].tolist(),
                                        "class_counts": np.bincount(values["labels"], minlength=10).tolist()}
                                 for name, values in partitions.items()}
            save_report(args.output_dir / f"seed-{seed}.json",
                        {**protocol, "seed": seed, "selected_temperatures": tuning["selected_temperatures"],
                         "tuning_sha256": hashlib.sha256(tuning_bytes).hexdigest(),
                         "partitions": partition_records, "results": results})
    print(f"Elapsed seconds: {time.perf_counter() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
