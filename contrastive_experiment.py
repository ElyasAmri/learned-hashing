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

from benchmark_transfer import (evaluate_saved_databases, fit_itq_artifact, measure_cpu_encoding,
                                saved_codes, unseen_partitions)
from diagnose_quantization import continuous_neighbors, load_artifact, save_artifact
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
    if objective in ["contrastive", "contrastive_projection"]:
        embeddings = model.projection_head(latent) if objective == "contrastive_projection" else latent
        base = contrastive_loss(embeddings[:len(batch)], embeddings[len(batch):], temperature)
    elif objective == "augmented_reconstruction":
        base = F.mse_loss(model.decoder(latent), views)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return base, quantization_penalty(latent)


def train_paired(train, validation, bits, seed, objective, temperature, epochs, batch_size, device):
    torch.manual_seed(seed)
    model = Autoencoder(bits).to(device)
    if objective in ["contrastive", "contrastive_projection"]:
        model.decoder.requires_grad_(False)
    if objective == "contrastive_projection":
        model.projection_head = torch.nn.Sequential(torch.nn.Linear(bits, 128), torch.nn.ReLU(),
                                                    torch.nn.Linear(128, 128)).to(device)
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
    head = None
    if objective == "contrastive_projection":
        head = model.projection_head
        del model.projection_head
    latent = encode(model, train, device)
    thresholds = np.median(latent, axis=0)
    fit_seconds = time.perf_counter() - start
    save_artifact(directory, "autoencoder", model, latent.mean(axis=0), thresholds, np.eye(bits),
                  binary_thresholds=thresholds)
    if head is not None:
        with (directory / "projection_head.pt").open("xb") as output:
            torch.save({key: value.detach().cpu() for key, value in head.state_dict().items()}, output)
    transform, saved = load_artifact(directory, device)
    restored = transform(validation)
    expected = encode(model, validation, device)
    np.testing.assert_allclose(restored, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(restored > saved["binary_thresholds"], expected > thresholds)
    return {"fit_seconds": fit_seconds, "validation_objective_by_epoch": history,
            "projection_head": [bits, 128, 128] if head is not None else None,
            "training_only_parameter_bytes": sum(p.numel() * p.element_size() for p in head.parameters()) if head is not None else 0,
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


def latent_diagnostics(database, queries, labels, query_labels, thresholds, mean, rotation):
    retrieval = {}
    for name, metric, centered in [("continuous_euclidean", "euclidean", True),
                                  ("continuous_cosine", "cosine", False),
                                  ("centered_cosine", "cosine", True)]:
        start = time.perf_counter()
        db = database - mean if centered else database
        q = queries - mean if centered else queries
        neighbors = continuous_neighbors(db, q, metric)
        retrieval[name] = {"validation_precision_at_10": float((labels[neighbors] == query_labels[:, None]).mean()),
                           "search_seconds": time.perf_counter() - start}
    for name, db_codes, q_codes in [
        ("median_binary", database > thresholds, queries > thresholds),
        ("centered_sign", database > mean, queries > mean),
        ("itq_binary", (database - mean) @ rotation > 0, (queries - mean) @ rotation > 0),
    ]:
        metrics = evaluate(db_codes, q_codes, labels, query_labels, 10)
        metrics["validation_precision_at_10"] = metrics.pop("same_class_precision_at_10")
        retrieval[name] = metrics
    return retrieval


def diagnose_selected(args):
    if args.output_dir.exists():
        raise ValueError("Diagnostic output directory exists; choose a fresh path")
    tuning_path = args.diagnose_from / "tuning.json"
    tuning_bytes = tuning_path.read_bytes()
    tuning = json.loads(tuning_bytes)
    config = tuning["config"]
    versions = {"numpy": np.__version__, "torch": torch.__version__,
                "torchvision": torchvision.__version__, "sklearn": sklearn.__version__}
    if config["tuning_dataset"] != "kmnist" or config["threshold_rule"] != "training_median":
        raise ValueError("Diagnostics require KMNIST validation tuning with median thresholds")
    if versions != tuning["versions"]:
        raise ValueError("Library versions differ from the saved encoders")
    requested_seed = getattr(args, "diagnostic_seed", None)
    seed = config["tuning_seed"] if requested_seed is None else requested_seed
    if requested_seed is not None and seed not in config["seeds"]:
        raise ValueError("Diagnostic seed must belong to the frozen experiment's seed list")
    use_projection = getattr(args, "projection_head", False)
    objective = "contrastive_projection" if use_projection else "contrastive"
    train_new_encoder = seed != config["tuning_seed"] or use_projection
    if train_new_encoder and config["quantization_weight"] != QUANTIZATION_WEIGHT:
        raise ValueError("Quantization weight differs from the frozen training recipe")
    selected = [next(r for r in tuning["results"] if r["bits"] == bits
                     and r["temperature"] == tuning["selected_temperatures"][str(bits)]) for bits in config["bits"]]
    source_paths = [tuning_path]
    if not train_new_encoder:
        for result in selected:
            source_paths.extend(Path(result["artifact_dir"]) / name for name in ["transform.npz", "autoencoder.pt"])
    if args.reference_itq_from is not None:
        reference_path = args.reference_itq_from / "protocol.json"
        reference = json.loads(reference_path.read_text())
        if (reference["config"]["dataset"] != "kmnist" or reference["versions"] != versions
                or reference["source_config"]["train_size"] != config["train_size"]
                or seed not in reference["config"]["seeds"]):
            raise ValueError("PCA ITQ reference must match the KMNIST training split and versions")
        source_paths.append(reference_path)
        source_paths.extend(args.reference_itq_from / "models" / f"seed-{seed}" / f"pca_itq-{bits}" / "transform.npz"
                            for bits in config["bits"])
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    protocol = {"config": {"dataset": "kmnist", "query_split": "validation", "seed": seed,
                           "bits": config["bits"], "train_size": config["train_size"],
                           "validation_size": config["validation_size"], "query_size": config["tuning_queries"],
                           "itq_iterations": 50, "encoders_retrained": train_new_encoder, "device": args.device,
                           "training_objective": objective, "projection_head": use_projection,
                           "projection_head_width": 128 if use_projection else None,
                           "data_dir": str(args.data_dir), "smoke": config["smoke"]},
                "source_training_config": config,
                "selected_temperatures": tuning["selected_temperatures"], "versions": versions,
                "source_sha256": hashes}
    save_report(args.output_dir / "protocol.json", protocol)
    start = time.perf_counter()
    train, validation, queries, labels, query_labels, ids = load_data(
        args.data_dir, seed, config["train_size"], config["validation_size"], config["tuning_queries"],
        query_split="validation", dataset="kmnist", return_ids=True)
    results = []
    for original in selected:
        bits = original["bits"]
        print(f"Diagnosing contrastive KMNIST validation, {bits} bits", flush=True)
        source = Path(original["artifact_dir"])
        encoder_fit = None
        if train_new_encoder:
            source = args.output_dir / "encoders" / f"{objective}-{bits}"
            encoder_fit = fit_paired_artifact(train, validation, bits, seed, objective, original["temperature"],
                                              config, args.device, source)
            files = ["transform.npz", "autoencoder.pt"] + (["projection_head.pt"] if use_projection else [])
            for name in files:
                path = source / name
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        directory = args.output_dir / "models" / f"contrastive_itq-{bits}"
        extra = fit_itq_artifact(train, source, directory, seed, device=args.device)
        transform, arrays = load_artifact(source, args.device)
        latent, query_latent = transform(train), transform(queries)
        restored, rotated = load_artifact(directory, args.device)
        np.testing.assert_array_equal(saved_codes(restored, rotated, queries, itq=True),
                                      (query_latent - rotated["latent_mean"]) @ rotated["rotation"] > 0)
        metrics = latent_diagnostics(latent, query_latent, labels, query_labels, arrays["binary_thresholds"],
                                     rotated["latent_mean"], rotated["rotation"])
        if not train_new_encoder:
            np.testing.assert_equal(metrics["median_binary"]["validation_precision_at_10"],
                                    original["validation_precision_at_10"])
        if args.reference_itq_from is not None:
            reference_dir = args.reference_itq_from / "models" / f"seed-{seed}" / f"pca_itq-{bits}"
            project, pca_arrays = load_artifact(reference_dir)
            reference_metrics = evaluate(saved_codes(project, pca_arrays, train, itq=True),
                                         saved_codes(project, pca_arrays, queries, itq=True), labels, query_labels, 10)
            reference_metrics["validation_precision_at_10"] = reference_metrics.pop("same_class_precision_at_10")
            metrics["pca_itq_reference"] = reference_metrics
        results.append({"bits": bits, "temperature": original["temperature"], "retrieval": metrics,
                        "source_artifact_dir": str(source), "artifact_dir": str(directory),
                        "median_score_matches_tuning": None if train_new_encoder else True,
                        "encoder_fit": encoder_fit, **extra})
        print("  " + ", ".join(f"{name}={value['validation_precision_at_10']:.5f}" for name, value in metrics.items()), flush=True)
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed during diagnosis: {path}")
    save_report(args.output_dir / "diagnostics.json",
                {**protocol, "partitions": {name: values.tolist() for name, values in ids.items()},
                 "source_files_unchanged": True, "results": results, "elapsed_seconds": time.perf_counter() - start})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "tune", "evaluate"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("results/contrastive"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Reduced KMNIST-only validation run; never loads MNIST")
    parser.add_argument("--diagnose-from", type=Path, help="Diagnose selected KMNIST encoders with frozen settings")
    parser.add_argument("--diagnostic-seed", type=int, help="Replicate a frozen source seed; fit its encoder if not the tuning seed")
    parser.add_argument("--projection-head", action="store_true", help="Diagnostic ablation: train with a bits/128/128 head, then discard it")
    parser.add_argument("--reference-itq-from", type=Path, help="Saved KMNIST PCA ITQ benchmark for the diagnostic")
    args = parser.parse_args()
    if args.diagnose_from is not None:
        if args.stage != "all" or args.smoke:
            parser.error("Diagnostic mode uses the source report's settings, not --stage or --smoke")
        diagnose_selected(args)
        return
    if args.reference_itq_from is not None or args.diagnostic_seed is not None or args.projection_head:
        parser.error("--reference-itq-from, --diagnostic-seed and --projection-head require --diagnose-from")
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
