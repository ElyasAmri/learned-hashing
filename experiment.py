"""Fashion-MNIST binary similarity-hashing pilot."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST, KMNIST


POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


class Autoencoder(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, bits))
        self.decoder = nn.Sequential(nn.Linear(bits, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))


def encode(model, x, device, batch_size=1024):
    model.eval()
    with torch.no_grad():
        return np.concatenate([
            model.encoder(torch.from_numpy(x[start:start + batch_size]).to(device)).cpu().numpy()
            for start in range(0, len(x), batch_size)
        ])


def reconstruction_mse(model, x, device, batch_size=1024):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device)
            total += nn.functional.mse_loss(model(batch), batch, reduction="sum").item()
    return total / x.size


def pair_distances(left, right, penalty):
    left_log = nn.functional.log_softmax(left, dim=1)
    right_log = nn.functional.log_softmax(right, dim=1)
    left_p, right_p = left_log.exp(), right_log.exp()
    if penalty == "inverse_kl":
        return (0.5 * ((left_p - right_p) * (left_log - right_log)).sum(dim=1)).clamp_min(0)
    if penalty == "inverse_wasserstein":
        # W1 on equally spaced bins in [0, 1]; the mean accounts for bin spacing.
        return (left_p - right_p).cumsum(dim=1)[:, :-1].abs().mean(dim=1)
    raise ValueError(f"Unknown penalty: {penalty}")


def separation_penalty(latent, penalty, generator):
    if len(latent) < 2:
        return latent.sum() * 0
    # A randomly ordered cycle has no self-pairs, unlike a random permutation.
    order = torch.randperm(len(latent), generator=generator).to(latent.device)
    distances = pair_distances(latent[order], latent[order.roll(1)], penalty)
    return (distances + 1e-3).reciprocal().mean()


def quantization_penalty(latent):
    return (latent.abs() - 1).square().mean()


def train_autoencoder(train, validation, bits, epochs, batch_size, device, seed,
                      penalty=None, penalty_weight=0.0):
    torch.manual_seed(seed)
    model = Autoencoder(bits).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.from_numpy(train)), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_loss = float("inf")
    best_state = None
    history = []
    pair_generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if penalty is None or penalty_weight == 0:
                loss = nn.functional.mse_loss(model(batch), batch)
            else:
                latent = model.encoder(batch)
                regularizer = (quantization_penalty(latent) if penalty == "quantization"
                               else separation_penalty(latent, penalty, pair_generator))
                loss = nn.functional.mse_loss(model.decoder(latent), batch) + penalty_weight * regularizer
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
        val_loss = reconstruction_mse(model, validation, device)
        history.append(val_loss)
        print(f"  bits={bits} epoch={epoch + 1} validation_mse={val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def quantize(values, thresholds):
    return values > thresholds


def code_statistics(codes):
    packed = np.packbits(codes, axis=1)
    _, counts = np.unique(packed, axis=0, return_counts=True)
    balance = codes.mean(axis=0)
    centered = codes.astype(np.float64) - balance
    covariance = centered.T @ centered / len(codes)
    std = np.sqrt(np.diag(covariance))
    denominator = np.outer(std, std)
    correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
    valid_pairs = np.triu(denominator > 0, k=1)
    return {
        "unique_codes": len(counts),
        "duplicate_item_fraction": float(1 - len(counts) / len(codes)),
        "occupied_bucket_size_percentiles": dict(zip(
            ["p50", "p90", "p99", "max"], np.percentile(counts, [50, 90, 99, 100]).tolist())),
        "bit_one_fractions": balance.tolist(),
        "constant_bits": int(np.count_nonzero(std == 0)),
        "mean_absolute_nonconstant_bit_correlation": (
            float(np.abs(correlation[valid_pairs]).mean()) if valid_pairs.any() else None),
        "packed_database_bytes": packed.nbytes,
    }


def hamming_neighbors(database, queries, k=10, batch_size=32):
    """Exact search; ties use database row order, independent of labels."""
    database = np.packbits(database, axis=1)
    queries = np.packbits(queries, axis=1)
    neighbors = []
    for start in range(0, len(queries), batch_size):
        distances = POPCOUNT[np.bitwise_xor(queries[start:start + batch_size, None], database[None])].sum(axis=2)
        # Integer composite keys make ties deterministic without sorting the entire database.
        keys = distances.astype(np.int64) * len(database) + np.arange(len(database))
        candidates = np.argpartition(keys, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(keys, candidates, axis=1), axis=1)
        neighbors.append(np.take_along_axis(candidates, order, axis=1))
    return np.concatenate(neighbors)


def evaluate(database_codes, query_codes, database_labels, query_labels, k):
    start = time.perf_counter()
    neighbors = hamming_neighbors(database_codes, query_codes, k)
    search_seconds = time.perf_counter() - start
    return {
        **code_statistics(database_codes),
        f"same_class_precision_at_{k}": float((database_labels[neighbors] == query_labels[:, None]).mean()),
        "search_seconds": search_seconds,
        "search_queries_per_second": len(query_codes) / search_seconds,
    }


def load_data(root, seed, train_size, validation_size, query_size, query_split="test", dataset="fashion_mnist",
              return_ids=False):
    if dataset == "fashion_mnist":
        dataset_class = FashionMNIST
    elif dataset == "kmnist":
        dataset_class = KMNIST
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    official_train = dataset_class(root, train=True, download=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(official_train))
    # Keep validation disjoint and fixed even when reducing the training size.
    train_ids = order[:train_size]
    validation_ids = order[50000:50000 + validation_size]
    if query_split == "validation":
        if query_size > validation_size:
            raise ValueError("Validation queries exceed validation size")
        query_dataset = official_train
        query_ids = validation_ids[:query_size]
    elif query_split == "test":
        query_dataset = dataset_class(root, train=False, download=True)
        query_ids = rng.permutation(len(query_dataset))[:query_size]
    else:
        raise ValueError(f"Unknown query split: {query_split}")

    def images(dataset, ids):
        return dataset.data[ids].numpy().reshape(len(ids), 784).astype(np.float32) / 255.0

    data = (images(official_train, train_ids), images(official_train, validation_ids),
            images(query_dataset, query_ids), official_train.targets[train_ids].numpy(),
            query_dataset.targets[query_ids].numpy())
    if return_ids:
        return (*data, {"train": train_ids, "validation": validation_ids, "queries": query_ids})
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/pilot.json"))
    parser.add_argument("--bits", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--validation-size", type=int, default=10000)
    parser.add_argument("--query-size", type=int, default=10000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if not (1 <= args.train_size <= 50000 and 1 <= args.validation_size <= 10000 and 1 <= args.query_size <= 10000):
        parser.error("split sizes must be positive and within 50000/10000/10000")
    if not (1 <= args.k <= args.train_size and args.epochs > 0 and args.batch_size > 0):
        parser.error("require 1 <= k <= train-size, positive epochs and batch-size")
    if any(bits < 2 or bits > min(args.train_size, 784) for bits in args.bits):
        parser.error("bits must be between 2 and min(train-size, 784)")
    if args.output.exists():
        parser.error("output already exists; choose a fresh --output path")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, validation, queries, labels, query_labels = load_data(
        args.data_dir, args.seed, args.train_size, args.validation_size, args.query_size)
    results = []
    for bits in args.bits:
        for method in ["random_hyperplane", "pca", "autoencoder"]:
            print(f"Running {method}, {bits} bits", flush=True)
            start = time.perf_counter()
            extra = {}
            if method == "random_hyperplane":
                projection = np.random.default_rng(args.seed).standard_normal((784, bits)).astype(np.float32)
                mean = train.mean(axis=0)
                transform = lambda x: (x - mean) @ projection
                thresholds = np.zeros(bits, dtype=np.float32)
                model_bytes = projection.nbytes + mean.nbytes
            elif method == "pca":
                pca = PCA(n_components=bits, svd_solver="randomized", random_state=args.seed).fit(train)
                transform = pca.transform
                thresholds = np.median(transform(train), axis=0)
                model_bytes = pca.components_.nbytes + pca.mean_.nbytes
            else:
                model, history = train_autoencoder(train, validation, bits, args.epochs,
                                                   args.batch_size, args.device, args.seed)
                transform = lambda x: encode(model, x, args.device)
                thresholds = np.median(transform(train), axis=0)
                model_bytes = sum(p.numel() * p.element_size() for p in model.encoder.parameters())
                extra = {"validation_mse_by_epoch": history, "best_validation_mse": min(history)}
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            database_codes = quantize(transform(train), thresholds)
            database_encode_seconds = time.perf_counter() - start
            start = time.perf_counter()
            query_codes = quantize(transform(queries), thresholds)
            query_encode_seconds = time.perf_counter() - start
            result = {"method": method, "bits": bits, "fit_seconds": fit_seconds,
                      "database_encode_seconds": database_encode_seconds,
                      "query_encode_seconds": query_encode_seconds,
                      "inference_parameter_bytes": model_bytes + thresholds.nbytes,
                      **extra, **evaluate(database_codes, query_codes, labels, query_labels, args.k)}
            results.append(result)
            print(f"  precision@{args.k}={result[f'same_class_precision_at_{args.k}']:.4f}", flush=True)
    report = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "versions": {"numpy": np.__version__, "torch": torch.__version__,
                           "sklearn": __import__("sklearn").__version__,
                           "torchvision": __import__("torchvision").__version__},
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
