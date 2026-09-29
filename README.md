# Learned hashing pilot

Test whether an autoencoder's binary latent codes preserve Fashion-MNIST clothing categories better than simple projection baselines at the same bit budget. This is a similarity-search experiment, not an exact-key hash-table benchmark.

## Run (PowerShell)

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m unittest -v

# Reduced real-data smoke test; not a quality benchmark.
.venv/Scripts/python experiment.py --train-size 1000 --validation-size 200 --query-size 100 --epochs 1 --bits 16 --output results/smoke.json

# Full pilot: three methods at 16, 32, and 64 bits.
.venv/Scripts/python experiment.py --output results/pilot.json
```

The first experiment run downloads Fashion-MNIST into `data/`. Reports go to ignored `results/`; use a fresh output filename for each run. CPU is the default. On this RTX 5080 machine, install the tested CUDA 12.8 wheels into the existing environment, then run:

```powershell
.venv/Scripts/python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python experiment.py --device cuda --output results/pilot-gpu.json
```

Only autoencoder training and encoding use CUDA. PCA, random projections, and exact Hamming search stay on the CPU.

## Protocol

- Seed 42 shuffles the official 60,000 training images: the first 50,000 are both training set and retrieval database; the last 10,000 are validation. Reduced runs take prefixes of these same partitions.
- The official 10,000 test images are unseen queries, shuffled with the same seeded generator. Test labels are used only for evaluation.
- Inputs are flattened 784-dimensional float32 vectors scaled to `[0, 1]`.
- Random hyperplanes: subtract the training mean, apply seeded Gaussian projections, threshold at zero. This is an angular-similarity baseline on centered inputs.
- PCA: fit randomized PCA on training inputs; threshold each component at its training median.
- Autoencoder: `784 -> 128 -> bits -> 128 -> 784`, ReLU hidden layers, linear latent layer, sigmoid output. Train with reconstruction MSE and Adam at learning rate 0.001 for 20 epochs, batch size 256. Select the lowest validation-MSE checkpoint, then threshold latent coordinates at their training medians. Labels never enter training.
- Retrieve exactly by Hamming distance, with ties broken by seeded database row order rather than labels. There is no approximate index and no query/database overlap.

## Report

Primary metric: **same-class precision@10**, averaged over test queries. Category agreement is only a coarse proxy for semantic similarity; this metric is not recall of pixel-space neighbors.

Code diagnostics: unique codes, duplicate-item fraction `(N - unique_codes) / N`, occupied-bucket size percentiles, per-bit one fractions, constant bits, and mean absolute correlation between nonconstant bits. Duplicate-item fraction is not pairwise collision probability. Do not interpret low collisions as good retrieval, or sparse occupancy of the entire 64-bit space as failure.

Storage: packed database bytes and inference-parameter bytes (encoder/projection, centering where applicable, and thresholds). These exclude labels, index metadata, runtime objects, and the autoencoder decoder. Float-valued latent vectors are not counted as binary hashes.

Timing: fit including threshold calibration, database encoding, query encoding, and exact search. These are single-run wall times, not controlled performance benchmarks. JSON includes configuration, library versions, and autoencoder validation history; it does not save trained models.

## Separation penalties

```powershell
# Tune on validation only; then apply the fixed weights to five test seeds.
.venv/Scripts/python penalty_experiment.py --stage tune
.venv/Scripts/python penalty_experiment.py --stage test
```

The runner also supports `--stage all` (default). Use `--output-dir` for a fresh run; existing reports are never overwritten. Test-only runs load `tuning.json` from that directory and check training settings and library versions. `--seeds` can select remaining test seeds after an interrupted run.

Both variants minimize `reconstruction_MSE + weight * mean(1 / (0.001 + distance))`. Distances compare `softmax(z)` distributions (temperature 1), not raw latent-vector geometry:

- **Inverse KL:** half the sum of both directed KL divergences, evaluated with log-softmax for stability.
- **Inverse Wasserstein:** exact 1D Wasserstein-1 on latent-coordinate bins equally spaced over `[0, 1]`, computed from cumulative probability differences. Coordinate ordering is an imposed assumption.

Each minibatch uses a seeded random cycle to pair distinct samples without self-pairs. A singleton batch contributes zero separation penalty. A separate pairing RNG preserves the original initialization and minibatch ordering. Decoder inputs and binary thresholds still use raw latent coordinates, not softmax outputs. As for the baseline, checkpoints minimize validation reconstruction MSE.

Weights `1e-6`, `1e-5`, `1e-4`, and `1e-3` are selected separately per penalty and bit length using precision@10 on the first 2,000 validation images at seed 42, against all 50,000 training/database images. Each candidate trains for 20 epochs. Ties favor the smaller weight. No test images are loaded during tuning. Selected weights are fixed for test seeds 42, 7, 21, 84, and 123. Validation labels therefore tune these variants, although they do not enter gradient training; the original baselines did not receive a comparable weight search.

Softmax does not bound KL divergence: sharpening logits can increase it. Wasserstein here is bounded by 1. Neither objective guarantees semantic separation or uniform binary codes, and both repel same-class as well as different-class pairs. Identical distributions have zero separation gradient despite the reciprocal penalty being high. Reports retain every tuning score, selected weight, test metric, and reconstruction history, but not model weights.

## Five-seed results

Seeds 42, 7, 21, 84, and 123; default splits and 20 epochs; RTX 5080 / Ryzen 9 9950X. The first run took 2 minutes 26 seconds; four additional runs took 8 minutes 27 seconds combined. Environment: torch 2.11.0+cu128, torchvision 0.26.0+cu128, numpy 2.2.6, scikit-learn 1.7.2. Local reports: `results/pilot-gpu.json` (42) and `results/pilot-gpu-seed-{seed}.json` (others).

Precision@10 below is mean +/- sample standard deviation across five runs, in percent; deviations and paired gaps are in percentage points.

| Bits | Random precision@10 | PCA precision@10 | Autoencoder precision@10 | AE + inverse KL | AE + inverse Wasserstein |
|---|---:|---:|---:|---:|---:|
| 16 | 51.24 +/- 1.28 | 65.63 +/- 0.11 | 61.86 +/- 1.32 | 62.62 +/- 0.79 | 62.76 +/- 1.31 |
| 32 | 62.95 +/- 0.48 | 72.99 +/- 0.11 | 70.17 +/- 0.53 | 70.03 +/- 0.66 | 70.24 +/- 0.47 |
| 64 | 70.47 +/- 1.33 | 76.78 +/- 0.10 | 75.20 +/- 0.22 | 75.36 +/- 0.34 | 75.48 +/- 0.51 |

The original three columns reuse the earlier five-seed reports. Penalty tuning took 4 minutes 51 seconds (24 candidates), and the five-seed evaluation took 11 minutes 21 seconds (30 models). Reports are in `results/penalties/tuning.json` and `results/penalties/seed-{seed}.json`. Every model trained for 20 epochs. Selected weights:

| Bits | Inverse KL weight | Inverse Wasserstein weight |
|---|---:|---:|
| 16 | 1e-4 | 1e-5 |
| 32 | 1e-4 | 1e-5 |
| 64 | 1e-6 | 1e-5 |

PCA beat all autoencoder variants in every run at each bit budget. Paired gains over the plain autoencoder, mean +/- sample standard deviation in percentage points:

| Bits | Inverse KL gain | Inverse Wasserstein gain | KL wins / 5 | Wasserstein wins / 5 |
|---|---:|---:|---:|---:|
| 16 | +0.77 +/- 1.67 | +0.90 +/- 2.27 | 4 | 4 |
| 32 | -0.14 +/- 0.90 | +0.07 +/- 0.66 | 2 | 4 |
| 64 | +0.17 +/- 0.29 | +0.28 +/- 0.54 | 3 | 4 |

Mean unique database codes across seeds:

| Bits | PCA | Autoencoder | AE + inverse KL | AE + inverse Wasserstein |
|---|---:|---:|---:|---:|
| 16 | 11,744.4 | 5,012.6 | 5,227.2 | 5,285.4 |
| 32 | 46,424.8 | 28,339.8 | 29,447.6 | 28,822.6 |
| 64 | 49,973.8 | 46,724.0 | 47,307.8 | 47,211.6 |

Both penalties modestly increased average code utilization, but neither reliably improved precision in every seed. At 64 bits, mean absolute bit correlation was 0.017 for PCA, 0.211 for the plain autoencoder, 0.208 for inverse KL, and 0.207 for inverse Wasserstein.

Changing the seed changes the training/validation split, database row order for Hamming ties, and model randomness together. All runs use the same official test images, so these are repeatability measurements, not independent test datasets or confidence intervals. More correlated autoencoder bits and fewer unique codes do not establish the cause of the retrieval gap. Models were not saved; reports retain metrics and validation histories.

## Next decision

The reciprocal-distance penalties did not close the gap to PCA; increasing their weights blindly is not supported by this sweep. The selected 64-bit KL weight lies at the lower search boundary, and selection used only one seed and a coarse four-weight grid, so this is not an exhaustive optimization. Next, compare continuous-latent retrieval against binary-code retrieval on validation queries before deciding whether to change the representation objective or the quantization. Do not tune against the already-inspected test results. Larger benchmarks and an unseen retrieval database remain necessary; Fashion-MNIST alone does not support a state-of-the-art claim.
