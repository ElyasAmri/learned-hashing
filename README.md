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

Timing: fit including threshold calibration, database encoding, query encoding, and exact search. These are single-run wall times, not controlled performance benchmarks. JSON includes configuration, library versions, and autoencoder validation history. The original pilot and penalty runners do not save trained models; the quantization diagnostic below does.

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

Changing the seed changes the training/validation split, database row order for Hamming ties, and model randomness together. All runs use the same official test images, so these are repeatability measurements, not independent test datasets or confidence intervals. More correlated autoencoder bits and fewer unique codes do not establish the cause of the retrieval gap. These earlier experiments did not save models; reports retain metrics and validation histories.

## Quantization diagnostic: validation only

```powershell
.venv/Scripts/python diagnose_quantization.py --output-dir results/quantization-next
```

Uses the same five seeds, 50,000 training/database images, 10,000 validation queries, and 20 autoencoder epochs. No official test images are loaded. These queries also select autoencoder checkpoints by reconstruction MSE; results are development diagnostics, not fresh test estimates. The plain PCA and autoencoder are fitted once per seed and bit length, then reused for all five retrieval variants:

- Continuous Euclidean and cosine retrieval: subtract the training latent mean for both; centering leaves Euclidean distances unchanged but changes cosine distances. Exact search uses float64 arithmetic and breaks ties by database row index. No whitening or row normalization is applied before Euclidean search.
- Median binary: the original training-median threshold rule.
- Centered sign: zero-threshold codes after training-mean centering, without rotation. This controls for ITQ's threshold change.
- ITQ binary: 50 alternating sign/orthogonal-Procrustes iterations on centered training latents, starting from a seeded random orthogonal rotation, then zero-threshold codes. No per-coordinate scaling or whitening; neither validation data nor labels fit the rotation. Orthogonal rotations preserve Euclidean distances before binarization.

All search and ITQ work runs on CPU; autoencoder training/encoding uses CUDA. Continuous representations use float coordinates, not the same storage budget as binary codes. Continuous-to-binary gaps compare both representation and retrieval metric, not an isolated information-loss measurement.

The completed run took 22 minutes 5 seconds. Mean +/- sample standard deviation of **validation precision@10 (%)** across five seeds:

| Bits | Encoder | Continuous Euclidean | Continuous cosine | Median binary | Centered sign | ITQ binary |
|---|---|---:|---:|---:|---:|---:|
| 16 | PCA | 78.49 +/- 0.26 | 79.09 +/- 0.31 | 66.38 +/- 0.33 | 66.54 +/- 0.34 | 64.80 +/- 0.52 |
| 16 | AE | 79.66 +/- 0.34 | 79.93 +/- 0.36 | 62.42 +/- 1.43 | 61.87 +/- 0.96 | 62.67 +/- 1.50 |
| 32 | PCA | 80.43 +/- 0.36 | 81.02 +/- 0.34 | 73.54 +/- 0.26 | 73.62 +/- 0.28 | 71.73 +/- 0.52 |
| 32 | AE | 80.62 +/- 0.26 | 81.17 +/- 0.30 | 70.87 +/- 0.73 | 70.57 +/- 0.52 | 71.76 +/- 0.71 |
| 64 | PCA | 81.19 +/- 0.33 | 81.86 +/- 0.32 | 77.49 +/- 0.47 | 77.48 +/- 0.37 | 76.29 +/- 0.28 |
| 64 | AE | 81.08 +/- 0.35 | 81.61 +/- 0.33 | 75.90 +/- 0.23 | 75.60 +/- 0.42 | 76.43 +/- 0.56 |

AE continuous retrieval was competitive with PCA, but its median-code gap was larger, especially at 16 bits (17.51 points below centered cosine versus PCA's 12.71). ITQ improved AE over median codes by 0.25, 0.90, and 0.53 points on average; it won in 2/5, 4/5, and 5/5 seeds respectively. PCA ITQ lost to median codes in every seed. Quantization objective values decreased in all 30 fits, showing that reducing that objective alone does not guarantee better category retrieval. The earlier penalty search used a coarse grid and did not close the gap to PCA; it was not an exhaustive optimization.

Completed reports and artifacts are in `results/quantization-v2/`. Each seed has a JSON report plus `seed-{seed}/{method}-{bits}/transform.npz`; AE directories also contain `autoencoder.pt` with the selected full state dict. NPZ files hold training latent means, median thresholds, ITQ rotations, and PCA input means/components. Input preprocessing is still flattened float32 pixels in `[0, 1]`. `load_artifact(directory, device)` in `diagnose_quantization.py` returns an encoder callable and these arrays. Encode new inputs, then use either `z > median_thresholds` or `(z - latent_mean) @ rotation > 0`.

Every completed artifact was reloaded and checked against all 10,000 validation query encodings and both median/ITQ codes. An initial run in `results/quantization/` stopped on one changed PCA bit after reload: identical coefficients with a different memory layout changed float32 rounding near a threshold. PCA components now use a contiguous layout before encoding and saving; the full rerun passed all 30 round trips. The interrupted directory remains intact and is not part of the result table.

## Next decision

Keep PCA with median thresholds as the strongest binary baseline here. For the autoencoder, test quantization-aware training or bit decorrelation next: its continuous representation is useful, but this ITQ configuration does not close the binary-retrieval gap. Choose changes using validation data, not the already-inspected test results. Larger benchmarks and an unseen retrieval database remain necessary; Fashion-MNIST alone does not support a state-of-the-art claim.
