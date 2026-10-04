# Learned hashing pilot

[Complete experiment summary](EXPERIMENT_SUMMARY.md): every configuration, tuning candidate, result, and cost comparison. Original completed JSON reports are included under [results/](results/).

Test whether an autoencoder's binary latent codes preserve category similarity better than projection baselines at the same bit budget. Fashion-MNIST and KMNIST support development experiments, with frozen-configuration evaluations on KMNIST and MNIST. This is similarity search, not an exact-key hash-table benchmark.

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

The first experiment run downloads Fashion-MNIST into `data/`. Completed JSON reports are published under `results/`; datasets, checkpoints, smoke reports, and interrupted runs stay ignored. Use fresh output filenames/directories for reruns. Commands that reuse saved model artifacts require regenerating those models first; published JSON reports alone do not provide model weights. CPU is the default for the original pilot. On this RTX 5080 machine, install the tested CUDA 12.8 wheels into the existing environment, then run:

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

Timing: fit including threshold calibration, database encoding, query encoding, and exact search. These are single-run wall times, not controlled performance benchmarks. JSON includes configuration, library versions, and autoencoder validation history. The original pilot does not save models; the penalty runner saves selected models with `--save-models`, and the quantization diagnostic always saves them.

## Separation penalties

```powershell
# Tune on validation only; then apply the fixed weights to five test seeds.
.venv/Scripts/python penalty_experiment.py --stage tune
.venv/Scripts/python penalty_experiment.py --stage test
```

The runner also supports `--stage all` (default). Use `--output-dir` for a fresh run; existing reports are never overwritten. Test-only runs load `tuning.json` from that directory and check training settings, penalty and quantization-threshold choices, and library versions. `--seeds` can select remaining evaluation seeds after an interrupted run. `--query-split validation` keeps final evaluation on validation data even when the stage is named `test`. `--penalties` selects which losses to run; defaults remain the two reciprocal-distance penalties.

Both variants minimize `reconstruction_MSE + weight * mean(1 / (0.001 + distance))`. Distances compare `softmax(z)` distributions (temperature 1), not raw latent-vector geometry:

- **Inverse KL:** half the sum of both directed KL divergences, evaluated with log-softmax for stability.
- **Inverse Wasserstein:** exact 1D Wasserstein-1 on latent-coordinate bins equally spaced over `[0, 1]`, computed from cumulative probability differences. Coordinate ordering is an imposed assumption.

Each minibatch uses a seeded random cycle to pair distinct samples without self-pairs. A singleton batch contributes zero separation penalty. A separate pairing RNG preserves the original initialization and minibatch ordering. Decoder inputs and binary thresholds still use raw latent coordinates, not softmax outputs. As for the baseline, checkpoints minimize validation reconstruction MSE.

Weights `1e-6`, `1e-5`, `1e-4`, and `1e-3` are selected separately per penalty and bit length using precision@10 on the first 2,000 validation images at seed 42, against all 50,000 training/database images. Each candidate trains for 20 epochs. Ties favor the smaller weight. No test images are loaded during tuning. Selected weights are fixed for test seeds 42, 7, 21, 84, and 123. Validation labels therefore tune these variants, although they do not enter gradient training; the original baselines did not receive a comparable weight search.

Softmax does not bound KL divergence: sharpening logits can increase it. Wasserstein here is bounded by 1. Neither objective guarantees semantic separation or uniform binary codes, and both repel same-class as well as different-class pairs. Identical distributions have zero separation gradient despite the reciprocal penalty being high. Reports retain every tuning score, selected weight, evaluation metric, and reconstruction history. The original reciprocal-distance runs did not save models.

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

## Quantization-aware training: validation only

```powershell
.venv/Scripts/python penalty_experiment.py --penalties quantization --query-split validation --save-models --weights 0.001 0.01 0.1 1.0 --output-dir results/qat-next
```

The architecture and decoder inputs remain unchanged. Training adds `weight * mean((abs(z) - 1)^2)` to reconstruction MSE, encouraging raw latent coordinates toward either binary target. This is soft quantization regularization, not a hard binary bottleneck or a straight-through estimator. It does not enforce bit balance or decorrelation; the absolute-value implementation has zero gradient at exactly zero.

Weights are selected separately for each bit length using **zero-threshold** precision@10 on the first 2,000 validation images at seed 42. The grid is `0.001, 0.01, 0.1, 1.0`; ties favor smaller weights. Checkpoints still minimize validation reconstruction MSE. Selected weights are fixed for five seeds and all 10,000 validation queries. Median-threshold retrieval is a prespecified secondary diagnostic, not used to choose weights. No official test images are loaded. These are development results: the final validation queries overlap tuning and checkpoint selection, and the baselines did not receive a comparable loss-weight search.

The run took 10 minutes 25 seconds: 12 tuning candidates and 15 selected-weight models, each trained for 20 epochs. The smallest weight, `0.001`, was selected at all three bit lengths, motivating the smaller-weight sweep below. Mean +/- sample standard deviation of validation precision@10 (%):

| Bits | PCA median | Plain AE median | AE + ITQ | Quantization AE zero (primary) | Quantization AE median (diagnostic) |
|---|---:|---:|---:|---:|---:|
| 16 | 66.38 +/- 0.33 | 62.42 +/- 1.43 | 62.67 +/- 1.50 | 64.95 +/- 1.29 | 67.06 +/- 0.72 |
| 32 | 73.54 +/- 0.26 | 70.87 +/- 0.73 | 71.76 +/- 0.71 | 69.14 +/- 0.64 | 73.18 +/- 0.62 |
| 64 | 77.49 +/- 0.47 | 75.90 +/- 0.23 | 76.43 +/- 0.56 | 73.39 +/- 0.67 | 77.27 +/- 0.60 |

The first three columns reuse `results/quantization-v2/`. Primary zero-threshold retrieval changed by +2.52, -1.73, and -2.50 points relative to the plain median-code AE. No primary run beat PCA. With median thresholds, the regularized AE improved over the plain AE in all five seeds at each bit length; mean gains were +4.64, +2.32, and +1.38 points. It beat PCA in 4/5 seeds at 16 bits and 2/5 seeds at 32 and 64 bits; the 16-bit mean advantage was 0.68 points, not a significance claim.

At weights 0.1 and 1.0, every database item received the same zero-threshold code at all three bit lengths during tuning (9.865% precision@10). Even at the selected weight, mean absolute bit imbalance from 50% was 0.187, 0.235, and 0.254 for 16/32/64-bit zero-threshold codes. Pulling values toward binary targets alone does not ensure useful or balanced codes. The median diagnostic is promising but does not change the mixed primary outcome.

Reports are in `results/qat/`; selected models are in `results/qat/models/seed-{seed}/quantization-{bits}/`. Use `load_artifact` to obtain the encoder and arrays. Primary codes are `z > binary_thresholds` (saved zeros); diagnostic codes are `z > median_thresholds`. These artifacts contain an identity rotation only for format compatibility, not a fitted ITQ transform. All 15 artifacts passed reload checks on all 10,000 query encodings and both code types. Tuning candidates were not saved.

## Smaller weights with fixed median thresholds

```powershell
.venv/Scripts/python penalty_experiment.py --penalties quantization --quantization-threshold median --query-split validation --save-models --weights 0.00001 0.00003 0.0001 0.0003 0.001 --output-dir results/qat-median-next
```

The threshold rule is fixed before both selection and evaluation; every threshold is a training-set median. `--quantization-threshold` defaults to `zero` for previous commands. A resumed evaluation rejects a different threshold rule from its tuning report. All other training, seed, checkpoint, and validation settings match the preceding experiment. Weights are selected on 2,000 validation queries at seed 42, then fixed for five-seed evaluation on all 10,000 validation queries. No official test images are loaded.

The run took 7 minutes 31 seconds: 15 tuning candidates and 15 selected-weight models, each trained for 20 epochs. Mean +/- sample standard deviation of validation precision@10 (%):

| Bits | PCA median | Plain AE median | Previous quantization AE median | Tuned quantization AE median | Selected weight |
|---|---:|---:|---:|---:|---:|
| 16 | 66.38 +/- 0.33 | 62.42 +/- 1.43 | 67.06 +/- 0.72 | 67.06 +/- 0.72 | 0.001 |
| 32 | 73.54 +/- 0.26 | 70.87 +/- 0.73 | 73.18 +/- 0.62 | 73.63 +/- 0.31 | 0.0003 |
| 64 | 77.49 +/- 0.47 | 75.90 +/- 0.23 | 77.27 +/- 0.60 | 77.73 +/- 0.41 | 0.0003 |

The 16-bit weight and scores are unchanged. At 32 and 64 bits, smaller weights improved the mean over the preceding median diagnostic by 0.45 and 0.46 points. Paired mean gains over PCA were 0.68, 0.09, and 0.25 points, with wins in 4/5 seeds at each bit length. Gains over the plain AE were 4.64, 2.77, and 1.84 points, with wins in all five seeds. These small PCA margins are not significance claims: development queries have been reused, seed-42 tuning overlaps evaluation, and PCA did not receive comparable hyperparameter search.

Reports and all 15 selected model artifacts are in `results/qat-median/`; every artifact passed reload checks on all 10,000 validation encodings and codes. Both saved `binary_thresholds` and `median_thresholds` are the training medians. The first two comparison columns reuse `results/quantization-v2/`, and the previous quantization column uses `results/qat/`. Models were retrained for the new sweep, including the unchanged 16-bit reference.

## Frozen-configuration KMNIST benchmark

```powershell
# Smoke uses only KMNIST validation queries, never official test queries.
.venv/Scripts/python benchmark_transfer.py --smoke --output-dir results/kmnist-smoke-next
.venv/Scripts/python benchmark_transfer.py --output-dir results/kmnist-frozen-next
```

`benchmark_transfer.py` reads `results/qat-median/tuning.json` by default (`--source` selects another compatible report). It writes `protocol.json` with the source SHA-256, fixed weights, training settings, and library versions before loading the benchmark. It requires a fresh output directory and the same library versions as the source. There is no weight-search option.

KMNIST contains 28x28 grayscale Japanese characters in ten classes. Models are trained from scratch on KMNIST; this transfers hyperparameters, not learned weights or zero-shot predictions. The architecture, preprocessing, 20 epochs, batch size 256, training-median thresholds, and weights 0.001/0.0003/0.0003 for 16/32/64 bits remain fixed from Fashion-MNIST. Each of the same five seeds uses 50,000 official training images as both fit set and retrieval database, 10,000 remaining training images for validation-MSE checkpoint selection, and the official 10,000 test images as queries. No test labels select models or settings. PCA and the plain AE use the same splits and bit budgets.

The initial smoke run stopped before training because Python's downloader could not validate the official server's TLS certificate. Archives were then downloaded from `https://codh.rois.ac.jp/kmnist/dataset/kmnist/` through Windows certificate-checked HTTPS and matched against all four torchvision-supplied MD5 digests. No TLS checks were disabled. The retry used the local cache and completed validation-only evaluation; the frozen settings were not changed after the smoke results.

The full run took 12 minutes 53 seconds. **KMNIST test precision@10 (%)**, mean +/- sample standard deviation across five seeds:

| Bits | PCA median | Plain AE median | Frozen quantization AE median | Gain over PCA (points) |
|---|---:|---:|---:|---:|
| 16 | 53.74 +/- 0.13 | 53.94 +/- 0.89 | 57.06 +/- 0.28 | +3.32 |
| 32 | 65.41 +/- 0.12 | 63.11 +/- 0.69 | 65.13 +/- 0.19 | -0.28 |
| 64 | 71.81 +/- 0.09 | 72.24 +/- 0.33 | 73.44 +/- 0.34 | +1.63 |

The quantization AE beat the plain AE in all five seeds at every bit length, with mean gains of 3.13, 2.02, and 1.19 points. Against PCA it won 5/5 seeds at 16 bits, 2/5 at 32 bits, and 5/5 at 64 bits. This supports transfer of the quantization benefit over the plain AE, but not uniform superiority to PCA or a state-of-the-art claim. Seed variation reuses the same test images and is not a confidence interval over independent datasets. The retrieval database in this experiment contains training items; the follow-up below evaluates unseen database items. Broader baseline tuning remains unchecked.

Reports, the pre-run protocol, and all 45 fitted model/transform artifacts are in `results/kmnist-frozen/`. Every artifact passed query-encoding and binary-code reload checks on all 10,000 test queries. `results/kmnist-smoke/` retains the failed download's protocol only; `results/kmnist-smoke-v2/` contains the successful smoke results. Do not merge these KMNIST test scores with the preceding Fashion-MNIST validation tables.

## Unseen retrieval database: frozen models

```powershell
.venv/Scripts/python benchmark_transfer.py --unseen-database-from results/kmnist-frozen --output-dir results/kmnist-unseen-next
```

Without `--itq-and-costs`, this mode only loads saved models; it does not train, fit PCA, select checkpoints, or recalibrate thresholds. For each seed, the original seeded KMNIST test permutation is split into its first 5,000 images as the unseen database and last 5,000 as queries. The query and unseen-database IDs are disjoint. A matched control uses the first 5,000 images from that model's training permutation as the database, against the exact same query codes. All thresholds remain medians from the original 50,000 training images.

The protocol is written before evaluation, including source-protocol and artifact SHA-256 hashes. Seed reports save the official split, row IDs, and class counts of all three partitions. `completion.json` records that all 75 source files (45 transforms and 30 AE state dicts) remained unchanged. The run evaluated 45 saved models against both databases in 65 seconds, without creating new model artifacts. Reports are in `results/kmnist-unseen/`.

Precision@10 (%) below is mean +/- sample standard deviation over five seeds. Each database has 5,000 images, and each run has 5,000 queries:

| Bits | Model | Training database control | Unseen database |
|---|---|---:|---:|
| 16 | PCA | 46.11 +/- 0.61 | 48.72 +/- 0.34 |
| 16 | Plain AE | 46.70 +/- 0.94 | 50.00 +/- 1.17 |
| 16 | Quantization AE | 49.35 +/- 1.08 | 52.99 +/- 0.38 |
| 32 | PCA | 51.60 +/- 0.53 | 57.62 +/- 0.18 |
| 32 | Plain AE | 51.28 +/- 0.85 | 56.78 +/- 0.91 |
| 32 | Quantization AE | 52.53 +/- 0.52 | 58.14 +/- 0.39 |
| 64 | PCA | 52.51 +/- 0.45 | 60.52 +/- 0.22 |
| 64 | Plain AE | 56.98 +/- 0.73 | 64.93 +/- 0.46 |
| 64 | Quantization AE | 57.64 +/- 0.91 | 65.03 +/- 0.45 |

On the unseen database, quantization AE beat PCA in all five seeds at every bit length; mean gains were 4.27, 0.52, and 4.51 points. Gains over the plain AE were 2.99, 1.36, and 0.10 points, with wins in 5/5, 5/5, and 3/5 seeds. The 64-bit quantization benefit is therefore small and inconsistent, despite the advantage over PCA.

All methods scored higher on the unseen database than on the matched training control. Both unseen database and queries come from the official test split, whereas the control database comes from the official training split. Equal sizes and shared queries control database size and query identity, not differences in split distributions, class composition, or sampling; the gap cannot be attributed solely to training exposure. These are follow-up diagnostics on a previously inspected test pool, not a new untouched benchmark. Query pools overlap across seeds, and these deviations are not confidence intervals over independent datasets. Do not compare these absolute scores directly with the previous 50,000-database/10,000-query results.

## PCA + ITQ baseline and costs

```powershell
.venv/Scripts/python benchmark_transfer.py --unseen-database-from results/kmnist-frozen --itq-and-costs --output-dir results/kmnist-itq-costs-next
```

Existing encoders and thresholds remain frozen. PCA + ITQ adds a rotation fitted only to the original 50,000 training images projected by the saved PCA. It uses the same 50-iteration, training-mean-centered, unwhitened procedure as the earlier Fashion-MNIST diagnostic, with the run's seed and zero thresholds after rotation. There is no KMNIST weight search, iteration search, or query/database calibration. All four methods use the identical 5,000 unseen database IDs, 5,000 disjoint query IDs, and matched training-database control from the previous section.

The run took 89 seconds, including 15 ITQ fits, 120 retrieval evaluations, and encoding measurements. Unseen-database precision@10 (%), mean +/- sample standard deviation across five seeds:

| Bits | PCA | PCA + ITQ | Plain AE | Quantization AE |
|---|---:|---:|---:|---:|
| 16 | 48.72 +/- 0.34 | 56.18 +/- 1.02 | 50.00 +/- 1.17 | 52.99 +/- 0.38 |
| 32 | 57.62 +/- 0.18 | 66.78 +/- 0.57 | 56.78 +/- 0.91 | 58.14 +/- 0.39 |
| 64 | 60.52 +/- 0.22 | 73.77 +/- 0.42 | 64.93 +/- 0.46 | 65.03 +/- 0.45 |

PCA + ITQ beat both PCA and the quantization AE in all five seeds at each bit length. Mean gains over the quantization AE were 3.19, 8.64, and 8.74 points. The matched training-database control also favored PCA + ITQ: its means were 52.65, 60.83, and 65.98%, versus 49.35, 52.53, and 57.64% for the quantization AE. All three existing methods reproduced their previous retrieval scores exactly.

### Encoding and storage

All methods are timed on CPU with PyTorch and BLAS/OpenMP limited to one thread. Each timed run encodes all 5,000 query images already in RAM and includes thresholding and bit packing, but excludes model loading, fitting, data I/O, and search. Two warm-up runs precede seven timed repeats; packed outputs must match across repeats. Entries below average each seed's median elapsed time. These are warm-batch implementation measurements on this Ryzen 9 9950X, not per-request latency or GPU serving comparisons. Torch thread settings are restored afterward.

| Bits | PCA ms / 5,000 | PCA + ITQ ms / 5,000 | Plain AE ms / 5,000 | Quantization AE ms / 5,000 |
|---|---:|---:|---:|---:|
| 16 | 0.88 | 0.97 | 7.70 | 7.80 |
| 32 | 1.31 | 2.00 | 8.10 | 7.88 |
| 64 | 2.58 | 3.69 | 8.45 | 8.40 |

Inference parameter storage includes the encoder or PCA projection, required centering, thresholds, and ITQ rotation. It excludes decoder weights, optimizer state, labels, index metadata, and working memory. PCA projections and neural parameters are float32; ITQ's centering/rotation use float64. This counts the implemented parameter representation, not an optimized/fused projection or checkpoint-file size. Both AEs have identical inference architecture and size.

| Bits | PCA KiB | PCA + ITQ KiB | Either AE KiB | Packed 5,000-item database bytes |
|---|---:|---:|---:|---:|
| 16 | 52.13 | 54.31 | 400.63 | 10,000 |
| 32 | 101.19 | 109.56 | 408.75 | 20,000 |
| 64 | 199.31 | 232.06 | 425.00 | 40,000 |

### Fitting cost provenance

PCA/AE fitting times below are reused from `results/kmnist-frozen/`, not rerun here. They include fitting, checkpoint evaluation for AEs, training-set encoding, and threshold calibration; PCA ran on CPU and AE training on CUDA. The new ITQ column measures its additional CPU training projection and rotation fitting, excluding artifact saving. These are mean wall times across seeds from different runs, not a controlled end-to-end training benchmark. No tuning-search cost is included.

| Bits | Recorded PCA fit s | Newly measured additional ITQ s | Recorded plain AE fit s | Recorded quantization AE fit s |
|---|---:|---:|---:|---:|
| 16 | 0.24 | 0.29 | 7.21 | 7.90 |
| 32 | 0.28 | 0.61 | 7.22 | 8.39 |
| 64 | 0.50 | 1.39 | 7.19 | 8.09 |

Reports and the 15 new ITQ transforms are in `results/kmnist-itq-costs/`. After `load_artifact`, ITQ inference uses `saved_codes(transform, arrays, images, itq=True)` to apply the saved centering and rotation. The protocol records timing settings and source-report hashes; completion checks all 75 original model/transform files remained unchanged. ITQ artifacts passed reload comparisons against their training codes, and all 15 quantization objectives were nonincreasing within numerical tolerance. This remains a follow-up on the previously inspected KMNIST test pool, not another untouched benchmark. ITQ underperformed on the earlier Fashion-MNIST diagnostic, so this result should not be generalized to every dataset.

## Contrastive objective with fresh MNIST evaluation

```powershell
# Smoke stays on KMNIST validation; it never loads MNIST.
.venv/Scripts/python contrastive_experiment.py --smoke --output-dir results/contrastive-smoke-next
.venv/Scripts/python contrastive_experiment.py --stage tune --output-dir results/contrastive-next
.venv/Scripts/python contrastive_experiment.py --stage evaluate --output-dir results/contrastive-next
```

The same `784 -> 128 -> bits` encoder is trained directly with symmetric NT-Xent contrastive loss, with no extra projection head or reconstruction loss. Two independent views of each image are positives; other minibatch images are negatives, including possible same-class images. The loss uses L2-normalized embeddings, but inference hashes raw encoder outputs at training medians. The weak quantization term is `0.001 * mean((abs(z) - 1)^2)`.

Views use bilinear affine sampling with rotations within 10 degrees, scale factors 0.9 to 1.1, sampling-grid translations within two pixels per axis, and zero padding. No flips are used. Adam remains at learning rate 0.001, with 20 epochs and 256 source images per batch (512 augmented views). The contrastive decoder is neither trained nor used; its initialized state is retained only for compatibility with the existing artifact format and excluded from inference parameter counts.

The matched **augmented reconstruction** control uses the same encoder architecture, initialization, image ordering, two-view augmentations, and quantization weight. It reconstructs the augmented inputs through the existing decoder. Each neural model selects its checkpoint by its own base objective on fixed seeded validation views, excluding the quantization term. This controls augmentation and encoder capacity, though the losses, decoder training, and checkpoint criteria necessarily differ.

Temperature candidates 0.1, 0.2, and 0.5 are selected separately per bit length using median-code precision@10 on 2,000 KMNIST validation queries against the 50,000 training images at seed 42; ties favor lower temperatures. No labels enter gradient training, but validation labels select temperature. The selected temperatures were **0.5, 0.5, and 0.2** at 16/32/64 bits. These were frozen before any MNIST evaluation. No model weights transfer between datasets: all evaluated models are fitted from scratch on MNIST. The PCA + ITQ baseline remains the fixed 50-iteration unwhitened procedure, and no MNIST retrieval scores change any settings.

MNIST was reserved as the new benchmark because the KMNIST test pool had already been inspected. For each of the same five seeds, models fit 50,000 official training images, use the remaining 10,000 for unsupervised checkpoint selection, then hash 5,000 official test images as an unseen database and the other 5,000 as disjoint queries. The same 5,000-item training-database control is also recorded. Test labels are used only for evaluation. The smoke run uses a reduced KMNIST validation split instead.

Tuning took 2 minutes 19 seconds (nine candidates); frozen evaluation took 7 minutes 55 seconds (60 model/transform artifacts and 120 retrieval evaluations). **MNIST unseen-database precision@10 (%)**, mean +/- sample standard deviation across five seeds:

| Bits | PCA | PCA + ITQ | Matched augmented reconstruction | Contrastive |
|---|---:|---:|---:|---:|
| 16 | 65.31 +/- 0.17 | 74.73 +/- 0.85 | 69.79 +/- 0.91 | 76.24 +/- 1.28 |
| 32 | 72.22 +/- 0.27 | 82.39 +/- 0.52 | 75.07 +/- 0.75 | 81.12 +/- 0.79 |
| 64 | 72.74 +/- 0.25 | 85.70 +/- 0.46 | 81.11 +/- 0.42 | 83.83 +/- 0.76 |

Contrastive gains over the matched control were 6.45, 6.05, and 2.72 points, with wins in all five seeds at each bit length. Against PCA + ITQ, gains were +1.51, -1.27, and -1.86 points, with wins in 4/5, 0/5, and 0/5 seeds. The matched training-database control gave the same mean ranking: contrastive scored 75.51, 80.67, and 82.94%, versus PCA + ITQ's 74.20, 81.72, and 85.08%. Do not compare these absolute scores with KMNIST or compare the augmented reconstruction control with an unaugmented AE as though they were the same experiment.

### Cost of the new objective

CPU encoding follows the same single-thread, two-warm-up/seven-repeat protocol above. Times are means of per-seed median elapsed times for encoding and packing all 5,000 queries. Inference size remains 400.63/408.75/425.00 KiB for either neural encoder versus 54.31/109.56/232.06 KiB for PCA + ITQ.

| Bits | PCA + ITQ encode ms | Matched reconstruction encode ms | Contrastive encode ms | PCA + ITQ fit s | Matched reconstruction fit s | Contrastive fit s |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 1.01 | 7.80 | 7.79 | 0.56 | 12.00 | 13.08 |
| 32 | 1.94 | 8.02 | 7.97 | 0.90 | 11.45 | 12.94 |
| 64 | 4.06 | 8.37 | 8.57 | 1.84 | 11.58 | 12.80 |

Unlike the previous cost table, fitting times here are from the new MNIST run. Neural fits include training, validation, training encoding, and median calibration on CUDA. PCA + ITQ sums CPU PCA fit/calibration and the subsequent training projection/rotation fit. They exclude saving and search and are implementation wall times, not a hardware-neutral algorithmic comparison. Temperature-search cost is separate.

Reports and all nine tuning checkpoints plus 60 evaluation artifacts are in `results/contrastive/`. `protocol.json` reserves the complete setup before tuning; `tuning.json` records every candidate and fixed temperatures. Each evaluation report includes its tuning-report SHA-256, official split IDs, and class counts. Saved neural models passed validation-encoding/code reload checks, and PCA/ITQ transforms passed training-code reload checks. The full experiment did not change its settings after MNIST results. This is evidence for this shallow contrastive recipe, not a conclusion about all contrastive methods; repeated seeds share the official test pool and are not independent datasets or confidence intervals.

## Contrastive ITQ: validation-only follow-up

```powershell
.venv/Scripts/python contrastive_experiment.py --diagnose-from results/contrastive --reference-itq-from results/kmnist-itq-costs --output-dir results/contrastive-itq-validation-next
```

This diagnostic reuses the selected seed-42 KMNIST contrastive encoders at temperatures 0.5/0.5/0.2, without retraining or changing the encoder weights. It fits one 50-iteration ITQ rotation per bit length on all 50,000 training embeddings, using training-mean centering and no whitening. Evaluation uses the same 2,000 validation queries previously used for temperature selection, against the 50,000 training database items. No MNIST or official test queries are loaded. The optional PCA + ITQ reference reuses its saved seed-42 KMNIST projection/rotation and is scored against the identical database and queries.

These are **single-seed development precision@10 (%)**, not five-seed means or fresh test estimates:

| Representation | 16 dimensions/bits | 32 dimensions/bits | 64 dimensions/bits |
|---|---:|---:|---:|
| Contrastive continuous Euclidean | 82.445 | 83.240 | 86.890 |
| Contrastive continuous cosine | 83.075 | 83.735 | 87.855 |
| Contrastive centered continuous cosine | 83.200 | 83.815 | 87.710 |
| Contrastive median binary | 67.915 | 76.575 | 82.355 |
| Contrastive centered sign | 69.490 | 76.355 | 82.475 |
| Contrastive + ITQ binary | 73.490 | 78.290 | 84.310 |
| PCA + ITQ reference | 72.295 | 82.435 | 88.710 |

ITQ improved contrastive median codes by 5.575, 1.715, and 1.955 points. It also improved over the centered-sign control at every width, so the gain was not solely a change from median to mean thresholds. It exceeded PCA + ITQ by 1.195 points at 16 bits but remained 4.145 and 4.400 points behind at 32/64 bits. The original median-code scores reproduced the tuning report exactly. Continuous retrieval still exceeded contrastive binary retrieval, but uses float storage and a different distance metric, so it is not a same-budget comparison or a strict upper bound.

The run took 36 seconds. All source files remained hash-identical; all three output encoder checkpoints are byte-identical copies of their originals. New artifacts retain the saved encoder and add only centering, rotation, and zero thresholds; inference parameter sizes are 402.81, 417.13, and 457.75 KiB. Reloaded ITQ artifacts matched training and validation query codes. Encoding speed was not remeasured in this diagnostic.

Reports are in `results/contrastive-itq-validation/`, with source hashes, dataset row IDs, objective histories, and all seven retrieval variants. The two JSON reports are tracked; model files remain ignored. After `load_artifact`, use `saved_codes(transform, arrays, images, itq=True)` for rotated codes. `fit_itq_artifact` now supports both PCA and saved neural encoders. On a fresh clone, regenerate the selected KMNIST checkpoints and PCA ITQ artifacts before running the diagnostic.

## Next decision

Replicate contrastive + ITQ across additional development seeds before adopting it. The single-seed gain is promising, especially at 16 bits, but it does not establish a fresh-test improvement or close the 32/64-bit gap to PCA + ITQ. Keep the encoder frozen during that comparison and do not tune against the inspected MNIST test results.
