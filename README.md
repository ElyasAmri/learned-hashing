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

## First full run

Seed 42, default splits and 20 epochs, RTX 5080 / Ryzen 9 9950X. Total wall time was 2 minutes 26 seconds. Environment: torch 2.11.0+cu128, torchvision 0.26.0+cu128, numpy 2.2.6, scikit-learn 1.7.2. Detailed local report: `results/pilot-gpu.json`.

| Bits | Random precision@10 | PCA precision@10 | Autoencoder precision@10 | PCA unique codes | Autoencoder unique codes |
|---|---:|---:|---:|---:|---:|
| 16 | 49.874% | 65.592% | 61.549% | 11,695 | 4,937 |
| 32 | 62.209% | 73.032% | 69.888% | 46,393 | 29,406 |
| 64 | 68.643% | 76.652% | 75.351% | 49,974 | 47,458 |

PCA led retrieval quality at every bit budget in this single-seed run. The autoencoder beat random projections but used fewer distinct codes than PCA. This does not establish statistical significance or show that collisions caused the quality gap. Models were not saved; the report retains metrics and validation histories.

## Next decision

Compare quality at each bit budget before adding losses or architecture complexity. If PCA matches or beats the autoencoder, nonlinear reconstruction has not justified itself. If codes are redundant, investigate balance/decorrelation; if binarization destroys useful neighborhoods, investigate quantization-aware training. Repeat across seeds before drawing conclusions. Then add an unseen retrieval database and larger benchmarks with established learned-hashing methods. Fashion-MNIST alone does not support a state-of-the-art claim.
