# Experiment summary

## Bottom line

Reconstruction alone was not competitive enough. Weak quantization regularization improved it, but PCA + ITQ proved substantially stronger on KMNIST. Contrastive training improved over matched reconstruction and beat PCA + ITQ at 16 bits on fresh MNIST, but not at 32 or 64 bits. Neural encoding remained more expensive.

This file covers all completed full experiments and every tuning setting. Smoke tests and interrupted runs are not quality benchmarks. [README.md](README.md) has reproduction commands, sample standard deviations, and detailed protocol notes. Compact original JSON reports are published under [results/](results/); datasets and trained models are not included.

## Reading the results

- Metric: same-class precision@10 (%), higher is better. This is a category-similarity proxy, not exact-key hashing or nearest-neighbor recall.
- Main tables show means across seeds 42, 7, 21, 84, and 123 unless marked as single-seed diagnostics.
- Tuning tables show single-seed validation results, not five-seed evaluations.
- Do not compare absolute scores across datasets, database sizes, or validation/test protocols.
- Repeated seeds share datasets; their variation is not a confidence interval over independent benchmarks.

## Shared configuration

| Setting | Configuration |
|---|---|
| Inputs | Flattened 28x28 grayscale images, float32 in `[0, 1]` |
| Encoder | `784 -> 128 -> 16/32/64`, ReLU hidden layer, linear latent output |
| Reconstruction decoder | `bits -> 128 -> 784`, sigmoid output |
| Neural training | Adam, learning rate 0.001, 20 epochs, batch size 256 |
| Training and validation | 50,000 training images and 10,000 validation images |
| Binary retrieval | Exact Hamming search; deterministic, label-independent tie breaking |
| Standard thresholds | Per-coordinate training medians unless specified otherwise |
| Labels | Never used in gradient training; used for retrieval evaluation and hyperparameter selection |

Reconstruction models selected checkpoints by validation reconstruction MSE. Contrastive and matched-augmentation controls selected checkpoints by their respective base objectives on fixed seeded validation views. PCA/ITQ fit without labels. Neural training used an RTX 5080; PCA, ITQ, and Hamming search used CPU on a Ryzen 9 9950X.

## 1. Initial baselines and separation penalties

**Fashion-MNIST: 50,000 training/database images, 10,000 official test queries.**

- Random hyperplanes: centered inputs, seeded Gaussian projection, zero thresholds.
- PCA: randomized PCA with training-median thresholds.
- Plain autoencoder (AE): reconstruction MSE only.
- Inverse KL: reciprocal symmetric KL between softmax latent distributions.
- Inverse Wasserstein: reciprocal Wasserstein-1 over ordered softmax-coordinate bins.

Both separation variants used `MSE + weight * mean(1 / (0.001 + distance))`, softmax temperature 1, and randomly paired distinct samples without self-pairs. They repelled arbitrary pairs, including same-class images.

| Configuration | 16 bits | 32 bits | 64 bits |
|---|---:|---:|---:|
| Random hyperplanes | 51.24 | 62.95 | 70.47 |
| PCA | **65.63** | **72.99** | **76.78** |
| Plain AE | 61.86 | 70.17 | 75.20 |
| AE + inverse KL | 62.62 | 70.03 | 75.36 |
| AE + inverse Wasserstein | 62.76 | 70.24 | 75.48 |

| Selected penalty weight | 16 bits | 32 bits | 64 bits |
|---|---:|---:|---:|
| Inverse KL | 0.0001 | 0.0001 | 0.000001 |
| Inverse Wasserstein | 0.00001 | 0.00001 | 0.00001 |

**Finding:** PCA won consistently. Separation penalties produced small, inconsistent improvements and did not close the gap. Softmax does not bound KL divergence; these penalties do not directly enforce semantic similarity or useful binary codes.

Sources: `results/pilot-gpu.json`, `results/pilot-gpu-seed-*.json`, and [results/penalties/](results/penalties/).

## 2. Continuous retrieval, thresholding, and ITQ

**Fashion-MNIST: 50,000 training/database images, 10,000 validation queries.**

We compared continuous Euclidean retrieval, continuous cosine retrieval centered at the training latent mean, median binary codes, centered sign codes, and ITQ binary codes. Centered sign means training-mean centering followed by zero thresholds.

ITQ used 50 alternating sign/orthogonal-Procrustes iterations on centered training latents, initialized with a seeded random orthogonal rotation. No whitening or per-coordinate scaling was used. Continuous rows use 16/32/64 floating-point dimensions, not equivalent binary storage budgets.

| Representation | 16 dimensions/bits | 32 dimensions/bits | 64 dimensions/bits |
|---|---:|---:|---:|
| PCA continuous Euclidean | 78.49 | 80.43 | 81.19 |
| PCA continuous cosine | 79.09 | 81.02 | 81.86 |
| PCA median binary | 66.38 | 73.54 | 77.49 |
| PCA centered sign | 66.54 | 73.62 | 77.48 |
| PCA + ITQ | 64.80 | 71.73 | 76.29 |
| AE continuous Euclidean | 79.66 | 80.62 | 81.08 |
| AE continuous cosine | 79.93 | 81.17 | 81.61 |
| AE median binary | 62.42 | 70.87 | 75.90 |
| AE centered sign | 61.87 | 70.57 | 75.60 |
| AE + ITQ | 62.67 | 71.76 | 76.43 |

**Findings:** AE continuous retrieval was competitive with PCA, but its binary retrieval gap was larger. ITQ modestly improved AE and hurt PCA on this dataset. Reducing the quantization objective did not automatically improve category retrieval. Continuous-to-binary gaps change both representation and retrieval metric; they are not isolated information-loss measurements.

Source: [results/quantization-v2/](results/quantization-v2/).

## 3. Quantization-aware reconstruction training

**Fashion-MNIST: 50,000 training/database images, 10,000 validation queries.**

We added `weight * mean((abs(z) - 1)^2)` to reconstruction MSE. This encourages raw latent coordinates toward -1 or +1, without a hard binary bottleneck, straight-through estimator, balance loss, or decorrelation loss.

First sweep: weights 0.001, 0.01, 0.1, and 1.0. Selection used zero-threshold retrieval; median thresholds were a separate diagnostic. Weight 0.001 won at every bit length.

Second sweep: median thresholds fixed before selection; weights 0.00001, 0.00003, 0.0001, 0.0003, and 0.001. Selected weights were 0.001, 0.0003, and 0.0003 for 16/32/64 bits.

| Configuration | 16 bits | 32 bits | 64 bits |
|---|---:|---:|---:|
| PCA reference | 66.38 | 73.54 | 77.49 |
| Plain AE reference | 62.42 | 70.87 | 75.90 |
| AE + ITQ reference | 62.67 | 71.76 | 76.43 |
| Quantization AE, 0.001, zero thresholds | 64.95 | 69.14 | 73.39 |
| Quantization AE, 0.001, median thresholds | 67.06 | 73.18 | 77.27 |
| Quantization AE, smaller-weight sweep, median thresholds | **67.06** | **73.63** | **77.73** |

**Findings:** weak regularization with median thresholds substantially improved AE. Smaller weights narrowly exceeded PCA's mean, but these were reused validation data. Zero thresholds were worse at 32/64 bits. At weights 0.1 and 1.0, every database item collapsed to one zero-threshold code during tuning. Median thresholds could still recover distinctions from residual latent variation; the collapse statement applies specifically to zero-threshold codes.

Sources: [results/qat/](results/qat/) and [results/qat-median/](results/qat-median/).

## 4. Frozen configuration transferred to KMNIST

**KMNIST: 50,000 training/database images, 10,000 official test queries.**

Models were trained from scratch on KMNIST, transferring the Fashion-MNIST-selected hyperparameters, not learned weights. Quantization weights stayed at 0.001/0.0003/0.0003 with median thresholds. No KMNIST weight search was performed.

| Configuration | 16 bits | 32 bits | 64 bits |
|---|---:|---:|---:|
| PCA | 53.74 | **65.41** | 71.81 |
| Plain AE | 53.94 | 63.11 | 72.24 |
| Frozen quantization AE | **57.06** | 65.13 | **73.44** |

**Findings:** quantization AE beat plain AE in every seed at every bit length. It consistently beat PCA at 16/64 bits, but PCA retained a small mean advantage at 32 bits.

Source: [results/kmnist-frozen/](results/kmnist-frozen/).

## 5. Unseen database and stronger PCA + ITQ baseline

**KMNIST: 5,000 official test images as database, the other 5,000 as disjoint queries.**

Existing encoders and thresholds were reused without retraining. A matched 5,000-image training-database control used identical queries. PCA + ITQ was added afterward, fitting its rotation only on the original training images using the fixed 50-iteration procedure.

### Unseen database

| Configuration | 16 bits | 32 bits | 64 bits |
|---|---:|---:|---:|
| PCA | 48.72 | 57.62 | 60.52 |
| Plain AE | 50.00 | 56.78 | 64.93 |
| Quantization AE | 52.99 | 58.14 | 65.03 |
| PCA + ITQ | **56.18** | **66.78** | **73.77** |

### Matched training database

| Configuration | 16 bits | 32 bits | 64 bits |
|---|---:|---:|---:|
| PCA | 46.11 | 51.60 | 52.51 |
| Plain AE | 46.70 | 51.28 | 56.98 |
| Quantization AE | 49.35 | 52.53 | 57.64 |
| PCA + ITQ | **52.65** | **60.83** | **65.98** |

**Findings:** quantization AE's advantage over plain PCA persisted on unseen database items. However, PCA + ITQ beat quantization AE in every seed at every bit length. Unlike Fashion-MNIST, ITQ was very effective here.

All methods scored higher with the unseen database than with the training control. Equal sizes and shared queries do not eliminate differences between the official train/test distributions, class composition, or sampling. The gap cannot be attributed solely to training exposure. This was a follow-up on an already inspected test pool, not another fresh benchmark.

Sources: [results/kmnist-unseen/](results/kmnist-unseen/) and [results/kmnist-itq-costs/](results/kmnist-itq-costs/).

## 6. Contrastive learning with fresh MNIST evaluation

The contrastive encoder used symmetric NT-Xent loss on L2-normalized embeddings from two augmented views, no projection head, a fixed quantization weight of 0.001, and training-median thresholds on raw encoder outputs. Views used small affine rotations, translations, and scaling, without flips. Batches contained 256 source images and 512 views.

The matched reconstruction control used the same encoder, initialization, image order, augmentations, and quantization weight, but reconstructed the augmented inputs. Its loss, decoder training, and checkpoint criterion necessarily differ from the contrastive model.

Temperatures 0.1, 0.2, and 0.5 were selected on KMNIST validation. Frozen temperatures for 16/32/64 bits were 0.5/0.5/0.2. All evaluation models were then fitted from scratch on MNIST, which had not previously been evaluated in this experiment series. MNIST retrieval results did not change the configuration.

**MNIST: 5,000 unseen database images and 5,000 disjoint queries.**

### Unseen database

| Configuration | 16 bits | 32 bits | 64 bits |
|---|---:|---:|---:|
| PCA | 65.31 | 72.22 | 72.74 |
| PCA + ITQ | 74.73 | **82.39** | **85.70** |
| Matched augmented reconstruction | 69.79 | 75.07 | 81.11 |
| Contrastive | **76.24** | 81.12 | 83.83 |

### Matched training database

| Configuration | 16 bits | 32 bits | 64 bits |
|---|---:|---:|---:|
| PCA | 64.64 | 71.30 | 71.33 |
| PCA + ITQ | 74.20 | **81.72** | **85.08** |
| Matched augmented reconstruction | 69.16 | 74.14 | 80.03 |
| Contrastive | **75.51** | 80.67 | 82.94 |

**Findings:** contrastive training beat matched reconstruction in every seed at every bit length. It beat PCA + ITQ at 16 bits on average, winning 4/5 seeds. PCA + ITQ won every seed at 32/64 bits. This supports changing the objective, not a universal advantage for neural hashing.

Source: [results/contrastive/](results/contrastive/).

## 6b. Contrastive + ITQ validation follow-up

**Single seed 42, KMNIST validation only: 50,000 training/database images and 2,000 validation queries.** This is not a five-seed or fresh-test result. The previously selected encoders and temperatures stayed fixed; ITQ fitted 50 iterations on their training embeddings only. Source encoder files remained unchanged. No MNIST or official test queries were loaded.

| Representation | 16 dimensions/bits | 32 dimensions/bits | 64 dimensions/bits |
|---|---:|---:|---:|
| Contrastive continuous Euclidean | 82.445 | 83.240 | 86.890 |
| Contrastive continuous cosine | 83.075 | 83.735 | 87.855 |
| Contrastive centered continuous cosine | 83.200 | 83.815 | 87.710 |
| Contrastive median binary | 67.915 | 76.575 | 82.355 |
| Contrastive centered sign | 69.490 | 76.355 | 82.475 |
| Contrastive + ITQ binary | 73.490 | 78.290 | 84.310 |
| PCA + ITQ on the identical queries | 72.295 | 82.435 | 88.710 |

ITQ gained 5.575, 1.715, and 1.955 points over contrastive median codes, and also beat the centered-sign control at every width. The rotated contrastive codes beat PCA + ITQ at 16 bits but trailed it at 32/64 bits. The encoder was not retrained; the new rotation increases inference parameters to 402.81/417.13/457.75 KiB. Speed was not remeasured. The same validation queries previously selected temperatures, so this promising 36-second diagnostic needs replication rather than a generalization claim.

Source: [results/contrastive-itq-validation/](results/contrastive-itq-validation/).

## 6c. Contrastive + ITQ five-seed development replication

The four additional seeds (7, 21, 84, 123) each trained new KMNIST encoders with frozen temperatures 0.5/0.5/0.2, quantization weight 0.001, and the same augmentations, optimizer, 20 epochs, and batch size. Each encoder stayed fixed during ITQ fitting and the paired comparison. No hyperparameter search or MNIST/test evaluation was performed. Each seed used its own 50,000 training/database images and 2,000 disjoint validation queries; seed 42 reuses the preceding diagnostic.

Five-seed mean validation precision@10 (%):

| Representation | 16 dimensions/bits | 32 dimensions/bits | 64 dimensions/bits |
|---|---:|---:|---:|
| Contrastive continuous Euclidean | 82.92 | 83.05 | 87.24 |
| Contrastive continuous cosine | 83.42 | 83.50 | 87.99 |
| Contrastive centered continuous cosine | 83.47 | 83.55 | 87.95 |
| Contrastive median binary | 69.34 | 76.29 | 82.37 |
| Contrastive centered sign | 69.46 | 76.24 | 82.50 |
| Contrastive + ITQ binary | 72.93 | 77.66 | 84.72 |
| PCA + ITQ reference | 72.75 | 83.16 | 88.91 |

ITQ improved contrastive median codes by 3.58, 1.36, and 2.35 points, winning all five seeds at every width. It also beat the centered-sign control in every comparison. The mean difference from PCA + ITQ was +0.18, -5.50, and -4.19 points, with wins in 4/5, 0/5, and 0/5 seeds. The 16-bit result is a near tie, not a robust advantage; paired-gap standard deviation was 1.16 points. These remain reused development splits, not fresh test estimates or independent datasets.

The four replications took 5 minutes 48 seconds combined, fitting 12 encoders before freezing them for comparison. Reports are in `results/contrastive-itq-seed-7/`, `results/contrastive-itq-seed-21/`, `results/contrastive-itq-seed-84/`, and `results/contrastive-itq-seed-123/`, alongside the seed-42 report in `results/contrastive-itq-validation/`. Model copies remained byte-identical through rotation. Encoding cost was not remeasured.

## 6d. Training-only projection head: negative result

This five-seed KMNIST validation ablation added a `bits -> 128 -> 128` ReLU head for the contrastive loss and discarded it before hashing. The encoder, augmentations, original temperatures 0.5/0.5/0.2, weak raw-latent quantization penalty, training duration, and ITQ settings stayed fixed. All 15 variant encoders were trained from scratch, then frozen for ITQ. No head-size search, temperature retuning, or MNIST/test evaluation was performed.

Five-seed mean validation precision@10 (%) on the same 50,000-item training database and 2,000 validation queries per seed:

| Representation | 16 dimensions/bits | 32 dimensions/bits | 64 dimensions/bits |
|---|---:|---:|---:|
| Projection-trained encoder, continuous Euclidean | 78.36 | 80.64 | 83.53 |
| Projection-trained encoder, continuous cosine | 79.17 | 82.12 | 84.96 |
| Projection-trained encoder, centered cosine | 78.49 | 80.95 | 83.76 |
| Projection-trained encoder, median binary | 59.69 | 69.27 | 76.66 |
| Projection-trained encoder, centered sign | 59.73 | 68.63 | 76.18 |
| Projection-trained encoder + ITQ | 61.43 | 70.39 | 76.77 |
| Headless contrastive + ITQ control | 72.93 | 77.66 | 84.72 |
| PCA + ITQ reference | 72.75 | 83.16 | 88.91 |

The head variant lost to the headless ITQ control in all 15 comparisons, by 11.50, 7.27, and 7.95 points on average. Its raw continuous retrieval also worsened. The extra head added 73/81/97 KiB of training-only parameters but no inference parameters; it is saved separately and absent from inference encoder checkpoints. This is a negative result for the fixed recipe, not a general conclusion about projection heads. The headless configuration remains the default.

The runs took 8 minutes 53 seconds combined. Reports are in `results/contrastive-projection-seed-{42,7,21,84,123}/`. Every variant used identical split IDs and PCA-reference scores to the preceding comparison, and encoder bytes remained unchanged during ITQ. Encoding speed was not remeasured.

## 7. Every tuning configuration

All tables in this section use seed 42, 2,000 validation queries, and a 50,000-image training database. These are selection scores, not the five-seed evaluation results above. Bold scores identify the selected configuration per bit length and penalty family.

### A. Separation penalties on Fashion-MNIST

| Penalty | Weight | 16 bits | 32 bits | 64 bits |
|---|---:|---:|---:|---:|
| Inverse KL | 0.000001 | 64.485 | 69.180 | **76.290** |
| Inverse KL | 0.00001 | 63.310 | 71.100 | 75.520 |
| Inverse KL | 0.0001 | **65.080** | **71.255** | 75.500 |
| Inverse KL | 0.001 | 64.605 | 70.255 | 75.305 |
| Inverse Wasserstein | 0.000001 | 63.665 | 69.350 | 75.480 |
| Inverse Wasserstein | 0.00001 | **65.105** | **71.395** | **76.540** |
| Inverse Wasserstein | 0.0001 | 64.110 | 71.045 | 76.365 |
| Inverse Wasserstein | 0.001 | 63.680 | 71.270 | 75.715 |

### B. Initial quantization sweep on Fashion-MNIST

Each cell is **zero-threshold score / median-threshold score**. Selection used zero-threshold retrieval only.

| Weight | 16 bits | 32 bits | 64 bits |
|---|---:|---:|---:|
| 0.001 | **64.655** / 66.780 | **70.900** / 74.005 | **73.480** / 77.110 |
| 0.01 | 49.645 / 62.950 | 60.895 / 71.220 | 66.615 / 74.930 |
| 0.1 | 9.865 / 67.030 | 9.865 / 72.125 | 9.865 / 77.075 |
| 1.0 | 9.865 / 60.775 | 9.865 / 66.535 | 9.865 / 71.420 |

### C. Smaller weights with fixed median thresholds on Fashion-MNIST

| Weight | 16 bits | 32 bits | 64 bits |
|---|---:|---:|---:|
| 0.00001 | 64.100 | 72.215 | 77.060 |
| 0.00003 | 65.445 | 71.730 | 76.825 |
| 0.0001 | 65.200 | 73.520 | 77.855 |
| 0.0003 | 66.680 | **74.190** | **78.180** |
| 0.001 | **66.780** | 74.005 | 77.110 |

### D. Contrastive temperatures on KMNIST

Quantization weight remained 0.001 and thresholds remained training medians.

| Temperature | 16 bits | 32 bits | 64 bits |
|---|---:|---:|---:|
| 0.1 | 53.805 | 70.755 | 80.690 |
| 0.2 | 64.730 | 75.835 | **82.355** |
| 0.5 | **67.915** | **76.575** | 79.510 |

## 8. Computational cost

Encoding measurements use one CPU thread, two warm-up runs, and seven measured repeats. They include encoding, thresholding, and packing 5,000 images already in RAM, but exclude loading and search. Entries average each seed's median elapsed time; they are not single-query latency or GPU serving measurements.

| Benchmark and model | CPU milliseconds per 5,000 images: 16 / 32 / 64 bits |
|---|---|
| KMNIST PCA | 0.88 / 1.31 / 2.58 |
| KMNIST PCA + ITQ | 0.97 / 2.00 / 3.69 |
| KMNIST plain AE | 7.70 / 8.10 / 8.45 |
| KMNIST quantization AE | 7.80 / 7.88 / 8.40 |
| MNIST PCA | 0.90 / 1.33 / 2.73 |
| MNIST PCA + ITQ | 1.01 / 1.94 / 4.06 |
| MNIST matched reconstruction | 7.80 / 8.02 / 8.37 |
| MNIST contrastive | 7.79 / 7.97 / 8.57 |

| Inference parameter storage | 16 bits | 32 bits | 64 bits |
|---|---:|---:|---:|
| PCA | 52.13 KiB | 101.19 KiB | 199.31 KiB |
| PCA + ITQ | 54.31 KiB | 109.56 KiB | 232.06 KiB |
| Neural encoder | 400.63 KiB | 408.75 KiB | 425.00 KiB |

These exclude decoder weights and runtime working memory. PCA/encoder parameters are float32; ITQ centering/rotation are float64. The packed 5,000-item database takes 10,000/20,000/40,000 bytes at 16/32/64 bits for every method.

For the latest MNIST run:

| Fitting wall time | 16 bits | 32 bits | 64 bits |
|---|---:|---:|---:|
| PCA + ITQ | 0.56 s | 0.90 s | 1.84 s |
| Matched reconstruction | 12.00 s | 11.45 s | 11.58 s |
| Contrastive | 13.08 s | 12.94 s | 12.80 s |

PCA/ITQ fit on CPU; neural training used CUDA. Fits include validation where applicable and training encoding/calibration, not saving, search, or temperature-search cost. These are implementation wall times, not hardware-neutral comparisons. Earlier KMNIST cost tables in README distinguish historical base fitting times from newly measured ITQ overhead.

## 9. Conclusions and limits

1. Reconstruction alone was not a strong enough objective for these binary retrieval experiments.
2. Arbitrary pair repulsion via inverse KL/Wasserstein added only small gains.
3. Thresholding mattered: weak quantization penalties worked much better with median thresholds than zero thresholds.
4. Useful continuous representations can lose substantial retrieval quality after binarization.
5. Beating plain PCA did not mean beating PCA + ITQ.
6. ITQ was dataset-dependent: it hurt PCA on Fashion-MNIST but helped strongly on KMNIST and MNIST.
7. Contrastive training is the most promising learned configuration tried here, especially at 16 bits.
8. PCA + ITQ remained the better accuracy/cost choice at 32/64 bits in the latest five-seed evaluation.
9. Adding ITQ to contrastive encoders improved all three widths in every development seed, but did not close the higher-bit gap to PCA + ITQ; the 16-bit aggregate was only a near tie.
10. A fixed training-only projection head substantially worsened raw-encoder retrieval, with and without ITQ, despite leaving inference size unchanged. It was not adopted.

Development queries were reused extensively and baseline tuning budgets were not identical. MNIST was held for the frozen contrastive comparison, but is now inspected too. No state-of-the-art or universal superiority claim is supported, and these test pools should not be used for more tuning.

## Operational record

- The first Fashion-MNIST ITQ run stopped on one PCA bit changing after reload because memory layout affected rounding near a threshold. Canonical contiguous components fixed it; the full rerun supplied the reported results.
- The first KMNIST download failed TLS validation. Official archives were downloaded through certificate-checked Windows HTTPS and matched to torchvision's digests; certificate checks were not disabled.
- Smoke tests checked execution paths, not model quality. Smoke and interrupted-run reports are excluded from the published result bundle.
- The latest experiment development test run had 43 passing unit tests, plus compilation and dependency checks.
- Public JSON reports retain configurations, per-seed metrics, tuning candidates, split IDs where recorded, validation histories, timing samples, and provenance hashes. Artifact paths refer to local checkpoints that must be regenerated; their presence in JSON does not mean checkpoints are published.
