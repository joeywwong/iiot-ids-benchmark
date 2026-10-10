# WUSTL-IIoT-2021: default subset experiment

Measured on **2026-10-11, Asia/Shanghai** (manifest timestamp
`2026-10-10T17:28:42.500631+00:00`). This is an offline binary flow-detection
comparison using authentic university data, not a production IDS validation.
The [aggregate record](wustl_iiot_2021_aggregates.json) preserves measured values,
config, calibration, feature list, split hashes, provenance and quality checks.

## Provenance and schema

The [official dataset page](https://www.cse.wustl.edu/~jain/iiot2/index.html) links
to [the university archive](https://www.cse.wustl.edu/~jain/iiot2/ftp/wustl_iiot_2021.zip).
It was downloaded from that source, without substituting a derivative dataset.

- Archive: `wustl_iiot_2021.zip`, 106,192,911 bytes.
- Archive SHA-256: `d80ef62029ff8a494061668c3b47f69ddce1fdd7784dae7ca9bd3f3c58d0763b`.
- Extracted CSV: `wustl_iiot_2021.csv`, 409,800,698 bytes.
- CSV SHA-256: `f897b24578cc6fdeb3e7a0e9ff63efd5bbdc926a545abbda725e0dbb348c6bca`.
- CSV columns: 49; excluding six author-identified metadata fields and the two
  labels leaves 41 candidate fields. The existing training-only audit additionally
  excluded `IdleTime` and `RunTime` through its conservative name heuristic,
  leaving 39 numeric predictors. These two are flow timing aggregates, not a
  verified leakage finding; this heuristic is an additional methodological limit.
- Original labels: `normal`, `DoS`, `Reconn`, `CommInj`, `Backdoor`.
  `Reconn` and `CommInj` denote reconnaissance and command injection.
- Required CSV columns match the original schema. The separate `Target` column
  agrees with the `Traffic` binary mapping on every row, with no missing values;
  both labels and all six prohibited fields were excluded from predictors.

Full-file counts, validated by scanning all rows:

| Original Traffic value | Rows |
| --- | ---: |
| normal | 1,107,448 |
| DoS | 78,305 |
| Reconn | 8,240 |
| CommInj | 259 |
| Backdoor | 212 |
| Total | 1,194,464 |

The normal and total attack counts (87,016) match the authors' reported totals.
This full-file scan does **not** mean every row was used to train the models.
Dataset attribution: M. Zolanvari, M. A. Teixeira, L. Gupta, K. M. Khan, and R. Jain,
“WUSTL-IIOT-2021 Dataset for IIoT Cybersecurity Research,” Washington University
in St. Louis, USA, October 2021.

## Protocol and reproduction

Default settings in `configs/default.yaml`: uniform, unbalanced 100,000-row sample;
seed 42; 100 trees per forest; one worker; 1% empirical benign-validation FPR
target. No model or threshold settings were changed to improve test scores.

After downloading/extracting the original CSV and installing `.[dev]`, run into
a fresh directory:

```powershell
python -m iiot_ids_benchmark.cli run --data data/raw/wustl_iiot_2021.csv --config configs/default.yaml --output results/wustl-default-reproduction/
```

A preceding smoke run used 10,000 sampled rows, 20 trees per forest, 200 validation
rows for importance and one repeat for importance/timing:

```powershell
python -m iiot_ids_benchmark.cli run --data data/raw/wustl_iiot_2021.csv --config configs/default.yaml --max-rows 10000 --rf-trees 20 --if-trees 20 --importance-rows 200 --importance-repeats 1 --timing-repeats 1 --output results/wustl-smoke-reproduction/
```

Only these two real model experiments were run. No all-rows model fit was performed;
the default subset was the planned primary comparison, not selected from multiple
runs for its score.

**Actual split: row-level, stratified by original Traffic.** All timestamps parse,
but 421 full-file rows (41 sampled rows) have `LastTime < StartTime`. The existing
strict interval check rejects chronological splitting and records its fallback.
Those rows were retained as observations, while timestamp predictors were excluded.
No verified session/capture grouping was available or invented. All categories
appear in every default-run partition:

| Traffic | Selected | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| normal | 92,743 | 55,646 | 18,548 | 18,549 |
| DoS | 6,555 | 3,933 | 1,311 | 1,311 |
| Reconn | 664 | 398 | 133 | 133 |
| CommInj | 20 | 12 | 4 | 4 |
| Backdoor | 18 | 11 | 4 | 3 |
| Total | 100,000 | 60,000 | 20,000 | 20,000 |

Attack prevalence: train 7.2567%, validation 7.2600%, test 7.2550%. Logistic
Regression and Random Forest fit the same 60,000 labeled training rows. Isolation
Forest fits its entire pipeline on only the 55,646 benign training rows. Labels
select those normal rows and inform the shared training-only feature audit;
this comparison uses different amounts of supervision.

Transforms fit only their training/fitting rows. Validation supplies only benign
scores for calibration; test data never select thresholds. An attack prediction
requires `score > threshold`, with ties benign and at most
`floor(0.01 * n_benign_validation)` allowed false positives.

| Model | Threshold | Validation FPR | Test FPR |
| --- | ---: | ---: | ---: |
| Logistic Regression | -1.2587090966068002 | 0.997412% | 0.938056% |
| Random Forest | 0.006252808988764045 | 0.975846% | 1.056661% |
| Isolation Forest | 0.7759596083157362 | 0.981238% | 0.954229% |

The Random Forest test FPR exceeds 1%; the target constrains validation, not test.

## Held-out measurements

Each model evaluated the identical 20,000 test observations. AP means scikit-learn
**average precision**, not trapezoidal integration of the PR curve. Values below
are rounded; the aggregate record retains full measured precision.

| Model | Precision | Attack recall | F1 | AP | ROC-AUC | Train (s) | Test batch (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.892460 | 0.995176 | 0.941023 | 0.993627 | 0.999538 | 11.369 | 0.592 |
| Random Forest | 0.880996 | 1.000000 | 0.936733 | 0.999999 | 1.000000 | 11.222 | 0.524 |
| Isolation Forest | 0.372340 | 0.072364 | 0.121177 | 0.632195 | 0.982015 | 3.930 | 0.770 |

Confusion counts use rows=true / columns=predicted, label order normal then attack:

| Model | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 18,375 | 174 | 7 | 1,444 |
| Random Forest | 18,353 | 196 | 0 | 1,451 |
| Isolation Forest | 18,372 | 177 | 1,346 | 105 |

Binary detection recall by original attack category (not attack-type classification):

| Attack | Test rows | Logistic Regression | Random Forest | Isolation Forest |
| --- | ---: | ---: | ---: | ---: |
| Backdoor | 3 | 1/3 (33.33%) | 3/3 (100%) | 0/3 (0%) |
| CommInj | 4 | 4/4 (100%) | 4/4 (100%) | 0/4 (0%) |
| DoS | 1,311 | 1,311/1,311 (100%) | 1,311/1,311 (100%) | 105/1,311 (8.01%) |
| Reconn | 133 | 128/133 (96.24%) | 133/133 (100%) | 0/133 (0%) |

Rare-category denominators are too small for robust comparative claims. In the
smoke run, Backdoor was absent from training and validation, and CommInj absent
from validation; its report flags those categories. Smoke outcomes were not
substituted for the primary experiment.

Timing uses `perf_counter`, one warmup and the median of three 20,000-row batches,
including preprocessing. Approximate throughput: Logistic Regression 33,782 flows/s,
Random Forest 38,172 flows/s, Isolation Forest 25,982 flows/s. Amortized batch cost:
0.029601 / 0.026197 / 0.038488 ms per flow respectively. These are offline batch
measurements on one machine, not live per-packet latency or a service guarantee.
Machine context: Windows, AMD64, AMD64 Family 25 Model 68 Stepping 1 processor.
Python 3.13.7; NumPy 2.5.3; pandas 2.3.3; scikit-learn 1.9.1; matplotlib 3.11.2;
PyYAML 6.0.3; benchmark package 0.1.0. Timing varies with system load.

## High-score and dependency review

The near-perfect supervised ranking warrants caution. The label columns, six
author-prohibited fields and extra heuristic exclusions are absent from the 39
predictors. Fit paths and automated tests verify that evaluation rows do not fit
preprocessing or models; the benign-only Isolation Forest path remains separate.
Those checks do not prove freedom from attack-specific patterns in remaining fields.

A descriptive audit hashed exact CSV string values in the final predictor columns
with `pandas.util.hash_pandas_object(index=False)`, then tested validation/test hashes
for membership in the training hashes. It found 67/20,000 validation rows (0.335%)
and 77/20,000 test rows (0.385%) matching training predictor rows. No identical
predictor hash in the selected sample had conflicting binary labels. This hash
check can miss near-duplicates and is not a session-independence proof. It did not
change the split, predictors, models or thresholds after seeing test results.

Permutation importance used an unbalanced uniform 1,000-row validation subset,
containing 76 attacks, with three repeats and AP scoring. Every single-column AP
decrease was effectively zero (floating-point noise below 2e-16). This output does
**not** justify an informative feature ranking or show that all features are
irrelevant: redundant/correlated predictors and saturated AP can mask reliance.
No causal interpretation is claimed.

The dataset is a testbed with severe imbalance. Row-level temporal/session
dependence, remaining identifier/protocol patterns, tiny rare-attack counts,
different supervision, and the conservative timing-field heuristic limit
generalization. There is no evidence here of production performance or detection
of previously unseen attacks. A verified capture/session split, interval-quality
handling chosen before testing, and independent datasets remain future work.

Raw data, observation-level exports, model binaries and runtime directories are
not included. The small aggregate record contains only counts, settings and
measured summaries. No software license was selected as part of this experiment.
