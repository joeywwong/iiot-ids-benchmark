# Supervised and Anomaly-Based Intrusion Detection for Industrial IoT Network Traffic

An offline Python benchmark of binary attack detection on **WUSTL-IIoT-2021** network
flows. The research question is how supervised classifiers and a normal-only anomaly
detector compare on attack recall at a false-positive constraint, per-attack recall,
and batch inference cost.

| Baseline | Training information | Rationale |
| --- | --- | --- |
| Logistic Regression | Labeled normal and attack training rows | Scaled, linear reference; balanced class weights |
| Random Forest | The same labeled training rows | Nonlinear tabular reference; balanced subsample weights |
| Isolation Forest | Only normal training rows, selected using labels | Normal-only anomaly reference; higher score = more suspicious |

The models **receive different training supervision**. Isolation Forest's selection
of normal rows uses labels; this is not a fully label-free experiment. This project
does not capture live traffic, classify attack families, or establish production,
security, scientific novelty, or unseen-attack detection claims.

## Data and setup

Use Python 3.11+ on a laptop; no GPU or cloud service is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On Linux/macOS activate with `source .venv/bin/activate`.

1. Open the [official dataset page](https://www.cse.wustl.edu/~jain/iiot2/index.html)
   ([university mirror](https://research.engineering.wustl.edu/~jain/iiot2/index.html)).
2. Follow its dataset download link and extract the CSV locally into `data/raw/`.
   Keep the original filename and schema. See [data access notes](data/README.md).
3. Inspect the CSV header and distinct `Traffic` values. The loader requires the
   original six identifier/time columns and `Traffic`; ambiguous/duplicate column
   names and unknown labels fail explicitly. Normalized spellings are accepted.
   The default vocabulary is in `configs/default.yaml`; if the original file uses
   different category spellings, document their semantics and edit `label_mapping`.
   Unknown labels are never silently interpreted as attacks.

Dataset attribution: M. Zolanvari, M. A. Teixeira, L. Gupta, K. M. Khan, and R. Jain.
“WUSTL-IIOT-2021 Dataset for IIoT Cybersecurity Research,” Washington University in
St. Louis, USA, October 2021. The authors report 1,194,464 observations, 1,107,448
normal and 87,016 attack observations, and 41 features. These are **source-reported
properties**, not measurements from this project. Each run scans all labels and
records observed counts, discrepancies, the input SHA-256 and sampled counts.
The data come from a research testbed, not production industrial infrastructure.

## Run and reproduce

After installing the package, replace the example filename with the extracted CSV:

```powershell
python -m iiot_ids_benchmark.cli run --data data/raw/WUSTL-IIoT-2021.csv --output results/
```

Defaults retain a **uniform, unbalanced subset of up to 100,000 rows**, with seed 42,
100 trees per forest and one worker. The entire CSV is scanned for label validation,
counts and bounded random sampling; this is not a head-of-file sample. Memory is
roughly the retained sample plus a 50,000-row input chunk and model overhead.
Forest depth is capped at 20, leaf size at 2; Isolation Forest uses up to 256 rows
per tree. Logistic Regression uses `C=1`, liblinear, and 2,000 maximum iterations.
Categorical encoding caps each feature at 64 categories; missing numeric values
and infinities are imputed. A full-data run requires more RAM and runtime:

```powershell
python -m iiot_ids_benchmark.cli run --data data/raw/WUSTL-IIoT-2021.csv --config configs/default.yaml --max-rows 0 --rf-trees 100 --n-jobs 2 --output results/full/
python -m iiot_ids_benchmark.cli run --help
```

CLI flags override YAML, then built-in defaults. Use a new output directory for
each run; existing experiment artifacts are protected from overwriting. Versions,
settings, feature typing per model, estimator parameters, source-row split hashes,
hardware context and timing setup are stored in `manifest.json`. For repeat runs,
use identical input hash, config and library versions. Wall-clock timings vary.

## Protocol and leakage controls

- Default proportions are 60% train / 20% validation / 20% held-out test, before
  chronological purging; actual counts and prevalence are always recorded.
- Before dropping timestamps, `auto` checks complete ISO or epoch `StartTime` /
  `LastTime` intervals. When feasible, it sorts chronologically, keeps equal start
  times together and purges earlier flows ending at/after the next split boundary.
  Training must contain both classes and validation must contain normal rows.
  If those conditions fail, `auto` records the reason and falls back to stratified
  rows. `--split chronological` fails instead of falling back. No capture/session
  ID is assumed from IP addresses; a verified one can use `--split group
  --group-column SessionId`, which excludes that field and keeps groups separate.
  Group proportions are approximate and missing attack categories are reported.
- Row splits stratify by original `Traffic` when counts permit, otherwise by the
  binary target. Rare categories may be missing. Related sessions, repeated flows
  and temporal dependence can inflate row-split results.
- Predictors exclude **StartTime, LastTime, SrcAddr, DstAddr, sIpId, dIpId**,
  `Traffic`, split groups, other target-like names, exact target copies/complements,
  constants, suspicious identifiers and very high-cardinality text fields.
  This conservative review uses training rows only and records every exclusion;
  it cannot guarantee all leakage has been removed. Review unexpected predictors
  manually. Flow aggregates may require a completed flow at inference time.
- Each model fits its typing, imputer, encoder and (for Logistic Regression)
  scaler only on its fitting rows. Isolation Forest fits the entire pipeline on
  benign training rows; a shared feature audit uses labeled training observations.
  Validation and test never fit transforms or estimators.
- Scores are Logistic Regression decision values, Random Forest attack
  probabilities, and **negative** Isolation Forest `score_samples`.

The primary comparison is **test attack recall at a target empirical validation
FPR of 1%**. For each model, only normal validation scores set the threshold.
With `n` such scores, at most `floor(target_fpr * n)` may exceed the threshold.
The threshold is the `(n - floor(target_fpr*n))`-th score in ascending order
(one-based); a score **strictly above** it is an attack, while ties are normal.
Discrete scores can make this conservative; fewer than 100 normal validation rows
permit no validation false positives at 1%. Test FPR may exceed the target.
Test labels never choose thresholds or model settings.

## Outputs and current evidence

**No real dataset experiment has been run; real benchmark results are pending.**
Synthetic fixtures verify implementation and cannot substantiate dataset performance.
Real results must be inspected before publication; no metric table or performance
chart is supplied as a measured WUSTL result here.

Every successful run writes:

- `metrics.csv`: split sizes/prevalence, actual fitting size, threshold, validation
  FPR, test precision/recall/F1/FPR, average precision (the PR-AUC summary used
  here), ROC-AUC, training time, median batch inference time and throughput.
- `per_attack_recall.csv`: **binary detection recall by original attack category**,
  with test counts and undefined recall for categories absent from test.
- `confusion_matrices.json` / `.png` and `precision_recall.png`: identical held-out
  rows and label order; PR curves are descriptive only.
- `permutation_importance.csv`: Random Forest decrease in average precision after
  permuting original feature columns on a uniform validation subset capped at
  1,000 rows, with three repeats. Correlation can mask importance; it is not causal.
  If the subset contains one class, importance is explicitly undefined.
- `manifest.json`: provenance, configuration, feature audit, split/category counts,
  calibration convention, versions and approximate machine context.

Undefined metric denominators and single-class AUCs are blank in CSV, not replaced
with perfect scores. Timing includes transformation and scoring, after one warmup,
with the median of three whole-test batches. Amortized flow latency is **offline
batch timing**, not real-time per-packet latency. Data, outputs and model binaries
are ignored by default; review small aggregate outputs before deciding to publish.

## Verification and limitations

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

Tests use explicitly synthetic fixtures, covering leakage exclusions, label failures,
disjoint splits/groups, training-only transforms, benign-only Isolation Forest fits,
anomaly score direction, threshold ties, FPR denominators and an end-to-end CLI run.
CI runs these checks without downloading data. `--data-kind synthetic` marks fixture
reports and plots; keep them separate from real experiment output.

Limitations include testbed realism, severe imbalance, rare attack categories,
possible row-level dependence, normal-selection label information, fixed offline
flow features, heuristic leakage checks and non-causal explanations. None of these
models guarantees detection of an unseen attack. Future work: documented capture /
session-aware validation, additional IIoT/ICS datasets, concept drift, protocol-aware
rules and interpretable per-alert analysis. A code license has not been selected;
dataset rights remain with its providers.
