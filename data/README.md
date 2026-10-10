# Local dataset access and provenance

Download **WUSTL-IIoT-2021** from the dataset link on the
[official Washington University page](https://www.cse.wustl.edu/~jain/iiot2/index.html)
or its [university mirror](https://research.engineering.wustl.edu/~jain/iiot2/index.html).
The page links to `ftp/wustl_iiot_2021.zip`, which contains
`wustl_iiot_2021.csv`. Extract locally into `data/raw/` and pass that filename with
`--data`; no renaming is needed. No dataset is distributed with this project.

Keep `Traffic` and the original metadata columns in the input: splitting examines
timestamps before feature exclusion. The program requires `StartTime`, `LastTime`,
`SrcAddr`, `DstAddr`, `sIpId`, `dIpId` and `Traffic` (case/spacing variants accepted).
The six author-identified leakage fields never enter predictors. Distinct `Traffic`
values must have explicit normal=0 / attack=1 semantics in `label_mapping`.
The original university CSV inspected on 2026-10-11 has the values `normal`,
`DoS`, `Reconn`, `CommInj`, and `Backdoor`. `Reconn` means reconnaissance and
`CommInj` means command injection; these aliases and the supported long-form
names are accepted with case/whitespace normalization. Unknown values still
fail with a message rather than an assumed label.

The CSV has 49 columns: six author-excluded metadata fields, two label fields
(`Traffic` and `Target`), and 41 other fields before the training-only feature
audit. `Target` matches the binary mapping in all 1,194,464 rows but never enters
predictors. The measured class counts and checksum are in the
[experiment record](../docs/experiments/wustl_iiot_2021.md). The loader's required
columns matched the original file; no schema relaxation was needed.

Inspect without fitting a model:

```powershell
python -c "import pandas as pd; p='data/raw/wustl_iiot_2021.csv'; print(pd.read_csv(p,nrows=0).columns.tolist()); print(pd.read_csv(p,usecols=['Traffic']).value_counts())"
```

Use the source-provided data under its applicable terms. The run records the file
SHA-256, file size, full observed class counts and discrepancies against reported
properties. Input paths are not stored in manifests; only the basename is recorded.
Raw/processed data and archives are ignored; do not publish them in Git or CI.

Attribution: M. Zolanvari, M. A. Teixeira, L. Gupta, K. M. Khan, and R. Jain.
“WUSTL-IIOT-2021 Dataset for IIoT Cybersecurity Research,” Washington University in
St. Louis, USA, October 2021. This is a research testbed dataset.
