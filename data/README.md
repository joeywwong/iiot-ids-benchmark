# Local dataset access and provenance

Download **WUSTL-IIoT-2021** from the dataset link on the
[official Washington University page](https://www.cse.wustl.edu/~jain/iiot2/index.html)
or its [university mirror](https://research.engineering.wustl.edu/~jain/iiot2/index.html).
Extract locally into `data/raw/`. Pass the actual CSV filename with `--data`; no
renaming is needed. No dataset is distributed with this project.

Keep `Traffic` and the original metadata columns in the input: splitting examines
timestamps before feature exclusion. The program requires `StartTime`, `LastTime`,
`SrcAddr`, `DstAddr`, `sIpId`, `dIpId` and `Traffic` (case/spacing variants accepted).
The six author-identified leakage fields never enter predictors. Distinct `Traffic`
values must have explicit normal=0 / attack=1 semantics in `label_mapping`.
The official page documents categories but does not establish all CSV spellings;
unknown values fail with a message rather than an assumed label.

Inspect without fitting a model:

```powershell
python -c "import pandas as pd; p='data/raw/WUSTL-IIoT-2021.csv'; print(pd.read_csv(p,nrows=0).columns.tolist()); print(pd.read_csv(p,usecols=['Traffic']).value_counts())"
```

Use the source-provided data under its applicable terms. The run records the file
SHA-256, file size, full observed class counts and discrepancies against reported
properties. Input paths are not stored in manifests; only the basename is recorded.
Raw/processed data and archives are ignored; do not publish them in Git or CI.

Attribution: M. Zolanvari, M. A. Teixeira, L. Gupta, K. M. Khan, and R. Jain.
“WUSTL-IIOT-2021 Dataset for IIoT Cybersecurity Research,” Washington University in
St. Louis, USA, October 2021. This is a research testbed dataset.
