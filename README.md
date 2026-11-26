# Canadian road safety in 2019

**CSCI 6686 Big Data Analytics · Deliverable 2**  
Xiaoshuang Yang and Ruxue Yang

This project studies fatal outcomes among recorded Canadian fatal and non-fatal injury collisions. PySpark reduces person and vehicle records to one row per collision, builds weather and surface summaries, and times two local partition settings. Regularized logistic regression compares a control-only specification with the same model plus weather and surface.

## Reproduce the analysis

Use Python 3.11 or 3.12 and Java 17 on a local machine. Spark 3.5.7 also ran under OpenJDK 21 in the recorded analysis environment. In Terminal from this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data
```

Download the English 2019 CSV from [Transport Canada’s official resource page](https://open.canada.ca/data/en/dataset/1eb9eba7-71d1-4b30-9fb1-30cbdab7e63a/resource/1e2a6aa0-c35f-4736-984a-71f2eac1c1dc) and place it at `data/2019_dataset_en.csv`. The supplied file used in the report has SHA-256 `d1d4fcc176b387281dadbf02b9e32890174e7e9e07f05961c54e5e7c4487a87c`. The official resource includes the data dictionary. Data attribution goes to Transport Canada under the [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada).

```bash
python analysis.py --input data/2019_dataset_en.csv --out results
```

If your CSV has a different local name, pass that path to `--input`. The notebook `Demo.ipynb` displays the recorded artifacts and contains a cell to rerun the full pipeline. To run the notebook interactively, install JupyterLab with `python -m pip install jupyterlab` and launch `jupyter lab` from the repository root.

The CSV is read by Spark with string-valued fields. Spark normalizes keys and codes, groups by collision key, checks for conflicting values, filters invalid labels and aggregates fatal outcomes by weather and surface. The same joint summary is checked against an independent pandas calculation for the supplied file. The model uses unique collision records with a fixed, stratified holdout. Timing forces the Spark result to materialize and checks agreement for every run.

## Result files

| File | Contents |
| --- | --- |
| `audit.json` | Source dimensions, blanks, duplicates, counts, hash and runtime metadata |
| `weather_surface.csv`, `environment.csv` | Spark joint counts and marginal fatal shares with Wilson intervals |
| `metrics.csv`, `coefficients.csv` | Primary holdout metrics and learned coefficients |
| `metrics_known.csv`, `coefficients_known.csv` | Sensitivity restricted to records with standard codes on all predictors |
| `timing_runs.csv`, `timing_summary.csv` | All measured runs and median/range |
| `*.png` | Fatal-share, precision–recall and processing-performance charts |
| `collisions.csv`, `split_assignments.csv`, `test_predictions.csv` | Generated locally by a full run |

The report accompanies this repository. See [the viva guide](VIVA.md) for the demonstration and team contributions.

## AI assistance

OpenAI ChatGPT (Codex) provided basic support with coding and assisted with report drafting and review.
