# SpaceX Falcon 9 First-Stage Landing Prediction

Predicting whether the Falcon 9 first stage will land successfully, using public SpaceX launch data, SQL and visual exploration, geospatial analysis, an interactive dashboard, and machine-learning classifiers.

SpaceX lists a Falcon 9 launch at about **$62 million**; other providers charge upward of **$165 million**, mostly because SpaceX reuses the first stage. Predicting whether the first stage lands is therefore a proxy for the cost of a launch — useful to a competitor bidding against SpaceX.

> Completed as part of the IBM Applied Data Science Capstone (Coursera / Skills Network).

---

## Key results

| Metric | Value |
|---|---|
| Falcon 9 launches in the modelling dataset | 90 |
| Overall first-stage landing success rate | 66.7% (60 landed / 30 not) |
| Best cross-validation accuracy | 87.5% (decision tree) |
| Test accuracy (all four tuned models) | 83.3% (15 of 18 correct) |
| Site with the largest share of successful launches | KSC LC-39A (41.7%) |

- Landing success rises with experience: before March 2017 only 8 of 31 recorded landing outcomes were successful landings, and "No attempt" was the most common outcome.
- Launch sites sit right on the coast (CCAFS LC-40 is 0.58 km from the shoreline), close to highways and railways, and well away from cities (about 53 km to Melbourne, FL).
- All four models catch every real landing in the test set (recall 100%) but predict 3 failed landings as successes (precision 80%).

---

## Repository structure

```
├── notebooks/
│   ├── 1_data_collection_api.ipynb            # SpaceX REST API requests and data cleaning
│   ├── 2_data_wrangling.ipynb                 # Profiling, missing values, Class label
│   ├── 3_eda_sql_sqlite.ipynb                 # Exploratory analysis with SQL (SQLite)
│   ├── 4_launch_site_location_folium.ipynb    # Interactive map with Folium
│   └── 5_machine_learning_prediction.ipynb    # Model training, tuning and evaluation
├── images/
│   └── spacex_dashboard.png                   # Plotly Dash dashboard screenshot
├── presentation/
│   ├── SpaceX_Falcon9_Landing_Prediction.pptx
│   └── SpaceX_Falcon9_Landing_Prediction.pdf
└── README.md
```

Rename the uploaded notebooks to match, or update the paths above.

---

## Methodology

### 1. Data collection — SpaceX REST API
- `GET` request to `https://api.spacexdata.com/v4/launches/past`; a static JSON snapshot provided by the course is used for reproducible results.
- `pd.json_normalize()` flattens the nested JSON into a DataFrame.
- Helper functions resolve ID columns through further API calls:
  - `getBoosterVersion` → booster name (`/rockets`)
  - `getLaunchSite` → site name, longitude, latitude (`/launchpads`)
  - `getPayloadData` → payload mass, orbit (`/payloads`)
  - `getCoreData` → landing outcome and type, flights, grid fins, legs, reuse count, block, serial (`/cores`)
- Launches with multiple cores or payloads are removed, dates are restricted to on or before 2020-11-13, Falcon 1 launches are dropped, and missing `PayloadMass` values are replaced with the column mean.

### 2. Data wrangling
- Profiled 90 rows × 17 columns; only `LandingPad` has missing values (28.9%, meaning no pad was used).
- `value_counts()` on launch sites (CCAFS SLC 40: 55, KSC LC 39A: 22, VAFB SLC 4E: 13) and orbits (GTO: 27, ISS: 21, VLEO: 14, …).
- Created the binary target `Class`: outcomes in `{False ASDS, False Ocean, False RTLS, None ASDS, None None}` → `0`, otherwise `1`.

### 3. Exploratory data analysis with SQL
- Loaded `Spacex.csv` into SQLite (`sqlite3` + `ipython-sql`), dropping blank rows into `SPACEXTABLE` (101 rows).
- Queries use `DISTINCT`, `LIKE`, `SUM`/`AVG`/`MIN`/`MAX`, `GROUP BY`, subqueries and `substr()` for dates.
- Findings include four distinct launch sites, 100 successful missions out of 101, 12 Block 5 boosters that carried the maximum payload, and a ranking of landing outcomes between 2010-06-04 and 2017-03-20.

### 4. Interactive visual analytics
- **Folium:** marked every launch site, clustered launch outcomes (green = landed, red = failed), and measured distances to the coastline, highway, railway and nearest city with the Haversine formula.
- **Plotly Dash:** a dashboard with a launch-site dropdown and a payload range slider driving a success pie chart and a payload-vs-outcome scatter plot.

![SpaceX Launch Records Dashboard](images/spacex_dashboard.png)

### 5. Predictive analysis
- Features from `dataset_part_3.csv` (numeric plus one-hot encoded categoricals), standardised with `StandardScaler`.
- `train_test_split` with `test_size=0.2`, `random_state=2` → 72 training and 18 test samples.
- Hyperparameters tuned with `GridSearchCV` (10-fold CV):

| Model | Best parameters | CV accuracy | Test accuracy |
|---|---|---|---|
| Logistic regression | C = 0.01, L2, lbfgs | 0.846 | 0.833 |
| Support vector machine | sigmoid kernel, C = 1.0, gamma = 0.0316 | 0.848 | 0.833 |
| Decision tree | gini, max_depth 12, sqrt features, min_samples_leaf 2, min_samples_split 10, random splitter | **0.875** | 0.833 |
| K-nearest neighbours | k = 10, p = 1 (Manhattan) | 0.848 | 0.833 |

Confusion matrix (identical for all four models): TN = 3, FP = 3, FN = 0, TP = 12.

---

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install pandas numpy requests matplotlib seaborn scikit-learn folium dash plotly ipython-sql prettytable
jupyter notebook
```

Run the notebooks in order (1 → 5). The datasets are read directly from IBM Skills Network cloud storage URLs inside each notebook, so an internet connection is required.

---

## Known issues and next steps

- **SQL tasks 3–5** use string filters that don't match the data. Corrected versions:
  ```sql
  -- Task 3
  SELECT SUM(PAYLOAD_MASS__KG_) FROM SPACEXTABLE WHERE Customer = 'NASA (CRS)';
  -- Task 4
  SELECT AVG(PAYLOAD_MASS__KG_) FROM SPACEXTABLE WHERE Booster_Version LIKE 'F9 v1.1%';
  -- Task 5
  SELECT MIN(Date) FROM SPACEXTABLE WHERE Landing_Outcome = 'Success (ground pad)';
  ```
- **Data collection:** the live API returned an SSL error in the saved run; the Falcon 9 filter should be `data[data['BoosterVersion'] != 'Falcon 1']`; the mean-imputation step for `PayloadMass` still needs code.
- **Decision tree grid:** `max_features='auto'` was removed in recent scikit-learn versions, so half the grid fails; replace it with `None` or `'log2'`.
- **Small test set:** 18 samples cause all models to tie; repeated stratified cross-validation would separate them better.
- **Class imbalance:** failures are one third of the data; `class_weight='balanced'` or tuning on F1 could reduce false positives.
- **Folium distances:** the railway, highway and city coordinates are approximate and should be re-measured on the map.

---

## Tools

Python · pandas · NumPy · requests · SQLite / ipython-sql · Folium · Plotly Dash · scikit-learn · seaborn / matplotlib

## Acknowledgements

Lab templates and datasets from the IBM Skills Network *Applied Data Science Capstone*. Data originally from the [SpaceX API](https://github.com/r-spacex/SpaceX-API).
