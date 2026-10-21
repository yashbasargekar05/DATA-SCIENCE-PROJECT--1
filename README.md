# Nassau Candy Distributor — Shipping Route Efficiency Analysis

## Setup
```bash
pip install -r requirements.txt
```
(Pinned versions tested: pandas 3.0.2, numpy 2.4.4, matplotlib 3.10.8,
seaborn 0.13.2, plotly 7.1.0, streamlit 1.64.0.)

## 1. Run the analysis pipeline
Place the raw source file `Nassau Candy Distributor.csv` in the same folder, then:
```bash
python 01_analysis_pipeline.py
```
This produces:
- `cleaned_shipments.csv` — full cleaned dataset with all derived fields
- `route_summary.csv` — Factory → State route aggregation
- `route_region_summary.csv` — Factory → Region route aggregation
- `state_summary.csv` — State/Province-level aggregation + bottleneck flags
- `kpi_summary.csv` — consolidated KPI table (routes + region-routes + states)
- `ship_mode_summary.csv` — ship mode comparison
- `eda_*.png`, `route_efficiency_top_bottom.png` — EDA charts
- `geo_choropleth_map.html` — standalone interactive US map

## 2. Run the dashboard
Once the CSVs above exist in the same folder:
```bash
streamlit run 02_streamlit_app.py
```

## Important data note
Order Date and Ship Date in the source data come from non-overlapping ranges
(2024–25 vs. 2026–30), so raw "days between order and ship" is not a literal
shipping duration. All delay/efficiency metrics in this project are computed
**relative to the dataset's own distribution** (percentile-based) rather than
as absolute day counts, and should be read as comparative rankings across
routes/states/ship modes — see the "DATA QUALITY FLAG" printed by the
pipeline and the note shown at the top of the dashboard.

## Testing performed
Both scripts were verified in a clean, isolated virtual environment built
only from `requirements.txt`:
- `01_analysis_pipeline.py` runs end-to-end with exit code 0 from the raw CSV.
- `02_streamlit_app.py` was tested with Streamlit's official headless
  `AppTest` API across 9 scenarios: initial load, ship-mode filtering,
  delay-percentile slider changes, state filtering (including the
  Virginia/West Virginia exact-match check), an empty-filter-result edge
  case, filter recovery, route-granularity toggling, and the drill-down tab
  — all passed with zero exceptions and figures cross-checked against
  ground-truth pandas calculations.
