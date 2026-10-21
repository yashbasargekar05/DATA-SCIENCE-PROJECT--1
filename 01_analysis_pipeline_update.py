# ============================================================
# Nassau Candy Distributor — Factory-to-Customer Shipping
# Route Efficiency Analysis — CORRECTED PIPELINE
# ============================================================
# Fixes applied vs. the original notebook:
#   1. Is_Delayed / Lead_Time_Bucket redefined RELATIVELY (percentile-based)
#      instead of fixed day thresholds — the original always returned
#      100% delayed because Order Date (2024-25) and Ship Date (2026-30)
#      come from non-overlapping ranges.
#   2. Added the missing Factory -> Region route summary (route_region_summary)
#      alongside the existing Factory -> State summary (route_summary).
#   3. Is_Bottleneck is computed AND explicitly surfaced/printed.
#   4. ONE canonical state-level efficiency score (previously two
#      different scores existed — one from US-only rows, one from
#      US+Canada rows).
#   5. Canada is kept in all tables/exports but clearly labeled and
#      excluded only from the US choropleth map (which can't render it).
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe; remove if running in Jupyter
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

print("Libraries imported successfully.")

# ============================================================
# STEP 1 — Load the data
# ============================================================
df = pd.read_csv("Nassau Candy Distributor.csv")
print("Shape:", df.shape)

# ============================================================
# STEP 2 — Reference tables
# ============================================================
factory_coords = pd.DataFrame({
    "Factory": ["Lot's O' Nuts", "Wicked Choccy's", "Sugar Shack", "Secret Factory", "The Other Factory"],
    "Latitude": [32.881893, 32.076176, 48.11914, 41.446333, 35.1175],
    "Longitude": [-111.768036, -81.088371, -96.18115, -90.565487, -89.971107],
})

product_factory_map = pd.DataFrame({
    "Division": ["Chocolate","Chocolate","Chocolate","Chocolate","Chocolate",
                 "Sugar","Sugar","Sugar","Sugar","Other",
                 "Sugar","Sugar","Other","Other","Other"],
    "Product Name": [
        "Wonka Bar - Nutty Crunch Surprise","Wonka Bar - Fudge Mallows","Wonka Bar -Scrumdiddlyumptious",
        "Wonka Bar - Milk Chocolate","Wonka Bar - Triple Dazzle Caramel",
        "Laffy Taffy","SweeTARTS","Nerds","Fun Dip","Fizzy Lifting Drinks",
        "Everlasting Gobstopper","Hair Toffee","Lickable Wallpaper","Wonka Gum","Kazookles"
    ],
    "Factory": [
        "Lot's O' Nuts","Lot's O' Nuts","Lot's O' Nuts","Wicked Choccy's","Wicked Choccy's",
        "Sugar Shack","Sugar Shack","Sugar Shack","Sugar Shack","Sugar Shack",
        "Secret Factory","The Other Factory","Secret Factory","Secret Factory","The Other Factory"
    ],
})

US_STATE_ABBREV = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO',
    'Connecticut':'CT','Delaware':'DE','District Of Columbia':'DC','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY',
    'Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN',
    'Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH',
    'New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND',
    'Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI',
    'South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT',
    'Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
}

# ============================================================
# STEP 3 — Clean and validate data
# ============================================================
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")

n_bad_dates = df["Order Date"].isna().sum() + df["Ship Date"].isna().sum()
print(f"Rows with unparseable dates: {n_bad_dates}")

invalid_date_order = df["Ship Date"] < df["Order Date"]
print(f"Rows where Ship Date < Order Date: {invalid_date_order.sum()}")
df = df[~invalid_date_order].copy()

before = len(df)
df = df.dropna(subset=["Order Date", "Ship Date"]).copy()
print(f"Rows dropped for missing/invalid dates: {before - len(df)}")

for col in ["State/Province", "City", "Region"]:
    df[col] = df[col].astype(str).str.strip().str.title()

before = len(df)
df = df.drop_duplicates().copy()
print(f"Exact duplicate rows dropped: {before - len(df)}")
print(f"Final cleaned shape: {df.shape}")

# ============================================================
# STEP 4 — Merge factory data
# ============================================================
df = df.merge(product_factory_map[["Product Name", "Factory"]], on="Product Name", how="left")
unmatched = df[df["Factory"].isna()]["Product Name"].unique()
print(f"Unmatched products (no factory mapping found): {len(unmatched)}")

df = df.merge(factory_coords, on="Factory", how="left")
df = df.rename(columns={"Latitude": "Factory_Lat", "Longitude": "Factory_Lon"})
print(f"Shape after merges: {df.shape}")

# ============================================================
# STEP 5 — Feature engineering
# ============================================================
df["Shipping Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

if df["Shipping Lead Time"].mean() > 100:
    print("DATA QUALITY FLAG: Mean Shipping Lead Time is "
          f"{df['Shipping Lead Time'].mean():.1f} days — Order Date and Ship "
          "Date appear to be drawn from non-corresponding ranges. All "
          "downstream lead-time/delay metrics are computed RELATIVE to this "
          "dataset's own distribution (percentile-based), not as literal "
          "shipping day counts, until the source data is corrected.")

df["Route"] = df["Factory"] + " -> " + df["State/Province"]
df["Route_Region"] = df["Factory"] + " -> " + df["Region"]

# ============================================================
# STEP 6 — RELATIVE delay & lead-time bucket logic  (FIX)
# ============================================================
# Because absolute day thresholds are meaningless here (see flag above),
# delay/bucket definitions are anchored to this dataset's own quantiles.
# DELAY_PERCENTILE is configurable — 75 means "slowest quartile counts as
# delayed", which yields a meaningful ~25% delay rate instead of 0% or 100%.
DELAY_PERCENTILE = 75
delay_threshold_days = df["Shipping Lead Time"].quantile(DELAY_PERCENTILE / 100)
df["Is_Delayed"] = df["Shipping Lead Time"] > delay_threshold_days

print(f"\nRelative delay threshold (P{DELAY_PERCENTILE}): "
      f"{delay_threshold_days:.1f} days")
print("Is_Delayed counts:")
print(df["Is_Delayed"].value_counts())
print(f"Delay rate: {100 * df['Is_Delayed'].mean():.1f}%")

# Relative lead-time bucket: Fast / Medium / Slow thirds of this dataset's
# own distribution, replacing the old fixed 0-2 / 3-5 / 6+ day buckets
# (which always landed 100% of rows in "6+ days").
df["Lead_Time_Bucket"] = pd.qcut(
    df["Shipping Lead Time"], q=3,
    labels=["Fast (relative)", "Medium (relative)", "Slow (relative)"]
)
print("\nLead_Time_Bucket counts:")
print(df["Lead_Time_Bucket"].value_counts())

print("\nShipping Lead Time stats:")
print(df["Shipping Lead Time"].describe())

# ============================================================
# STEP 7 — Exploratory Data Analysis (EDA)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df["Shipping Lead Time"], bins=40, kde=True, ax=axes[0])
axes[0].axvline(delay_threshold_days, color="red", linestyle="--", label=f"P{DELAY_PERCENTILE} delay cutoff")
axes[0].legend()
axes[0].set_title("Distribution of Shipping Lead Time")
sns.boxplot(x=df["Shipping Lead Time"], ax=axes[1])
axes[1].set_title("Shipping Lead Time — Boxplot")
lead_time_median = df["Shipping Lead Time"].median()
lead_time_mean = df["Shipping Lead Time"].mean()
lead_time_p10, lead_time_p90 = df["Shipping Lead Time"].quantile([0.10, 0.90])
delay_share = 100 * df["Is_Delayed"].mean()
if lead_time_mean > lead_time_median:
    distribution_note = "The higher mean suggests high lead-time values pull the average upward."
elif lead_time_mean < lead_time_median:
    distribution_note = "The lower mean suggests low lead-time values pull the average downward."
else:
    distribution_note = "Mean and median are similar, so the center is not strongly shifted."
fig.text(
    0.02, 0.02,
    f"Observation: median {lead_time_median:.0f}, mean {lead_time_mean:.0f}; "
    f"the middle 80% spans {lead_time_p10:.0f}–{lead_time_p90:.0f}. "
    f"{distribution_note} {delay_share:.1f}% are above the P{DELAY_PERCENTILE} cutoff.\n"
    "Consider: Order and Ship Dates come from non-overlapping ranges, so these are not literal transit days.",
    ha="left", va="bottom", fontsize=8, wrap=True,
)
plt.tight_layout(rect=(0, 0.17, 1, 1))
plt.savefig("eda_lead_time_distribution.png", dpi=100)
plt.close()

fig_mode, ax_mode = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x="Ship Mode", y="Shipping Lead Time", ax=ax_mode)
ax_mode.set_title("Lead Time by Ship Mode")
ship_mode_plot_summary = df.groupby("Ship Mode").agg(
    Shipment_Count=("Shipping Lead Time", "count"),
    Median_Lead_Time=("Shipping Lead Time", "median"),
).reset_index()
fastest_median_mode = ship_mode_plot_summary.loc[
    ship_mode_plot_summary["Median_Lead_Time"].idxmin()
]
slowest_median_mode = ship_mode_plot_summary.loc[
    ship_mode_plot_summary["Median_Lead_Time"].idxmax()
]
fig_mode.text(
    0.02, 0.02,
    f"Observation: {fastest_median_mode['Ship Mode']} has the lowest median "
    f"({fastest_median_mode['Median_Lead_Time']:.0f}; "
    f"n={fastest_median_mode['Shipment_Count']:,}); {slowest_median_mode['Ship Mode']} "
    f"has the highest ({slowest_median_mode['Median_Lead_Time']:.0f}; "
    f"n={slowest_median_mode['Shipment_Count']:,}).\n"
    "Consider: compare box heights and overlap as well as medians; group sizes differ, "
    "and the lead-time values are not literal transit days.",
    ha="left", va="bottom", fontsize=8, wrap=True,
)
fig_mode.tight_layout(rect=(0, 0.15, 1, 1))
plt.savefig("eda_lead_time_by_shipmode.png", dpi=100)
plt.close()

numeric_cols = ["Sales", "Units", "Cost", "Gross Profit", "Shipping Lead Time"]
corr = df[numeric_cols].corr()
fig_corr, ax_corr = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax_corr)
ax_corr.set_title("Correlation Heatmap")
lead_time_correlations = corr["Shipping Lead Time"].drop("Shipping Lead Time")
strongest_correlated_feature = lead_time_correlations.abs().idxmax()
strongest_correlation = lead_time_correlations[strongest_correlated_feature]
correlation_direction = "positive" if strongest_correlation > 0 else "negative"
fig_corr.text(
    0.02, 0.02,
    f"Observation: {strongest_correlated_feature} has the strongest linear association "
    f"with lead time among the other measures (r={strongest_correlation:.2f}, "
    f"{correlation_direction}).\n"
    "Consider: correlation does not establish cause, and it may miss non-linear patterns.",
    ha="left", va="bottom", fontsize=8, wrap=True,
)
fig_corr.tight_layout(rect=(0, 0.15, 1, 1))
plt.savefig("eda_correlation_heatmap.png", dpi=100)
plt.close()
print("\nSaved EDA charts: eda_lead_time_distribution.png, eda_lead_time_by_shipmode.png, eda_correlation_heatmap.png")

# ============================================================
# STEP 8 — Route-level aggregation (Factory -> State)
# ============================================================
route_summary = df.groupby("Route").agg(
    Shipment_Count=("Row ID", "count"),
    Mean_Lead_Time=("Shipping Lead Time", "mean"),
    Median_Lead_Time=("Shipping Lead Time", "median"),
    Std_Lead_Time=("Shipping Lead Time", "std"),
    Delay_Frequency_Pct=("Is_Delayed", lambda x: 100 * x.mean()),
).reset_index()
route_summary["Std_Lead_Time"] = route_summary["Std_Lead_Time"].fillna(0)
route_summary = route_summary.sort_values("Mean_Lead_Time", ascending=False)
print(f"\nTotal unique routes (Factory -> State): {len(route_summary)}")

# ============================================================
# STEP 9 — Route-Region aggregation (Factory -> Region)   (FIX: was missing)
# ============================================================
route_region_summary = df.groupby("Route_Region").agg(
    Shipment_Count=("Row ID", "count"),
    Mean_Lead_Time=("Shipping Lead Time", "mean"),
    Median_Lead_Time=("Shipping Lead Time", "median"),
    Std_Lead_Time=("Shipping Lead Time", "std"),
    Delay_Frequency_Pct=("Is_Delayed", lambda x: 100 * x.mean()),
).reset_index()
route_region_summary["Std_Lead_Time"] = route_region_summary["Std_Lead_Time"].fillna(0)
route_region_summary = route_region_summary.sort_values("Mean_Lead_Time", ascending=False)
print(f"Total unique routes (Factory -> Region): {len(route_region_summary)}")

# ============================================================
# STEP 10 — Efficiency scoring (helper, reused for route/region/state)
# ============================================================
def add_efficiency_score(frame, lead_time_col="Mean_Lead_Time"):
    min_lt, max_lt = frame[lead_time_col].min(), frame[lead_time_col].max()
    frame = frame.copy()
    if max_lt == min_lt:
        frame["Efficiency_Score"] = 100.0
    else:
        frame["Efficiency_Score"] = 100 * (1 - (frame[lead_time_col] - min_lt) / (max_lt - min_lt))
    return frame

route_summary = add_efficiency_score(route_summary).rename(columns={"Efficiency_Score": "Route_Efficiency_Score"})
route_region_summary = add_efficiency_score(route_region_summary).rename(columns={"Efficiency_Score": "Route_Efficiency_Score"})

top10_efficient = route_summary.sort_values("Route_Efficiency_Score", ascending=False).head(10)
bottom10_efficient = route_summary.sort_values("Route_Efficiency_Score", ascending=True).head(10)
print("\n=== Top 10 Most Efficient Routes (Factory -> State) ===")
print(top10_efficient[["Route", "Mean_Lead_Time", "Route_Efficiency_Score", "Delay_Frequency_Pct", "Shipment_Count"]].round(2))
print("\n=== Bottom 10 Least Efficient Routes (Factory -> State) ===")
print(bottom10_efficient[["Route", "Mean_Lead_Time", "Route_Efficiency_Score", "Delay_Frequency_Pct", "Shipment_Count"]].round(2))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(data=top10_efficient, y="Route", x="Route_Efficiency_Score", hue="Route", ax=axes[0], palette="Greens_r", legend=False)
axes[0].set_title("Top 10 Most Efficient Routes")
sns.barplot(data=bottom10_efficient, y="Route", x="Route_Efficiency_Score", hue="Route", ax=axes[1], palette="Reds", legend=False)
axes[1].set_title("Bottom 10 Least Efficient Routes")
best_route = route_summary.loc[route_summary["Mean_Lead_Time"].idxmin()]
slowest_route = route_summary.loc[route_summary["Mean_Lead_Time"].idxmax()]
fig.text(
    0.02, 0.02,
    f"Observation: {best_route['Route']} has the lowest average lead time "
    f"({best_route['Mean_Lead_Time']:.1f}; n={best_route['Shipment_Count']:,}); "
    f"{slowest_route['Route']} has the highest "
    f"({slowest_route['Mean_Lead_Time']:.1f}; n={slowest_route['Shipment_Count']:,}).\n"
    "Consider: route scores are relative min–max rankings; small shipment counts make "
    "route averages less stable, and lead-time values are not literal transit days.",
    ha="left", va="bottom", fontsize=8, wrap=True,
)
plt.tight_layout(rect=(0, 0.16, 1, 1))
plt.savefig("route_efficiency_top_bottom.png", dpi=100)
plt.close()

# ============================================================
# STEP 11 — Ship Mode Comparison
# ============================================================
ship_mode_summary = df.groupby("Ship Mode").agg(
    Shipment_Count=("Shipping Lead Time", "count"),
    Mean_Lead_Time=("Shipping Lead Time", "mean"),
    Median_Lead_Time=("Shipping Lead Time", "median"),
    Delay_Frequency_Pct=("Is_Delayed", lambda x: 100 * x.mean()),
    Avg_Cost=("Cost", "mean"),
    Avg_Sales=("Sales", "mean"),
    Avg_Gross_Profit=("Gross Profit", "mean"),
).reset_index().sort_values("Mean_Lead_Time")
print("\n=== Ship Mode Comparison Summary ===")
print(ship_mode_summary.round(2))
ship_mode_summary.to_csv("ship_mode_summary.csv", index=False)

# ============================================================
# STEP 12 — Geographic bottleneck analysis (State/Province, US+Canada)
# ============================================================
state_summary = df.groupby("State/Province").agg(
    Shipment_Volume=("Row ID", "count"),
    Mean_Lead_Time=("Shipping Lead Time", "mean"),
    Median_Lead_Time=("Shipping Lead Time", "median"),
    Delay_Frequency_Pct=("Is_Delayed", lambda x: 100 * x.mean()),
).reset_index()

vol_p75 = state_summary["Shipment_Volume"].quantile(0.75)
lt_p75 = state_summary["Mean_Lead_Time"].quantile(0.75)
state_summary["Is_Bottleneck"] = (
    (state_summary["Shipment_Volume"] > vol_p75) &
    (state_summary["Mean_Lead_Time"] > lt_p75)
)

state_summary["State_Abbrev"] = state_summary["State/Province"].map(US_STATE_ABBREV)
state_summary = add_efficiency_score(state_summary).rename(columns={"Efficiency_Score": "State_Efficiency_Score"})

# FIX: this is now the ONE canonical state efficiency score, used both
# in the table below and in the map in Step 13 (no second recomputation).
print("\n=== Geographic Bottleneck Flags (Is_Bottleneck = True) ===")
bottleneck_states = state_summary[state_summary["Is_Bottleneck"]]
print(bottleneck_states[["State/Province", "Shipment_Volume", "Mean_Lead_Time", "Delay_Frequency_Pct"]].round(2))
print(f"({len(bottleneck_states)} of {len(state_summary)} states/provinces flagged as bottlenecks: "
      f"> P75 shipment volume AND > P75 mean lead time)")

unmapped = state_summary[state_summary["State_Abbrev"].isna()]["State/Province"].tolist()
print(f"\nNon-US provinces (no abbreviation, excluded from US map only, retained in tables): {unmapped}")

# ============================================================
# STEP 13 — Geographic visualization (US-only map, using the same
#           canonical State_Efficiency_Score computed in Step 12)
# ============================================================
state_map_data = state_summary[state_summary["State_Abbrev"].notna()].copy()

fig = px.choropleth(
    state_map_data,
    locations="State_Abbrev",
    locationmode="USA-states",
    color="Mean_Lead_Time",
    color_continuous_scale="RdYlGn_r",
    scope="usa",
    hover_name="State/Province",
    hover_data={
        "State_Abbrev": False,
        "Shipment_Volume": True,
        "Mean_Lead_Time": ":.1f",
        "Delay_Frequency_Pct": ":.1f",
        "Is_Bottleneck": True,
    },
    title="Average Shipping Lead Time by State (US) — relative scale",
    labels={"Mean_Lead_Time": "Avg Lead Time (days)"},
)
fig.add_trace(go.Scattergeo(
    lon=factory_coords["Longitude"], lat=factory_coords["Latitude"],
    text=factory_coords["Factory"], mode="markers+text",
    marker=dict(size=14, color="black", symbol="star", line=dict(width=1, color="white")),
    textposition="top center", textfont=dict(size=11, color="black"),
    name="Factories", showlegend=True,
))
lowest_map_state = state_map_data.loc[state_map_data["Mean_Lead_Time"].idxmin()]
highest_map_state = state_map_data.loc[state_map_data["Mean_Lead_Time"].idxmax()]
map_insight = (
    f"Observation: lowest average is {lowest_map_state['State/Province']} "
    f"({lowest_map_state['Mean_Lead_Time']:.1f}; "
    f"n={lowest_map_state['Shipment_Volume']:,}); highest is "
    f"{highest_map_state['State/Province']} "
    f"({highest_map_state['Mean_Lead_Time']:.1f}; "
    f"n={highest_map_state['Shipment_Volume']:,}). "
    f"{int(state_map_data['Is_Bottleneck'].sum())} mapped states meet the bottleneck "
    "rule (high volume and high average lead time). "
    "Consider: compare shipment volumes; values are relative, not literal transit days."
)
fig.add_annotation(
    text=map_insight,
    xref="paper", yref="paper", x=0.01, y=-0.12,
    xanchor="left", yanchor="top", align="left",
    showarrow=False, font=dict(size=12),
)
fig.update_layout(geo=dict(scope="usa", projection_type="albers usa"), height=700, margin=dict(l=0, r=0, t=60, b=115))
fig.write_html("geo_choropleth_map.html")
print("\nSaved: geo_choropleth_map.html")

# ============================================================
# STEP 14 — Consolidated KPI summary (Route + Route_Region + State)
# ============================================================
route_kpis = route_summary.rename(columns={"Route": "Entity"})[[
    "Entity", "Shipment_Count", "Mean_Lead_Time", "Median_Lead_Time",
    "Delay_Frequency_Pct", "Route_Efficiency_Score"
]].copy()
route_kpis.insert(0, "Level", "Route (Factory->State)")
route_kpis["Is_Bottleneck"] = np.nan

route_region_kpis = route_region_summary.rename(columns={"Route_Region": "Entity"})[[
    "Entity", "Shipment_Count", "Mean_Lead_Time", "Median_Lead_Time",
    "Delay_Frequency_Pct", "Route_Efficiency_Score"
]].copy()
route_region_kpis.insert(0, "Level", "Route (Factory->Region)")
route_region_kpis["Is_Bottleneck"] = np.nan

state_kpis = state_summary.rename(columns={
    "State/Province": "Entity", "Shipment_Volume": "Shipment_Count",
    "State_Efficiency_Score": "Route_Efficiency_Score"
})[[
    "Entity", "Shipment_Count", "Mean_Lead_Time", "Median_Lead_Time",
    "Delay_Frequency_Pct", "Route_Efficiency_Score", "Is_Bottleneck"
]].copy()
state_kpis.insert(0, "Level", "State/Province")

kpi_summary = pd.concat([route_kpis, route_region_kpis, state_kpis], ignore_index=True).round(2)
print(f"\nTotal KPI rows: {len(kpi_summary)} "
      f"({(kpi_summary['Level']=='Route (Factory->State)').sum()} routes + "
      f"{(kpi_summary['Level']=='Route (Factory->Region)').sum()} region-routes + "
      f"{(kpi_summary['Level']=='State/Province').sum()} states/provinces)")

# ============================================================
# STEP 15 — Export
# ============================================================
df.to_csv("cleaned_shipments.csv", index=False)
route_summary.to_csv("route_summary.csv", index=False)
route_region_summary.to_csv("route_region_summary.csv", index=False)
state_summary.to_csv("state_summary.csv", index=False)
kpi_summary.to_csv("kpi_summary.csv", index=False)
ship_mode_summary.to_csv("ship_mode_summary.csv", index=False)

print("\nExported files:")
for fname, shape in [
    ("cleaned_shipments.csv", df.shape),
    ("route_summary.csv", route_summary.shape),
    ("route_region_summary.csv", route_region_summary.shape),
    ("state_summary.csv", state_summary.shape),
    ("kpi_summary.csv", kpi_summary.shape),
    ("ship_mode_summary.csv", ship_mode_summary.shape),
]:
    print(f"  {fname:<28} -> {shape}")
