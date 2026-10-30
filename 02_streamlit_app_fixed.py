# ============================================================
# Nassau Candy — Shipping Route Efficiency Dashboard
# ------------------------------------------------------------
# Expects in the working directory:
#   cleaned_shipments.csv, kpi_summary.csv
# Requires: streamlit, pandas, numpy, plotly (see requirements.txt)
# ============================================================

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="🍬 Nassau Candy Dashboard", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------
# Constants / compatibility
# ------------------------------------------------------------
REQUIRED_COLUMNS = [
    "Row ID", "Order Date", "Region", "State/Province", "Ship Mode",
    "Shipping Lead Time", "Route", "Route_Region", "Product Name", "Cost", "Sales",
]
MIN_TREND_N = 20  # minimum shipments before a trend line is fitted


def _streamlit_version():
    try:
        return tuple(int(p) for p in st.__version__.split(".")[:2])
    except ValueError:
        return (0, 0)


# width="stretch" needs a recent Streamlit; older versions only know
# use_container_width. Pick whichever this install supports.
STRETCH = {"width": "stretch"} if _streamlit_version() >= (1, 50) else {"use_container_width": True}

US_STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District Of Columbia': 'DC',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI',
    'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
    'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
    'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
}

# ------------------------------------------------------------
# Load + validate data (cached)
# ------------------------------------------------------------
@st.cache_data
def load_data():
    for path in ("cleaned_shipments.csv", "kpi_summary.csv"):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    data = pd.read_csv("cleaned_shipments.csv")
    kpis = pd.read_csv("kpi_summary.csv")

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"cleaned_shipments.csv is missing columns: {', '.join(missing)}")

    data["Order Date"] = pd.to_datetime(data["Order Date"], errors="coerce")
    if "Factory" not in data.columns:
        data["Factory"] = data["Route"].astype(str).str.split("->").str[0].str.strip()
    return data, kpis


try:
    df, kpi_summary = load_data()
except FileNotFoundError as e:
    st.error(f"Required data file not found: {e}. Run the analysis pipeline first "
             "and launch the app from the folder containing its CSV exports.")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

st.title("📦 Nassau Candy Shipping Route Efficiency Dashboard")
st.markdown("""
Use the filters on the left to explore **routes, states, ship modes, and delays**.

⚠️ **Data note:** Order Date and Ship Date in the source data come from non-overlapping
ranges, so "lead time in days" is **not** literal shipping time. All delay/efficiency
metrics are **relative to this dataset's own distribution** (percentile-based) and should
be read as comparative rankings only. By construction, a percentile cutoff always labels
some share of shipments "delayed" — compare against the baseline shown, not against zero.
""")


# ------------------------------------------------------------
# Aggregation helper — always computed fresh from the frame passed in
# ------------------------------------------------------------
def summarize(frame, group_col, count_label="Shipment_Count", min_n=1):
    """Group summary. Groups with fewer than `min_n` shipments are flagged
    Low_Sample and get no Efficiency_Score, and are excluded from the score's
    min-max scaling (so a tiny group can't stretch the scale)."""
    cols = [group_col, count_label, "Mean_Lead_Time", "Median_Lead_Time",
            "Delay_Frequency_Pct", "Low_Sample", "Efficiency_Score"]
    if frame.empty:
        return pd.DataFrame(columns=cols)

    summary = frame.groupby(group_col).agg(
        **{count_label: ("Row ID", "count")},
        Mean_Lead_Time=("Shipping Lead Time", "mean"),
        Median_Lead_Time=("Shipping Lead Time", "median"),
        Delay_Frequency_Pct=("Is_Delayed", lambda x: 100 * x.mean()),
    ).reset_index()
    summary["Low_Sample"] = summary[count_label] < min_n

    eligible = summary.loc[~summary["Low_Sample"], "Mean_Lead_Time"]
    summary["Efficiency_Score"] = np.nan
    if not eligible.empty:
        lo, hi = eligible.min(), eligible.max()
        if hi == lo:
            summary.loc[~summary["Low_Sample"], "Efficiency_Score"] = 100.0
        else:
            summary.loc[~summary["Low_Sample"], "Efficiency_Score"] = (
                100 * (1 - (eligible - lo) / (hi - lo))
            )
    return summary


def eligible_rows(summary):
    return summary[~summary["Low_Sample"]] if not summary.empty else summary


# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.header("🔎 Filters")

full_start = df["Order Date"].min().date()
full_end = df["Order Date"].max().date()
date_range = st.sidebar.date_input("Order Date Range", [full_start, full_end])
regions = st.sidebar.multiselect("Region", sorted(df["Region"].dropna().unique()))
states = st.sidebar.multiselect("State/Province", sorted(df["State/Province"].dropna().unique()))
ship_modes = st.sidebar.multiselect("Ship Mode", sorted(df["Ship Mode"].dropna().unique()))

delay_percentile = st.sidebar.slider(
    "Delay cutoff (percentile)", min_value=50, max_value=95, value=75, step=5,
    help="Shipments slower than this percentile of the FULL dataset's lead-time "
         "distribution count as 'delayed'. Higher = stricter bar for 'delayed' "
         "(fewer shipments flagged)."
)
delay_threshold_days = df["Shipping Lead Time"].quantile(delay_percentile / 100)
baseline_delay_pct = 100 * (df["Shipping Lead Time"] > delay_threshold_days).mean()
st.sidebar.caption(
    f"≈ {delay_threshold_days:.0f} days at P{delay_percentile} (relative scale). "
    f"Full-dataset baseline: {baseline_delay_pct:.1f}% flagged delayed."
)

min_n = st.sidebar.slider(
    "Minimum shipments per route/state", min_value=1, max_value=100, value=10,
    help="Groups below this size are labelled Low_Sample and excluded from "
         "efficiency scores, rankings, and best/worst call-outs."
)

# ------------------------------------------------------------
# Apply filters
# ------------------------------------------------------------
filtered_df = df.copy()

n_missing_dates = int(df["Order Date"].isna().sum())
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    if (date_range[0], date_range[1]) != (full_start, full_end):
        # end date is inclusive of the whole day
        filtered_df = filtered_df[
            (filtered_df["Order Date"] >= start_date)
            & (filtered_df["Order Date"] < end_date + pd.Timedelta(days=1))
        ]
elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
    filtered_df = filtered_df[filtered_df["Order Date"].dt.normalize() == pd.to_datetime(date_range[0])]

if regions:
    filtered_df = filtered_df[filtered_df["Region"].isin(regions)]
if states:
    filtered_df = filtered_df[filtered_df["State/Province"].isin(states)]
if ship_modes:
    filtered_df = filtered_df[filtered_df["Ship Mode"].isin(ship_modes)]

if filtered_df.empty:
    st.warning("No shipments match the current filters. Adjust filters in the sidebar.")
    st.stop()

filtered_df = filtered_df.copy()
filtered_df["Is_Delayed"] = filtered_df["Shipping Lead Time"] > delay_threshold_days

if n_missing_dates:
    st.sidebar.caption(
        f"ℹ️ {n_missing_dates:,} shipments have no valid Order Date; they are "
        "excluded whenever a custom date range is chosen."
    )

filtered_route_summary = summarize(filtered_df, "Route", min_n=min_n)
filtered_route_region_summary = summarize(filtered_df, "Route_Region", min_n=min_n)
filtered_state_summary = summarize(filtered_df, "State/Province",
                                   count_label="Shipment_Volume", min_n=min_n)
filtered_state_summary["State_Abbrev"] = (
    filtered_state_summary["State/Province"].astype(str).str.strip().str.title()
    .map(US_STATE_ABBREV)
)

# ------------------------------------------------------------
# KPI cards
# ------------------------------------------------------------
st.markdown("### 📊 Key Performance Indicators")
c1, c2, c3, c4 = st.columns(4)
delayed_pct = 100 * filtered_df["Is_Delayed"].mean()
c1.metric("Total Shipments", f"{len(filtered_df):,}")
c2.metric("Avg Lead Time (relative days)", f"{filtered_df['Shipping Lead Time'].mean():.1f}")
c3.metric(f"Delayed % (> P{delay_percentile})", f"{delayed_pct:.1f}%",
          delta=f"{delayed_pct - baseline_delay_pct:+.1f} pp vs dataset baseline",
          delta_color="off")
n_routes = filtered_df["Route"].nunique()
n_routes_ok = int((~filtered_route_summary["Low_Sample"]).sum())
c4.metric("Unique Routes in View", f"{n_routes}",
          delta=f"{n_routes_ok} with ≥ {min_n} shipments", delta_color="off")

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📋 KPI Overview", "🚀 Route Efficiency", "🗺️ Geographic Map",
    "🚢 Ship Mode Comparison", "🔍 Route Drill-Down",
])

# ---------------- Tab 0 — KPI Overview ----------------
with tab0:
    st.subheader("Consolidated KPI Table")
    st.warning(
        "This table is the pipeline's master export. It does **not** respond to the "
        "sidebar filters, the delay cutoff, or the minimum-shipments setting, so its "
        "delay figures may differ from the live KPI cards above."
    )
    level_choice = st.radio("Level", sorted(kpi_summary["Level"].unique()),
                            horizontal=True, key="kpi_level")
    level_df = kpi_summary[kpi_summary["Level"] == level_choice]
    if "Route_Efficiency_Score" in level_df.columns:
        level_df = level_df.sort_values("Route_Efficiency_Score", ascending=False)
    st.dataframe(level_df, **STRETCH)

# ---------------- Tab 1 — Route Efficiency ----------------
with tab1:
    granularity = st.radio("Route granularity", ["Factory → State", "Factory → Region"],
                           horizontal=True)
    if granularity == "Factory → Region":
        active_summary, label_col = filtered_route_region_summary, "Route_Region"
    else:
        active_summary, label_col = filtered_route_summary, "Route"

    st.subheader(f"Routes by Efficiency ({granularity}) — reflects active filters")
    st.caption(f"Routes with fewer than {min_n} shipments are flagged Low_Sample, have no "
               "score, and sort to the bottom. Scores are scaled only among routes "
               "meeting the minimum, within the current view.")
    st.dataframe(active_summary.sort_values("Efficiency_Score", ascending=False,
                                            na_position="last"), **STRETCH)

    ok = eligible_rows(active_summary)
    if ok.empty:
        st.info(f"No routes in this view have at least {min_n} shipments. "
                "Lower the minimum or widen the filters.")
    else:
        top_n = ok.sort_values("Efficiency_Score", ascending=False).head(15)
        fig1 = px.bar(top_n, x="Efficiency_Score", y=label_col, orientation="h",
                      color="Efficiency_Score", color_continuous_scale="Greens",
                      title=f"Top 15 Most Efficient Routes ({granularity})")
        st.plotly_chart(fig1, **STRETCH)
        best = ok.loc[ok["Mean_Lead_Time"].idxmin()]
        worst = ok.loc[ok["Mean_Lead_Time"].idxmax()]
        st.markdown(
            f"**Observation — Good:** {best[label_col]} has the lowest average lead time "
            f"({best['Mean_Lead_Time']:.1f}; {int(best['Shipment_Count']):,} shipments). "
            f"**Needs attention:** {worst[label_col]} has the highest "
            f"({worst['Mean_Lead_Time']:.1f}; {int(worst['Shipment_Count']):,} shipments). "
            f"**Consider:** only routes with ≥ {min_n} shipments are compared; lead times "
            "are relative, not literal transit days."
        )

# ---------------- Tab 2 — Geographic Map ----------------
with tab2:
    st.subheader("US Shipping Lead Time Map (reflects active filters)")
    st.caption("Shows destination states only; it aggregates across all factories.")
    map_data = eligible_rows(filtered_state_summary)
    map_data = map_data[map_data["State_Abbrev"].notna()]
    if map_data.empty:
        st.info(f"No US states in this view have at least {min_n} shipments.")
    else:
        fig2 = px.choropleth(
            map_data, locations="State_Abbrev", locationmode="USA-states",
            color="Mean_Lead_Time", color_continuous_scale="RdYlGn_r", scope="usa",
            hover_name="State/Province",
            hover_data=["Shipment_Volume", "Mean_Lead_Time", "Delay_Frequency_Pct"],
        )
        st.plotly_chart(fig2, **STRETCH)
        fast = map_data.loc[map_data["Mean_Lead_Time"].idxmin()]
        slow = map_data.loc[map_data["Mean_Lead_Time"].idxmax()]
        st.markdown(
            f"**Observation — Good:** {fast['State/Province']} has the lowest average lead "
            f"time ({fast['Mean_Lead_Time']:.1f}; {int(fast['Shipment_Volume']):,} shipments). "
            f"**Needs attention:** {slow['State/Province']} has the highest "
            f"({slow['Mean_Lead_Time']:.1f}; {int(slow['Shipment_Volume']):,} shipments). "
            "Green = lower, red = higher. States below the minimum sample are hidden."
        )

    hidden = filtered_state_summary[filtered_state_summary["Low_Sample"]
                                    & filtered_state_summary["State_Abbrev"].notna()]
    if not hidden.empty:
        st.caption(f"{len(hidden)} US state(s) hidden for having fewer than {min_n} shipments.")
    non_us = filtered_state_summary[filtered_state_summary["State_Abbrev"].isna()]
    if not non_us.empty:
        st.caption("Non-US provinces in current filter (not shown on map): "
                   f"{', '.join(non_us['State/Province'].astype(str).tolist())}")

# ---------------- Tab 3 — Ship Mode Comparison ----------------
with tab3:
    st.subheader("Ship Mode Performance (reflects active filters)")
    mode_summary = filtered_df.groupby("Ship Mode").agg(
        Shipment_Count=("Shipping Lead Time", "count"),
        Mean_Lead_Time=("Shipping Lead Time", "mean"),
        Delay_Frequency_Pct=("Is_Delayed", lambda x: 100 * x.mean()),
        Avg_Cost=("Cost", "mean"),
        Avg_Sales=("Sales", "mean"),
    ).reset_index()
    st.dataframe(mode_summary, **STRETCH)
    fig3 = px.bar(mode_summary, x="Ship Mode", y="Mean_Lead_Time",
                  color="Delay_Frequency_Pct", color_continuous_scale="RdYlGn_r",
                  title="Avg Lead Time vs Delay Frequency")
    st.plotly_chart(fig3, **STRETCH)
    fastest = mode_summary.loc[mode_summary["Mean_Lead_Time"].idxmin()]
    low_d = mode_summary.loc[mode_summary["Delay_Frequency_Pct"].idxmin()]
    high_d = mode_summary.loc[mode_summary["Delay_Frequency_Pct"].idxmax()]
    st.markdown(
        f"**Observation — Good:** {fastest['Ship Mode']} has the lowest average lead time "
        f"({fastest['Mean_Lead_Time']:.1f}); {low_d['Ship Mode']} has the lowest delay "
        f"frequency ({low_d['Delay_Frequency_Pct']:.1f}%). **Needs attention:** "
        f"{high_d['Ship Mode']} has the highest ({high_d['Delay_Frequency_Pct']:.1f}%). "
        f"**Consider:** the dataset baseline is {baseline_delay_pct:.1f}%; check shipment "
        "counts before choosing a mode."
    )

# ---------------- Tab 4 — Route Drill-Down ----------------
with tab4:
    st.subheader("Detailed Route Analysis")
    available_states = sorted(filtered_df["State/Province"].dropna().unique())
    if not available_states:
        st.info("No states available in the current filter selection.")
    else:
        dc1, dc2 = st.columns(2)
        state_choice = dc1.selectbox("Select State", available_states)
        state_df = filtered_df[filtered_df["State/Province"] == state_choice]

        factories = sorted(state_df["Factory"].dropna().unique())
        factory_choice = dc2.selectbox(
            "Factory", ["All factories"] + factories,
            help="Pick one factory to keep a changing factory mix from distorting the trend."
        )
        if factory_choice != "All factories":
            state_df = state_df[state_df["Factory"] == factory_choice]

        # Exact state match on the State/Province column (no string parsing).
        state_routes = summarize(state_df, "Route", min_n=min_n)
        st.dataframe(state_routes, **STRETCH)

        scatter_df = state_df.dropna(subset=["Order Date", "Shipping Lead Time"]).copy()
        fig4 = px.scatter(scatter_df, x="Order Date", y="Shipping Lead Time",
                          color="Is_Delayed", hover_data=["Route", "Product Name"],
                          title=f"Lead Time Timeline — {state_choice} ({factory_choice})")

        n_pts = len(scatter_df)
        x_days = ((scatter_df["Order Date"] - scatter_df["Order Date"].min())
                  .dt.total_seconds() / 86400).to_numpy(dtype=float)
        y_vals = scatter_df["Shipping Lead Time"].to_numpy(dtype=float)
        can_fit = n_pts >= MIN_TREND_N and np.unique(x_days).size >= 2

        if can_fit:
            slope, intercept = np.polyfit(x_days, y_vals, 1)
            fitted = slope * x_days + intercept
            ss_tot = np.sum((y_vals - y_vals.mean()) ** 2)
            ss_res = np.sum((y_vals - fitted) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            sxx = np.sum((x_days - x_days.mean()) ** 2)
            se = np.sqrt(ss_res / (n_pts - 2) / sxx) if sxx > 0 else np.nan
            order = np.argsort(x_days)
            fig4.add_scatter(
                x=scatter_df["Order Date"].iloc[order], y=fitted[order], mode="lines",
                name="Linear trend", line={"color": "#111827", "width": 3, "dash": "dash"},
                hovertemplate="Linear trend: %{y:.1f}<extra></extra>",
            )
        st.plotly_chart(fig4, **STRETCH)

        if can_fit:
            per30 = slope * 30
            ci30 = 1.96 * se * 30 if np.isfinite(se) else np.nan
            direction = "upward" if slope > 0 else "downward" if slope < 0 else "flat"
            ci_text = (f" (≈95% interval {per30 - ci30:+.1f} to {per30 + ci30:+.1f})"
                       if np.isfinite(ci30) else "")
            r2_text = f"R² ≈ {r2:.2f}" if np.isfinite(r2) else "R² undefined (no variation)"
            st.markdown(
                f"**Trend:** {direction}, {per30:+.1f} relative days per 30 days of "
                f"order date{ci_text}; {r2_text}; n = {n_pts:,}."
            )
            if -1.2 <= slope <= -0.8:
                st.warning(
                    "The slope is close to −1 day per day. That is exactly what you'd see "
                    "if Ship Dates were unrelated to Order Dates (lead time = Ship Date − "
                    "Order Date), so this trend is most likely an artifact of the "
                    "mismatched date ranges, not a real change in performance."
                )
            if factory_choice == "All factories" and len(factories) > 1:
                st.caption("Multiple factories are pooled here; a shifting factory mix over "
                           "time can create or hide a trend. Select one factory to check.")
            st.caption("Descriptive association only — not evidence that order date "
                       "causes lead-time changes.")
        else:
            st.info(f"A trend line needs at least {MIN_TREND_N} shipments with at least two "
                    f"distinct order dates (this selection has {n_pts:,}). Widen the date "
                    "range, clear the factory filter, or pick a state with more shipments.")
