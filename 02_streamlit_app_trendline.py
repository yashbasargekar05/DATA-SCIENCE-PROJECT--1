# ============================================================
# Streamlit App — Nassau Candy Shipping Route Efficiency
# CORRECTED VERSION
# ============================================================
# Fixes applied vs. the original app:
#   1. Delay slider now uses a PERCENTILE cutoff (50-95) instead of a
#      fixed 1-30 day range — the old slider was a no-op because every
#      lead time in this dataset is 900+ days.
#   2. Route/State summaries are RECOMPUTED from the filtered rows
#      instead of just being filtered by name membership — previously,
#      Tabs 1 & 2 silently ignored the user's filters.
#   3. Route Drill-Down now matches the exact "-> State" suffix instead
#      of substring search, fixing "Virginia" incorrectly also matching
#      "West Virginia".
#   4. File names match the corrected pipeline's exports.
#   5. Added a Factory->Region view and a KPI Overview tab so
#      kpi_summary.csv (previously loaded but unused) is actually used.
#   6. Replaced all use_container_width=True with width="stretch" —
#      use_container_width is deprecated in current Streamlit and is
#      scheduled for eventual removal; width="stretch" is the
#      forward-compatible replacement recommended by Streamlit's docs.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="🍬 Nassau Candy Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("📦 Nassau Candy Shipping Route Efficiency Dashboard")
st.markdown("""
Use the filters on the left to explore **routes, states, ship modes, and delays**.

⚠️ **Data note:** Order Date and Ship Date in the source data come from non-overlapping
ranges, so raw "lead time in days" is not literal shipping time. All delay/efficiency
metrics here are computed **relative to this dataset's own distribution** (percentile-based),
and should be read as comparative rankings, not literal day counts.
""")

# ------------------------------------------------------------
# Load data (cached)
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_shipments.csv")
    kpi_summary = pd.read_csv("kpi_summary.csv")

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    return df, kpi_summary

df, kpi_summary = load_data()

# ------------------------------------------------------------
# Aggregation helpers — always computed FRESH from whatever
# dataframe (filtered or not) is passed in, so every tab stays
# consistent with the active filters. (Fixes bug #2.)
# ------------------------------------------------------------
def summarize(frame, group_col, count_label="Shipment_Count"):
    if frame.empty:
        return pd.DataFrame(columns=[group_col, count_label, "Mean_Lead_Time",
                                      "Median_Lead_Time", "Delay_Frequency_Pct",
                                      "Efficiency_Score"])
    summary = frame.groupby(group_col).agg(
        **{count_label: ("Row ID", "count")},
        Mean_Lead_Time=("Shipping Lead Time", "mean"),
        Median_Lead_Time=("Shipping Lead Time", "median"),
        Delay_Frequency_Pct=("Is_Delayed", lambda x: 100 * x.mean()),
    ).reset_index()
    min_lt, max_lt = summary["Mean_Lead_Time"].min(), summary["Mean_Lead_Time"].max()
    if max_lt == min_lt:
        summary["Efficiency_Score"] = 100.0
    else:
        summary["Efficiency_Score"] = 100 * (1 - (summary["Mean_Lead_Time"] - min_lt) / (max_lt - min_lt))
    return summary

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

# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.header("🔎 Filters")

date_range = st.sidebar.date_input(
    "Order Date Range",
    [df["Order Date"].min().date(), df["Order Date"].max().date()]
)
regions = st.sidebar.multiselect("Region", sorted(df["Region"].dropna().unique()))
states = st.sidebar.multiselect("State/Province", sorted(df["State/Province"].dropna().unique()))
ship_modes = st.sidebar.multiselect("Ship Mode", sorted(df["Ship Mode"].dropna().unique()))

# FIX #1: percentile-based delay cutoff instead of a fixed 1-30 day
# range that could never trigger against this dataset's ~900-1600 day
# lead times. The threshold is computed on the FULL (unfiltered)
# dataset so its meaning doesn't drift as the user changes filters,
# then applied consistently to whatever subset is in view.
delay_percentile = st.sidebar.slider(
    "Delay cutoff (percentile)", min_value=50, max_value=95, value=75, step=5,
    help="Shipments slower than this percentile of the full dataset's lead-time "
         "distribution are counted as 'delayed'. Higher = stricter definition of on-time."
)
delay_threshold_days = df["Shipping Lead Time"].quantile(delay_percentile / 100)
st.sidebar.caption(f"≈ {delay_threshold_days:.0f} days at P{delay_percentile} (relative scale — see data note above)")

# ------------------------------------------------------------
# Apply filters
# ------------------------------------------------------------
filtered_df = df.copy()

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = filtered_df.loc[
        (filtered_df["Order Date"] >= start_date) & (filtered_df["Order Date"] <= end_date)
    ]
elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
    filtered_df = filtered_df.loc[filtered_df["Order Date"] == pd.to_datetime(date_range[0])]

if regions:
    filtered_df = filtered_df[filtered_df["Region"].isin(regions)]
if states:
    filtered_df = filtered_df[filtered_df["State/Province"].isin(states)]
if ship_modes:
    filtered_df = filtered_df[filtered_df["Ship Mode"].isin(ship_modes)]

filtered_df = filtered_df.copy()
filtered_df["Is_Delayed"] = filtered_df["Shipping Lead Time"] > delay_threshold_days

# FIX #2: recompute summaries from the filtered rows, every time.
filtered_route_summary = summarize(filtered_df, "Route")
filtered_route_region_summary = summarize(filtered_df, "Route_Region")
filtered_state_summary = summarize(filtered_df, "State/Province", count_label="Shipment_Volume")
filtered_state_summary["State_Abbrev"] = filtered_state_summary["State/Province"].map(US_STATE_ABBREV)

if filtered_df.empty:
    st.warning("No shipments match the current filters. Adjust filters in the sidebar.")
    st.stop()

# ------------------------------------------------------------
# KPI Cards
# ------------------------------------------------------------
st.markdown("### 📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Shipments", f"{len(filtered_df):,}")
col2.metric("Avg Lead Time (relative days)", f"{filtered_df['Shipping Lead Time'].mean():.1f}")
col3.metric(f"Delayed % (> P{delay_percentile})", f"{100 * filtered_df['Is_Delayed'].mean():.1f}%")
col4.metric("Unique Routes in View", f"{filtered_df['Route'].nunique()}")

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📋 KPI Overview",
    "🚀 Route Efficiency",
    "🗺️ Geographic Map",
    "🚢 Ship Mode Comparison",
    "🔍 Route Drill-Down",
])

# ------------------------------------------------------------
# Tab 0 — KPI Overview (uses kpi_summary.csv — was previously unused)
# ------------------------------------------------------------
with tab0:
    st.subheader("Consolidated KPI Table")
    level_choice = st.radio(
        "Level", sorted(kpi_summary["Level"].unique()), horizontal=True, key="kpi_level"
    )
    st.dataframe(
        kpi_summary[kpi_summary["Level"] == level_choice].sort_values(
            "Route_Efficiency_Score", ascending=False
        ),
        width="stretch",
    )
    st.caption("This table reflects the full dataset (not the sidebar filters) — "
               "it's the master reference export from the analysis pipeline.")

# ------------------------------------------------------------
# Tab 1 — Route Efficiency (State-level and Region-level, both FILTERED)
# ------------------------------------------------------------
with tab1:
    granularity = st.radio("Route granularity", ["Factory → State", "Factory → Region"], horizontal=True)
    active_summary = filtered_route_region_summary if granularity == "Factory → Region" else filtered_route_summary
    label_col = "Route_Region" if granularity == "Factory → Region" else "Route"

    st.subheader(f"Routes by Efficiency ({granularity}) — reflects active filters")
    st.dataframe(active_summary.sort_values("Efficiency_Score", ascending=False), width="stretch")

    top_n = active_summary.sort_values("Efficiency_Score", ascending=False).head(15)
    fig1 = px.bar(top_n, x="Efficiency_Score", y=label_col, orientation="h",
                  color="Efficiency_Score", color_continuous_scale="Greens",
                  title=f"Top 15 Most Efficient Routes ({granularity})")
    st.plotly_chart(fig1, width="stretch")
    if not active_summary.empty:
        best_route = active_summary.loc[active_summary["Mean_Lead_Time"].idxmin()]
        slowest_route = active_summary.loc[active_summary["Mean_Lead_Time"].idxmax()]
        st.markdown(
            f"**Observation / interpretation — Good:** {best_route[label_col]} has the "
            f"lowest average lead time ({best_route['Mean_Lead_Time']:.1f}; "
            f"{best_route['Shipment_Count']:,} shipments). **Needs attention:** "
            f"{slowest_route[label_col]} has the highest ({slowest_route['Mean_Lead_Time']:.1f}; "
            f"{slowest_route['Shipment_Count']:,} shipments). **Consider:** compare shipment "
            "counts before acting on route extremes; efficiency scores are relative to the "
            "routes in the current filtered view, and the date-quality note above means "
            "lead-time values are not literal transit days."
        )

# ------------------------------------------------------------
# Tab 2 — Geographic Map (recomputed from filtered_df, US-only)
# ------------------------------------------------------------
with tab2:
    st.subheader("US Shipping Lead Time Map (reflects active filters)")
    map_data = filtered_state_summary[filtered_state_summary["State_Abbrev"].notna()]
    if map_data.empty:
        st.info("No US states in the current filter selection.")
    else:
        fig2 = px.choropleth(
            map_data,
            locations="State_Abbrev",
            locationmode="USA-states",
            color="Mean_Lead_Time",
            color_continuous_scale="RdYlGn_r",
            scope="usa",
            hover_name="State/Province",
            hover_data=["Shipment_Volume", "Mean_Lead_Time", "Delay_Frequency_Pct"],
        )
        st.plotly_chart(fig2, width="stretch")
        fastest_state = map_data.loc[map_data["Mean_Lead_Time"].idxmin()]
        slowest_state = map_data.loc[map_data["Mean_Lead_Time"].idxmax()]
        st.markdown(
            f"**Observation / interpretation — Good:** {fastest_state['State/Province']} "
            f"has the lowest average lead time ({fastest_state['Mean_Lead_Time']:.1f}; "
            f"{fastest_state['Shipment_Volume']:,} shipments). **Needs attention:** "
            f"{slowest_state['State/Province']} has the highest "
            f"({slowest_state['Mean_Lead_Time']:.1f}; "
            f"{slowest_state['Shipment_Volume']:,} shipments). On this map, green indicates "
            "lower and red higher averages. **Consider:** these are filtered, relative "
            "comparisons; smaller shipment counts can make a state's average less reliable."
        )

    non_us = filtered_state_summary[filtered_state_summary["State_Abbrev"].isna()]
    if not non_us.empty:
        st.caption(f"Non-US provinces in current filter (not shown on map): "
                   f"{', '.join(non_us['State/Province'].tolist())}")

# ------------------------------------------------------------
# Tab 3 — Ship Mode Comparison (already correct, kept as-is + delay % fix)
# ------------------------------------------------------------
with tab3:
    st.subheader("Ship Mode Performance (reflects active filters)")
    mode_summary = filtered_df.groupby("Ship Mode").agg(
        Shipment_Count=("Shipping Lead Time", "count"),
        Mean_Lead_Time=("Shipping Lead Time", "mean"),
        Delay_Frequency_Pct=("Is_Delayed", lambda x: 100 * x.mean()),
        Avg_Cost=("Cost", "mean"),
        Avg_Sales=("Sales", "mean"),
    ).reset_index()
    st.dataframe(mode_summary, width="stretch")
    fig3 = px.bar(mode_summary, x="Ship Mode", y="Mean_Lead_Time",
                  color="Delay_Frequency_Pct", color_continuous_scale="RdYlGn_r",
                  title="Avg Lead Time vs Delay Frequency")
    st.plotly_chart(fig3, width="stretch")
    if not mode_summary.empty:
        fastest_mode = mode_summary.loc[mode_summary["Mean_Lead_Time"].idxmin()]
        lowest_delay_mode = mode_summary.loc[mode_summary["Delay_Frequency_Pct"].idxmin()]
        highest_delay_mode = mode_summary.loc[mode_summary["Delay_Frequency_Pct"].idxmax()]
        st.markdown(
            f"**Observation / interpretation — Good:** {fastest_mode['Ship Mode']} has "
            f"the lowest average lead time ({fastest_mode['Mean_Lead_Time']:.1f}); "
            f"{lowest_delay_mode['Ship Mode']} has the lowest delay frequency "
            f"({lowest_delay_mode['Delay_Frequency_Pct']:.1f}%). **Needs attention:** "
            f"{highest_delay_mode['Ship Mode']} has the highest delay frequency "
            f"({highest_delay_mode['Delay_Frequency_Pct']:.1f}%). **Consider:** the quickest "
            "average and lowest delay rate may belong to different modes; check shipment "
            "counts and the relative delay cutoff before choosing a mode."
        )

# ------------------------------------------------------------
# Tab 4 — Route Drill-Down (FIX #3: exact suffix match, not substring)
# ------------------------------------------------------------
with tab4:
    st.subheader("Detailed Route Analysis")
    available_states = sorted(filtered_df["State/Province"].dropna().unique())
    if not available_states:
        st.info("No states available in the current filter selection.")
    else:
        state_choice = st.selectbox("Select State", available_states)

        # FIX: exact match on the "-> State" suffix, so selecting "Virginia"
        # no longer also pulls in "West Virginia" via substring matching.
        exact_suffix = f"-> {state_choice}"
        state_routes = filtered_route_summary[
            filtered_route_summary["Route"].astype(str).str.endswith(exact_suffix)
        ]
        st.dataframe(state_routes, width="stretch")

        scatter_df = filtered_df[
            filtered_df["State/Province"] == state_choice
        ].dropna(subset=["Order Date", "Shipping Lead Time"]).copy()
        fig4 = px.scatter(
            scatter_df,
            x="Order Date", y="Shipping Lead Time",
            color="Is_Delayed", hover_data=["Route", "Product Name"],
            title=f"Lead Time Timeline for {state_choice}"
        )
        # Fit a simple OLS line directly with NumPy to avoid adding a
        # statsmodels dependency just for Plotly's trendline option.
        x_days = (
            (scatter_df["Order Date"] - scatter_df["Order Date"].min())
            .dt.total_seconds() / 86400
        ).to_numpy(dtype=float)
        y_values = scatter_df["Shipping Lead Time"].to_numpy(dtype=float)
        if len(scatter_df) >= 2 and np.unique(x_days).size >= 2:
            slope, intercept = np.polyfit(x_days, y_values, 1)
            fitted_values = slope * x_days + intercept
            total_variation = np.sum((y_values - y_values.mean()) ** 2)
            r_squared = (
                1 - np.sum((y_values - fitted_values) ** 2) / total_variation
                if total_variation > 0 else np.nan
            )
            date_order = np.argsort(x_days)
            fig4.add_scatter(
                x=scatter_df["Order Date"].iloc[date_order],
                y=fitted_values[date_order],
                mode="lines",
                name="Linear trend",
                line={"color": "#111827", "width": 3, "dash": "dash"},
                hovertemplate="Linear trend: %{y:.1f}<extra></extra>",
            )
        st.plotly_chart(fig4, width="stretch")
        if len(scatter_df) >= 2 and np.unique(x_days).size >= 2:
            trend_direction = "upward" if slope > 0 else "downward" if slope < 0 else "flat"
            r_squared_text = (
                f"R² = {r_squared:.3f} (the line accounts for about "
                f"{100 * r_squared:.1f}% of the variation)"
                if np.isfinite(r_squared)
                else "R² is undefined because lead time does not vary"
            )
            st.markdown(
                f"**Observation / interpretation — Trend:** the dashed fitted line is "
                f"{trend_direction}; {r_squared_text}. **Consider:** treat this as a "
                "descriptive association, not proof that order date causes lead-time "
                "changes. The source date ranges do not align, and small samples can make "
                "the fitted trend unstable."
            )
        else:
            st.markdown(
                "**Observation / interpretation:** there are not at least two distinct "
                "order dates in this selection, so a trend line cannot be estimated. "
                "**Consider:** widen the date range or select a state with more dated "
                "shipments; the source date-quality note still applies."
            )
