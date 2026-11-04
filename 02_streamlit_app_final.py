# Nassau Candy — Shipping Route Efficiency Dashboard (FINAL)
# ------------------------------------------------------------
# Expects in the working directory:  cleaned_shipments.csv, kpi_summary.csv, shipping_logic.py
# Requires: streamlit>=1.30, pandas, numpy, plotly   (tests: pytest test_shipping_logic.py)
#
# Changes vs. 02_streamlit_app_fixed.py (senior data-science review):
#   1. Data-quality banner at the TOP, built from live diagnostics (clusters, order-date R², effect sizes).
#   2. Order-year control: lead times can be compared WITHIN order year (default on); the delay flag can use
#      per-year percentiles so "delayed" is not just a proxy for "ordered later".
#   3. Uncertainty everywhere: 95% CI, score ± margin, and a "distinct from overall" flag per route.
#   4. Ship Mode tab reports effect size (η²) and refuses to rank modes when the effect is negligible.
#   5. Delay rule: percentile OR fixed SLA days, plus a delayed-% curve so the step behaviour is visible.
#   6. New Data Quality tab; all logic moved into shipping_logic.py with a pytest suite.
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import shipping_logic as sl
from shipping_logic import ADJ, LEAD, YEAR

st.set_page_config(page_title="🍬 Nassau Candy Dashboard", layout="wide", initial_sidebar_state="expanded")

REQUIRED_COLUMNS = ["Row ID", "Order Date", "Region", "State/Province", "Ship Mode", LEAD, "Route",
                    "Route_Region", "Product Name", "Cost", "Sales"]
NAVY, TEAL, RED, GREY = "#1F3A5F", "#2A9D8F", "#E76F51", "#8D99AE"


def _st_version():
    try:
        return tuple(int(p) for p in st.__version__.split(".")[:2])
    except ValueError:
        return (0, 0)


STRETCH = {"width": "stretch"} if _st_version() >= (1, 50) else {"use_container_width": True}

US_STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
    'Connecticut': 'CT', 'Delaware': 'DE', 'District Of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
    'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
    'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
}


# ------------------------------------------------------------
# Load + validate (cached)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
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
    data[LEAD] = pd.to_numeric(data[LEAD], errors="coerce")
    if "Ship Date" in data.columns:
        data["Ship Date"] = pd.to_datetime(data["Ship Date"], errors="coerce", dayfirst=True)
    if "Factory" not in data.columns:
        data["Factory"] = data["Route"].astype(str).str.split("->").str[0].str.strip()
    return sl.add_order_year_and_adjustment(data), kpis


try:
    df, kpi_summary = load_data()
except FileNotFoundError as e:
    st.error(f"Required data file not found: {e}. Run the analysis pipeline first and launch the app "
             "from the folder containing its CSV exports.")
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

if df["Order Date"].notna().sum() == 0:
    st.error("No valid Order Dates in cleaned_shipments.csv.")
    st.stop()

st.title("📦 Nassau Candy Shipping Route Efficiency Dashboard")

# ------------------------------------------------------------
# Data-quality banner (from live diagnostics on the FULL data)
# ------------------------------------------------------------
diag_full = sl.diagnostics(df)
if not diag_full["plausible"]:
    eta = diag_full["eta2"]
    st.error(
        f"**Data-quality alert — lead times are not literal shipping times.** Median lead time is "
        f"**{diag_full['median']:,.0f} days** (plausible transit would be under {sl.PLAUSIBLE_MAX_LEAD_DAYS}). "
        f"Lead times fall into **{len(diag_full['clusters'])} separate clusters**, order date alone explains "
        f"**{100 * diag_full['r2_date']:.0f}%** of their variance, and Ship Mode explains only "
        f"**{100 * eta.get('Ship Mode', float('nan')):.1f}%**. Treat every figure below as a *relative, exploratory* "
        f"comparison, not as delivery performance. See the **Data Quality** tab."
    )
else:
    st.success(f"Lead times look plausible (median {diag_full['median']:.1f} days). Efficiency figures are still "
               "relative to this dataset's own distribution.")

# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.header("🔎 Filters")
full_start, full_end = df["Order Date"].min().date(), df["Order Date"].max().date()
date_range = st.sidebar.date_input("Order Date Range", [full_start, full_end])
years_avail = sorted(df[YEAR].dropna().astype(int).unique())
years_sel = st.sidebar.multiselect("Order Year", years_avail)
regions = st.sidebar.multiselect("Region", sorted(df["Region"].dropna().unique()))
states = st.sidebar.multiselect("State/Province", sorted(df["State/Province"].dropna().unique()))
ship_modes = st.sidebar.multiselect("Ship Mode", sorted(df["Ship Mode"].dropna().unique()))

st.sidebar.header("⚙️ Method")
within_year = st.sidebar.checkbox(
    "Compare within order year (recommended)", value=True,
    help="Removes each order year's average lead time before scoring routes/modes, and uses per-year percentiles "
         "for the delay flag. Without this, a route's score mostly reflects WHEN its orders were placed.")
delay_mode = st.sidebar.radio("Delay rule", ["Percentile", "Fixed SLA (days)"], horizontal=True)
delay_percentile, sla_days = 75, None
if delay_mode == "Percentile":
    delay_percentile = st.sidebar.slider("Delay cutoff (percentile)", 50, 95, 75, 5,
                                         help="Shipments slower than this percentile are 'delayed'. By construction a "
                                              "share of shipments is always flagged — compare with the baseline shown.")
else:
    sla_days = st.sidebar.number_input("SLA threshold (days)", min_value=1.0, value=float(round(df[LEAD].quantile(0.75))),
                                       step=1.0, help="Shipments with lead time above this are 'delayed'.")
min_n = st.sidebar.slider("Minimum shipments per route/state", 1, 100, 10,
                          help="Smaller groups are flagged Low_Sample and excluded from scores, rankings and call-outs.")

mode_key = "sla" if delay_mode.startswith("Fixed") else "percentile"
full_flag, _thr = sl.delay_flag(df, df, mode_key, delay_percentile, sla_days, by_year=within_year)
baseline_delay_pct = 100 * full_flag.mean()
rule_text = (f"> {sla_days:,.0f} days (SLA)" if mode_key == "sla"
             else f"> P{delay_percentile}{' within each order year' if within_year else ' of all shipments'}")
st.sidebar.caption(f"Delay rule: {rule_text}. Full-dataset baseline: {baseline_delay_pct:.1f}% flagged.")
SCORE_COL = ADJ if within_year else LEAD

# ------------------------------------------------------------
# Apply filters
# ------------------------------------------------------------
filtered_df = df.copy()
n_missing_dates = int(df["Order Date"].isna().sum())
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    s_d, e_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    if (date_range[0], date_range[1]) != (full_start, full_end):
        filtered_df = filtered_df[(filtered_df["Order Date"] >= s_d) & (filtered_df["Order Date"] < e_d + pd.Timedelta(days=1))]
elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
    filtered_df = filtered_df[filtered_df["Order Date"].dt.normalize() == pd.to_datetime(date_range[0])]
if years_sel:
    filtered_df = filtered_df[filtered_df[YEAR].isin(years_sel)]
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
filtered_df["Is_Delayed"], _ = sl.delay_flag(df, filtered_df, mode_key, delay_percentile, sla_days, by_year=within_year)
if n_missing_dates:
    st.sidebar.caption(f"ℹ️ {n_missing_dates:,} shipments have no valid Order Date; they are excluded when a custom "
                       "date range is chosen and are left unadjusted for order year.")

route_sum = sl.summarize(filtered_df, "Route", min_n=min_n, score_col=SCORE_COL)
route_region_sum = sl.summarize(filtered_df, "Route_Region", min_n=min_n, score_col=SCORE_COL)
state_sum = sl.summarize(filtered_df, "State/Province", count_label="Shipment_Volume", min_n=min_n, score_col=SCORE_COL)
state_sum["State_Abbrev"] = state_sum["State/Province"].astype(str).str.strip().str.title().map(US_STATE_ABBREV)

# ------------------------------------------------------------
# KPI cards
# ------------------------------------------------------------
st.markdown("### 📊 Key Performance Indicators")
c1, c2, c3, c4 = st.columns(4)
delayed_pct = 100 * filtered_df["Is_Delayed"].mean()
c1.metric("Total Shipments", f"{len(filtered_df):,}")
c2.metric("Avg Lead Time (relative days)", f"{filtered_df[LEAD].mean():.1f}")
c3.metric(f"Delayed % ({rule_text})", f"{delayed_pct:.1f}%",
          delta=f"{delayed_pct - baseline_delay_pct:+.1f} pp vs dataset baseline", delta_color="off")
n_routes_ok = int((~route_sum["Low_Sample"]).sum())
c4.metric("Unique Routes in View", f"{filtered_df['Route'].nunique()}", delta=f"{n_routes_ok} with ≥ {min_n} shipments",
          delta_color="off")

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 KPI Overview", "🚀 Route Efficiency", "🗺️ Geographic Map", "🚢 Ship Mode Comparison",
    "🔍 Route Drill-Down", "🧪 Data Quality"])

# ---------------- Tab 0 ----------------
with tab0:
    st.subheader("Consolidated KPI Table")
    st.warning("This table is the pipeline's master export. It does **not** respond to the sidebar filters, the delay "
               "rule or the minimum-shipments setting, so its figures may differ from the live KPI cards above.")
    level_choice = st.radio("Level", sorted(kpi_summary["Level"].unique()), horizontal=True, key="kpi_level")
    level_df = kpi_summary[kpi_summary["Level"] == level_choice]
    if "Route_Efficiency_Score" in level_df.columns:
        level_df = level_df.sort_values("Route_Efficiency_Score", ascending=False)
    st.dataframe(level_df, **STRETCH)

# ---------------- Tab 1 ----------------
with tab1:
    granularity = st.radio("Route granularity", ["Factory → State", "Factory → Region"], horizontal=True)
    active, label_col = (route_region_sum, "Route_Region") if granularity == "Factory → Region" else (route_sum, "Route")
    basis = "order-year-adjusted lead time" if within_year else "raw lead time"
    st.subheader(f"Routes by Efficiency ({granularity}) — reflects active filters")
    st.caption(f"Scores use **{basis}**, scaled 0–100 only among routes with ≥ {min_n} shipments. Read each score as "
               "**score ± Score_Margin** (95% CI). *Distinct_From_Overall* = the route's mean differs from the overall "
               "mean by more than its own CI; routes that are not distinct should not be ranked against each other.")
    show_cols = [label_col, "Shipment_Count", "Mean_Lead_Time", "Mean_Adjusted_Lead", "Lead_CI95", "Median_Lead_Time",
                 "Delay_Frequency_Pct", "Efficiency_Score", "Score_Margin", "Distinct_From_Overall", "Low_Sample"]
    st.dataframe(active.sort_values("Efficiency_Score", ascending=False, na_position="last")[show_cols], **STRETCH)
    ok = sl.eligible_rows(active)
    if ok.empty:
        st.info(f"No routes in this view have at least {min_n} shipments. Lower the minimum or widen the filters.")
    else:
        top = ok.sort_values("Efficiency_Score", ascending=False).head(15)
        fig1 = px.bar(top, x="Efficiency_Score", y=label_col, orientation="h", error_x="Score_Margin",
                      color="Efficiency_Score", color_continuous_scale="Greens",
                      title=f"Top 15 Routes by Efficiency Score (bars show 95% CI) — {granularity}",
                      hover_data=["Shipment_Count", "Mean_Lead_Time", "Lead_CI95"])
        fig1.update_yaxes(autorange="reversed")
        st.plotly_chart(fig1, **STRETCH)
        best = ok.loc[ok["Efficiency_Score"].idxmax()]
        worst = ok.loc[ok["Efficiency_Score"].idxmin()]
        n_dist = int(ok["Distinct_From_Overall"].sum())
        overlap = (best["Efficiency_Score"] - (best["Score_Margin"] if pd.notna(best["Score_Margin"]) else 0)
                   <= worst["Efficiency_Score"] + (worst["Score_Margin"] if pd.notna(worst["Score_Margin"]) else 0))
        st.markdown(
            f"**Observation:** highest score **{best[label_col]}** ({best['Efficiency_Score']:.0f} ± "
            f"{0 if pd.isna(best['Score_Margin']) else best['Score_Margin']:.0f}; {int(best['Shipment_Count']):,} shipments); lowest "
            f"**{worst[label_col]}** ({worst['Efficiency_Score']:.0f} ± "
            f"{0 if pd.isna(worst['Score_Margin']) else worst['Score_Margin']:.0f}; {int(worst['Shipment_Count']):,} shipments). "
            f"Only **{n_dist} of {len(ok)}** eligible routes differ from the overall mean beyond their 95% CI. "
            + ("The best and worst intervals **overlap**, so the ranking is not reliable here. " if overlap else
               "The best and worst intervals do not overlap. ")
            + ("Scores are order-year-adjusted. " if within_year else
               "**Scores are NOT order-year-adjusted; routes with more recent orders will look worse.** ")
            + "Lead times are relative, not literal transit days."
        )

# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("US Shipping Lead Time Map (reflects active filters)")
    st.caption("Destination states only, aggregated across factories. Colour = mean of the scoring basis "
               f"({basis}); states below the minimum sample are hidden.")
    map_data = sl.eligible_rows(state_sum)
    map_data = map_data[map_data["State_Abbrev"].notna()]
    if map_data.empty:
        st.info(f"No US states in this view have at least {min_n} shipments.")
    else:
        map_data = map_data.assign(Mean_Basis=map_data["Mean_Adjusted_Lead"] if within_year else map_data["Mean_Lead_Time"])
        fig2 = px.choropleth(map_data, locations="State_Abbrev", locationmode="USA-states", color="Mean_Basis",
                             color_continuous_scale="RdYlGn_r", scope="usa", hover_name="State/Province",
                             hover_data={"Shipment_Volume": True, "Lead_CI95": ":.0f", "Delay_Frequency_Pct": ":.1f",
                                         "State_Abbrev": False, "Mean_Basis": ":.0f"})
        st.plotly_chart(fig2, **STRETCH)
        fast = map_data.loc[map_data["Mean_Basis"].idxmin()]
        slow = map_data.loc[map_data["Mean_Basis"].idxmax()]
        st.markdown(
            f"**Observation:** lowest {basis}: **{fast['State/Province']}** ({fast['Mean_Basis']:.0f} ± {fast['Lead_CI95']:.0f}; "
            f"{int(fast['Shipment_Volume']):,} shipments). Highest: **{slow['State/Province']}** ({slow['Mean_Basis']:.0f} ± "
            f"{slow['Lead_CI95']:.0f}; {int(slow['Shipment_Volume']):,}). "
            f"{int(map_data['Distinct_From_Overall'].sum())} of {len(map_data)} mapped states differ from the overall mean beyond "
            "their 95% CI. Green = lower, red = higher.")
    hidden = state_sum[state_sum["Low_Sample"] & state_sum["State_Abbrev"].notna()]
    if not hidden.empty:
        st.caption(f"{len(hidden)} US state(s) hidden for having fewer than {min_n} shipments.")
    non_us = state_sum[state_sum["State_Abbrev"].isna()]
    if not non_us.empty:
        st.caption("Non-US provinces in current filter (not shown on map): "
                   f"{', '.join(non_us['State/Province'].astype(str).tolist())}")

# ---------------- Tab 3 ----------------
with tab3:
    st.subheader("Ship Mode Performance (reflects active filters)")
    mode_sum = sl.summarize(filtered_df, "Ship Mode", min_n=1, score_col=SCORE_COL)
    extra = filtered_df.groupby("Ship Mode").agg(Avg_Cost=("Cost", "mean"), Avg_Sales=("Sales", "mean")).reset_index()
    mode_sum = mode_sum.merge(extra, on="Ship Mode")
    eta_mode = sl.eta_squared(filtered_df, "Ship Mode", SCORE_COL)
    st.dataframe(mode_sum[["Ship Mode", "Shipment_Count", "Mean_Lead_Time", "Mean_Adjusted_Lead", "Lead_CI95",
                           "Median_Lead_Time", "Delay_Frequency_Pct", "Avg_Cost", "Avg_Sales"]], **STRETCH)
    overall_mean = filtered_df[SCORE_COL].mean()
    mode_sum["Mean_Basis"] = filtered_df.groupby("Ship Mode")[SCORE_COL].mean().reindex(mode_sum["Ship Mode"]).to_numpy()
    fig3 = px.bar(mode_sum, x="Ship Mode", y="Mean_Basis", error_y="Lead_CI95", color="Delay_Frequency_Pct",
                  color_continuous_scale="RdYlGn_r", title=f"Mean lead time by ship mode ({basis}; bars = 95% CI)")
    fig3.add_hline(y=overall_mean, line_dash="dot", line_color=GREY, annotation_text="Overall mean")
    st.plotly_chart(fig3, **STRETCH)
    spread = mode_sum["Mean_Basis"].max() - mode_sum["Mean_Basis"].min()
    max_ci = mode_sum["Lead_CI95"].max()
    if np.isfinite(eta_mode) and eta_mode < sl.MEANINGFUL_ETA2:
        verdict = (f"Ship Mode explains only **{100 * eta_mode:.2f}%** of lead-time variation, so the modes are "
                   f"**practically indistinguishable** (largest gap between modes {spread:,.0f} days vs. a typical 95% CI of "
                   f"±{max_ci:,.0f}). **Do not rank or choose modes from this chart.**")
    else:
        verdict = (f"Ship Mode explains **{100 * eta_mode:.1f}%** of lead-time variation; the gap between best and worst mean "
                   f"is {spread:,.0f} days (typical 95% CI ±{max_ci:,.0f}).")
    st.markdown(f"**Observation:** {verdict} Dataset baseline delayed rate: {baseline_delay_pct:.1f}%.")
    if "Same Day" in set(mode_sum["Ship Mode"]) and "Standard Class" in set(mode_sum["Ship Mode"]):
        sd = mode_sum.loc[mode_sum["Ship Mode"] == "Same Day", "Mean_Basis"].iloc[0]
        stc = mode_sum.loc[mode_sum["Ship Mode"] == "Standard Class", "Mean_Basis"].iloc[0]
        if sd >= stc - max_ci:
            st.warning(f"Same Day (mean {sd:,.0f}) is not faster than Standard Class ({stc:,.0f}). That contradicts what the "
                       "service levels mean, which is further evidence that this lead-time field does not measure transit time.")

# ---------------- Tab 4 ----------------
with tab4:
    st.subheader("Detailed Route Analysis")
    available_states = sorted(filtered_df["State/Province"].dropna().unique())
    if not available_states:
        st.info("No states available in the current filter selection.")
    else:
        d1, d2, d3 = st.columns(3)
        state_choice = d1.selectbox("Select State", available_states)
        state_df = filtered_df[filtered_df["State/Province"] == state_choice]
        factories = sorted(state_df["Factory"].dropna().unique())
        factory_choice = d2.selectbox("Factory", ["All factories"] + factories,
                                      help="One factory keeps a changing factory mix from distorting the trend.")
        if factory_choice != "All factories":
            state_df = state_df[state_df["Factory"] == factory_choice]
        yrs = sorted(state_df[YEAR].dropna().astype(int).unique())
        year_choice = d3.selectbox("Order year", ["All years"] + [str(y) for y in yrs],
                                   help="A trend across years mixes in the year-to-year step; pick one year to see within-year drift.")
        if year_choice != "All years":
            state_df = state_df[state_df[YEAR] == int(year_choice)]

        st.dataframe(sl.summarize(state_df, "Route", min_n=min_n, score_col=SCORE_COL), **STRETCH)
        scatter_df = state_df.dropna(subset=["Order Date", LEAD]).copy()
        fig4 = px.scatter(scatter_df, x="Order Date", y=LEAD, color="Is_Delayed", hover_data=["Route", "Product Name"],
                          title=f"Lead Time Timeline — {state_choice} ({factory_choice}, {year_choice})")
        tf = sl.trend_fit(scatter_df["Order Date"], scatter_df[LEAD])
        if tf is not None:
            order = np.argsort(tf["x"])
            xs = scatter_df.loc[tf["index"], "Order Date"].to_numpy()[order]
            fig4.add_scatter(x=xs, y=tf["fitted"][order], mode="lines", name="Linear trend",
                             line={"color": "#111827", "width": 3, "dash": "dash"},
                             hovertemplate="Linear trend: %{y:.1f}<extra></extra>")
        st.plotly_chart(fig4, **STRETCH)
        if tf is not None:
            per30 = tf["slope"] * 30
            ci30 = 1.96 * tf["se"] * 30 if np.isfinite(tf["se"]) else np.nan
            direction = "upward" if tf["slope"] > 0 else "downward" if tf["slope"] < 0 else "flat"
            ci_text = f" (≈95% interval {per30 - ci30:+.1f} to {per30 + ci30:+.1f})" if np.isfinite(ci30) else ""
            r2_text = f"R² ≈ {tf['r2']:.2f}" if np.isfinite(tf["r2"]) else "R² undefined (no variation)"
            st.markdown(f"**Trend:** {direction}, {per30:+.1f} relative days per 30 days of order date{ci_text}; {r2_text}; "
                        f"n = {tf['n']:,}.")
            reasons = sl.artifact_reasons(tf["slope"], float(np.median(scatter_df[LEAD])))
            if reasons:
                st.warning("**Likely a data artifact, not a performance trend.** " + "; ".join(reasons).capitalize() +
                           ". Order Date and Ship Date come from mismatched ranges, so do not act on this trend.")
            if factory_choice == "All factories" and len(factories) > 1:
                st.caption("Multiple factories are pooled; a shifting factory mix over time can create or hide a trend.")
            if year_choice == "All years" and len(yrs) > 1:
                st.caption("Multiple order years are pooled, so the fitted slope includes the year-to-year step.")
            st.caption("Descriptive association only — not evidence that order date causes lead-time changes.")
        else:
            st.info(f"A trend line needs at least {sl.MIN_TREND_N} shipments with at least two distinct order dates "
                    f"(this selection has {len(scatter_df):,}). Widen the date range, clear the factory/year filter, or pick a "
                    "state with more shipments.")

# ---------------- Tab 5 ----------------
with tab5:
    st.subheader("Is the lead-time metric trustworthy?")
    dg = sl.diagnostics(filtered_df)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Median lead time (days)", f"{dg['median']:,.0f}")
    m2.metric("Lead-time clusters", f"{len(dg['clusters'])}", help=f"Groups separated by gaps > {sl.CLUSTER_GAP_DAYS} days")
    m3.metric("Variance explained by order date", f"{100 * dg['r2_date']:.0f}%" if np.isfinite(dg["r2_date"]) else "n/a")
    m4.metric("Variance explained by Ship Mode", f"{100 * dg['eta2'].get('Ship Mode', np.nan):.2f}%")
    e = dg["eta2"]
    st.markdown("**Variance in lead time explained by each grouping (η²; below "
                f"{100 * sl.MEANINGFUL_ETA2:.0f}% = no practical effect):** " +
                " · ".join(f"{k}: {100 * v:.2f}%" for k, v in e.items() if np.isfinite(v)))
    st.markdown("**Lead-time clusters**")
    st.dataframe(dg["clusters"], **STRETCH)
    fig5 = px.histogram(filtered_df.dropna(subset=[LEAD]), x=LEAD, nbins=80, color=filtered_df[YEAR].astype("Int64").astype(str),
                        title="Distribution of lead time (days) by order year", labels={"color": "Order year"})
    st.plotly_chart(fig5, **STRETCH)
    q = filtered_df.dropna(subset=["Order Date"]).groupby(filtered_df["Order Date"].dt.to_period("Q").astype(str))[LEAD] \
        .agg(["count", "mean"]).reset_index().rename(columns={"Order Date": "Quarter", "mean": "Mean_Lead_Time"})
    fig6 = px.bar(q, x="Quarter", y="Mean_Lead_Time", hover_data=["count"], title="Mean lead time by order quarter",
                  color_discrete_sequence=[NAVY])
    st.plotly_chart(fig6, **STRETCH)
    cv_raw, cv_yr = sl.delay_curve(df, False), sl.delay_curve(df, True)
    fig7 = go.Figure()
    fig7.add_scatter(x=cv_raw["Percentile"], y=cv_raw["Delayed_%"], mode="lines+markers", name="Pooled percentile", line=dict(color=RED))
    fig7.add_scatter(x=cv_yr["Percentile"], y=cv_yr["Delayed_%"], mode="lines+markers", name="Within order year", line=dict(color=TEAL))
    fig7.update_layout(title="Delayed % as the percentile cutoff moves (full dataset)", xaxis_title="Percentile cutoff",
                       yaxis_title="Delayed %")
    st.plotly_chart(fig7, **STRETCH)
    if "Ship Date" in filtered_df.columns and filtered_df["Ship Date"].notna().any():
        yo = pd.crosstab(filtered_df[YEAR], filtered_df["Ship Date"].dt.year).rename_axis("Order year", axis=0).rename_axis("Ship year", axis=1)
        st.markdown("**Order year × Ship year (shipments)** — ship dates years after order dates indicate the fields are not a real lag")
        st.dataframe(yo, **STRETCH)
    flat = float(np.nanmax(cv_raw["Delayed_%"]) - np.nanmin(cv_raw["Delayed_%"]))
    st.markdown(
        f"**Interpretation:** the pooled-percentile delay curve moves in steps (range {flat:.0f} pp across P50–P95) when lead times "
        "cluster, so small slider changes can swing the delayed share; the within-year curve is the safer default. "
        "Fix the Ship Date source before treating any of these numbers as operational performance.")
