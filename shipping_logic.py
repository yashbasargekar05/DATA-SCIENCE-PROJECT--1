"""
shipping_logic.py — pure, unit-tested logic for the Nassau Candy Shipping Route Efficiency dashboard.
No Streamlit imports here, so everything can be tested with pytest (see test_shipping_logic.py).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

LEAD = "Shipping Lead Time"
YEAR = "Order Year"
ADJ = "Lead_Adj"

# Trend / plausibility guards (lead time = Ship Date - Order Date)
MAX_PLAUSIBLE_SLOPE = 0.5       # |days of lead time per day of order date| beyond which a trend is not credible
PLAUSIBLE_MAX_LEAD_DAYS = 60    # median lead time above this cannot be literal transit time
MIN_TREND_N = 20
CLUSTER_GAP_DAYS = 30           # a gap this wide between sorted lead times starts a new "cluster"
MEANINGFUL_ETA2 = 0.01          # below 1% of variance explained, a grouping has no practical effect


# ────────────────────────── year adjustment ──────────────────────────
def add_order_year_and_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Order Year and Lead_Adj = lead - (mean lead of that order year) + (overall mean lead).
    Year means come from the FULL data so the adjustment does not drift as filters change.
    Rows without a valid Order Date are left unadjusted (never dropped)."""
    out = df.copy()
    out[YEAR] = out["Order Date"].dt.year
    overall = out[LEAD].mean()
    ymean = out.groupby(YEAR)[LEAD].mean()
    out[ADJ] = out[LEAD] - out[YEAR].map(ymean).fillna(overall) + overall
    return out


# ────────────────────────── delay definition ──────────────────────────
def delay_flag(full: pd.DataFrame, frame: pd.DataFrame, mode: str = "percentile", pct: float = 75,
               sla_days: Optional[float] = None, by_year: bool = True):
    """Returns (is_delayed Series, threshold Series) for `frame`.
    mode='sla'        -> fixed threshold in days.
    mode='percentile' -> percentile of the FULL data; computed within each order year when by_year=True so that
                         'delayed' is not just a proxy for 'ordered in the later year'."""
    if frame.empty:
        return pd.Series(dtype=bool), pd.Series(dtype=float)
    if mode == "sla":
        if sla_days is None:
            raise ValueError("sla_days is required when mode='sla'")
        thr = pd.Series(float(sla_days), index=frame.index)
    else:
        q = pct / 100.0
        overall = full[LEAD].quantile(q)
        if by_year and YEAR in full and full[YEAR].notna().any() and YEAR in frame:
            ymap = full.groupby(YEAR)[LEAD].quantile(q)
            thr = frame[YEAR].map(ymap).astype(float).fillna(overall)
        else:
            thr = pd.Series(float(overall), index=frame.index)
    return frame[LEAD] > thr, thr


def delay_curve(full: pd.DataFrame, by_year: bool, percentiles=range(50, 100, 5)) -> pd.DataFrame:
    rows = []
    for p in percentiles:
        flag, thr = delay_flag(full, full, "percentile", p, by_year=by_year)
        rows.append({"Percentile": p, "Delayed_%": 100 * flag.mean() if len(flag) else np.nan,
                     "Median_Cutoff_Days": float(thr.median()) if len(thr) else np.nan})
    return pd.DataFrame(rows)


# ────────────────────────── summaries ──────────────────────────
SUMMARY_COLS = ["Shipment_Count", "Mean_Lead_Time", "Median_Lead_Time", "Mean_Adjusted_Lead", "Lead_CI95",
                "Delay_Frequency_Pct", "Low_Sample", "Efficiency_Score", "Score_Margin", "Distinct_From_Overall"]


def summarize(frame: pd.DataFrame, group_col: str, count_label: str = "Shipment_Count", min_n: int = 1,
              score_col: str = LEAD) -> pd.DataFrame:
    """Group summary with uncertainty.
    * Lead_CI95 = 1.96 * std / sqrt(n) of `score_col` (NaN when n < 2).
    * Efficiency_Score = 100 * (1 - min-max scaled mean of `score_col`), scaled ONLY among groups with n >= min_n.
    * Score_Margin = the CI expressed in score points (read the score as score ± margin).
    * Distinct_From_Overall = True when the group's mean differs from the overall mean by more than its CI."""
    cols = [group_col, count_label] + [c for c in SUMMARY_COLS if c != "Shipment_Count"]
    if frame.empty:
        return pd.DataFrame(columns=cols)
    adj = ADJ if ADJ in frame.columns else LEAD
    g = frame.groupby(group_col).agg(
        **{count_label: (LEAD, "count")},
        Mean_Lead_Time=(LEAD, "mean"),
        Median_Lead_Time=(LEAD, "median"),
        Mean_Adjusted_Lead=(adj, "mean"),
        _sm=(score_col, "mean"),
        _ss=(score_col, "std"),
        Delay_Frequency_Pct=("Is_Delayed", lambda x: 100 * x.mean()),
    ).reset_index()
    n = g[count_label].astype(float)
    g["Lead_CI95"] = 1.96 * g["_ss"] / np.sqrt(n)
    g["Low_Sample"] = g[count_label] < min_n
    ok = ~g["Low_Sample"]
    g["Efficiency_Score"] = np.nan
    g["Score_Margin"] = np.nan
    if ok.any():
        lo, hi = g.loc[ok, "_sm"].min(), g.loc[ok, "_sm"].max()
        if hi == lo:
            g.loc[ok, "Efficiency_Score"] = 100.0
        else:
            g.loc[ok, "Efficiency_Score"] = 100 * (1 - (g.loc[ok, "_sm"] - lo) / (hi - lo))
            g.loc[ok, "Score_Margin"] = 100 * g.loc[ok, "Lead_CI95"] / (hi - lo)
    overall = frame[score_col].mean()
    g["Distinct_From_Overall"] = ok & ((g["_sm"] - overall).abs() > g["Lead_CI95"]).fillna(False)
    return g.drop(columns=["_sm", "_ss"])[cols]


def eligible_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[~summary["Low_Sample"]] if not summary.empty else summary


# ────────────────────────── diagnostics ──────────────────────────
def eta_squared(frame: pd.DataFrame, group_col: str, value_col: str = LEAD) -> float:
    """Share of variance in value_col explained by group_col (0 = none, 1 = all)."""
    if frame.empty or group_col not in frame.columns:
        return float("nan")
    v = frame[value_col].astype(float)
    ss_tot = ((v - v.mean()) ** 2).sum()
    if ss_tot == 0:
        return float("nan")
    gm = frame.groupby(group_col)[value_col].transform("mean")
    return float(((gm - v.mean()) ** 2).sum() / ss_tot)


def lead_time_clusters(values: pd.Series, gap: float = CLUSTER_GAP_DAYS) -> pd.DataFrame:
    """Splits sorted lead times wherever consecutive values are > `gap` days apart."""
    v = np.sort(values.dropna().to_numpy(dtype=float))
    if v.size == 0:
        return pd.DataFrame(columns=["Cluster", "From_Days", "To_Days", "Shipments", "Share_%"])
    cut = np.where(np.diff(v) > gap)[0] + 1
    parts = np.split(v, cut)
    return pd.DataFrame([{"Cluster": i + 1, "From_Days": p.min(), "To_Days": p.max(), "Shipments": len(p),
                          "Share_%": 100 * len(p) / v.size} for i, p in enumerate(parts)])


def order_date_r2(frame: pd.DataFrame) -> float:
    d = frame.dropna(subset=["Order Date", LEAD])
    if len(d) < 3 or d[LEAD].nunique() < 2:
        return float("nan")
    return float(np.corrcoef(d["Order Date"].astype("int64").astype(float), d[LEAD].astype(float))[0, 1] ** 2)


def diagnostics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0, "median": np.nan, "plausible": False, "clusters": lead_time_clusters(pd.Series(dtype=float)),
                "r2_date": np.nan, "eta2": {}}
    med = float(frame[LEAD].median())
    return {
        "n": len(frame), "median": med, "plausible": med <= PLAUSIBLE_MAX_LEAD_DAYS,
        "clusters": lead_time_clusters(frame[LEAD]), "r2_date": order_date_r2(frame),
        "eta2": {c: eta_squared(frame, c) for c in ["Ship Mode", "Region", "Factory", "State/Province"] if c in frame.columns},
    }


# ────────────────────────── trend ──────────────────────────
def trend_fit(dates: pd.Series, y: pd.Series, min_n: int = MIN_TREND_N) -> Optional[dict]:
    d = pd.DataFrame({"d": dates, "y": y}).dropna()
    if len(d) < min_n:
        return None
    x = ((d["d"] - d["d"].min()).dt.total_seconds() / 86400).to_numpy(dtype=float)
    yy = d["y"].to_numpy(dtype=float)
    if np.unique(x).size < 2:
        return None
    slope, intercept = np.polyfit(x, yy, 1)
    fitted = slope * x + intercept
    ss_tot = np.sum((yy - yy.mean()) ** 2)
    ss_res = np.sum((yy - fitted) ** 2)
    sxx = np.sum((x - x.mean()) ** 2)
    return {"slope": float(slope), "intercept": float(intercept), "n": len(d), "x": x, "fitted": fitted,
            "index": d.index, "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "se": float(np.sqrt(ss_res / (len(d) - 2) / sxx)) if sxx > 0 and len(d) > 2 else float("nan")}


def artifact_reasons(slope: float, median_lead: float, max_slope: float = MAX_PLAUSIBLE_SLOPE,
                     max_lead: float = PLAUSIBLE_MAX_LEAD_DAYS) -> list[str]:
    reasons = []
    if abs(slope) > max_slope:
        reasons.append(f"the slope is {slope:+.2f} days of lead time per day of order date (beyond ±{max_slope}); "
                       "a slope near −1 is what unrelated Ship Dates produce, and a steep positive slope is equally implausible")
    if median_lead > max_lead:
        reasons.append(f"median lead time is {median_lead:,.0f} days, far above any plausible transit time (> {max_lead} days)")
    return reasons
