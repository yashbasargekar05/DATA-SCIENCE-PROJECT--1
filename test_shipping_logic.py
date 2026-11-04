"""Run with:  pytest -q test_shipping_logic.py"""
import numpy as np
import pandas as pd
import pytest

import shipping_logic as L


def make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    df = pd.DataFrame({
        "Row ID": range(n), "Order Date": dates,
        "Shipping Lead Time": rng.normal(10, 2, n),
        "Route": rng.choice(["A -> X", "B -> Y", "C -> Z"], n),
        "Ship Mode": rng.choice(["S1", "S2"], n),
    })
    return L.add_order_year_and_adjustment(df)


def flagged(df, **kw):
    out = df.copy()
    out["Is_Delayed"], _ = L.delay_flag(out, out, **kw)
    return out


# ---------- year adjustment ----------
def test_year_adjustment_removes_year_effect_and_keeps_rows():
    df = make_df()
    df.loc[df["Order Year"] == 2025, "Shipping Lead Time"] += 500          # big artificial year step
    df = L.add_order_year_and_adjustment(df.drop(columns=["Order Year", "Lead_Adj"]))
    means = df.groupby("Order Year")[L.ADJ].mean()
    assert means.max() - means.min() < 1e-9
    assert len(df) == 200 and df[L.ADJ].notna().all()


def test_missing_order_date_row_kept_and_unadjusted():
    df = make_df(50)
    df.loc[0, "Order Date"] = pd.NaT
    out = L.add_order_year_and_adjustment(df.drop(columns=["Order Year", "Lead_Adj"]))
    assert len(out) == 50 and out.loc[0, L.ADJ] == pytest.approx(out.loc[0, L.LEAD])


# ---------- delay definition ----------
def test_percentile_delay_share_is_about_complement():
    df = make_df(2000)
    flag, _ = L.delay_flag(df, df, "percentile", 75, by_year=False)
    assert flag.mean() == pytest.approx(0.25, abs=0.01)


def test_by_year_delay_is_balanced_across_years():
    df = make_df(2000)
    df.loc[df["Order Year"] == 2025, "Shipping Lead Time"] += 500
    df = L.add_order_year_and_adjustment(df.drop(columns=["Order Year", "Lead_Adj"]))
    raw, _ = L.delay_flag(df, df, "percentile", 75, by_year=False)
    yr, _ = L.delay_flag(df, df, "percentile", 75, by_year=True)
    assert raw[df["Order Year"] == 2025].mean() > 0.4           # raw rule == "ordered in 2025"
    assert raw[df["Order Year"] == 2024].mean() < 0.01
    for y in (2024, 2025):
        assert yr[df["Order Year"] == y].mean() == pytest.approx(0.25, abs=0.03)


def test_sla_mode_and_validation():
    df = make_df()
    flag, thr = L.delay_flag(df, df, "sla", sla_days=10)
    assert (thr == 10).all() and flag.equals(df[L.LEAD] > 10)
    with pytest.raises(ValueError):
        L.delay_flag(df, df, "sla")


def test_delay_curve_is_non_increasing_and_empty_safe():
    c = L.delay_curve(make_df(500), by_year=False)
    assert c["Delayed_%"].is_monotonic_decreasing
    f, t = L.delay_flag(make_df(), make_df().iloc[0:0], "percentile", 75)
    assert f.empty and t.empty


# ---------- summaries ----------
def test_summarize_matches_manual_calc_and_flags_low_sample():
    df = flagged(make_df(300), mode="percentile", pct=75)
    s = L.summarize(df, "Route", min_n=10)
    r = s[s["Route"] == "A -> X"].iloc[0]
    sub = df[df["Route"] == "A -> X"][L.LEAD]
    assert r["Shipment_Count"] == len(sub)
    assert r["Mean_Lead_Time"] == pytest.approx(sub.mean())
    assert r["Lead_CI95"] == pytest.approx(1.96 * sub.std() / np.sqrt(len(sub)))
    assert s["Efficiency_Score"].max() == pytest.approx(100) and s["Efficiency_Score"].min() == pytest.approx(0)
    tiny = L.summarize(df, "Route", min_n=10_000)
    assert tiny["Low_Sample"].all() and tiny["Efficiency_Score"].isna().all()


def test_small_group_cannot_stretch_the_scale():
    df = flagged(make_df(300), mode="percentile", pct=75)
    extra = df.iloc[:1].copy(); extra["Route"] = "TINY -> T"; extra[L.LEAD] = 9999.0
    s = L.summarize(pd.concat([df, extra], ignore_index=True), "Route", min_n=10)
    assert s.loc[s["Route"] == "TINY -> T", "Efficiency_Score"].isna().all()
    assert s["Efficiency_Score"].min() == pytest.approx(0)       # scale still defined by the eligible routes


def test_distinct_from_overall_requires_gap_beyond_ci():
    df = flagged(make_df(600), mode="percentile", pct=75)
    df.loc[df["Route"] == "A -> X", L.LEAD] += 50
    s = L.summarize(df, "Route", min_n=10)
    assert bool(s.loc[s["Route"] == "A -> X", "Distinct_From_Overall"].iloc[0])


def test_summarize_empty_and_constant_safe():
    assert L.summarize(pd.DataFrame(columns=["Route", L.LEAD, "Is_Delayed"]), "Route").empty
    df = flagged(make_df(60), mode="percentile", pct=75); df[L.LEAD] = 5.0; df[L.ADJ] = 5.0
    s = L.summarize(df, "Route", min_n=1)
    assert (s["Efficiency_Score"] == 100).all()


# ---------- diagnostics ----------
def test_eta_squared_bounds():
    df = pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5, L.LEAD: [1.0] * 5 + [9.0] * 5})
    assert L.eta_squared(df, "g") == pytest.approx(1.0)
    df[L.LEAD] = np.arange(10.0); df["g"] = ["a", "b"] * 5
    assert 0 <= L.eta_squared(df, "g") < 0.1
    assert np.isnan(L.eta_squared(pd.DataFrame({"g": [1, 2], L.LEAD: [3.0, 3.0]}), "g"))


def test_cluster_detection():
    v = pd.Series([900, 905, 920, 1270, 1275, 1640, 1642])
    c = L.lead_time_clusters(v)
    assert list(c["Shipments"]) == [3, 2, 2] and c["Share_%"].sum() == pytest.approx(100)
    assert L.lead_time_clusters(pd.Series(dtype=float)).empty


def test_diagnostics_flags_implausible_lead_times():
    d = L.diagnostics(make_df(100))
    assert d["plausible"] is True
    big = make_df(100); big[L.LEAD] += 1000
    assert L.diagnostics(big)["plausible"] is False


# ---------- trend ----------
def test_trend_fit_recovers_known_slope_and_min_n():
    dates = pd.Series(pd.date_range("2024-01-01", periods=100))
    t = L.trend_fit(dates, pd.Series(12 + 0.1 * np.arange(100.0)))
    assert t["slope"] == pytest.approx(0.1) and t["r2"] == pytest.approx(1.0)
    assert L.trend_fit(dates.head(5), pd.Series(np.arange(5.0))) is None
    assert L.trend_fit(pd.Series([pd.Timestamp("2024-01-01")] * 30), pd.Series(np.arange(30.0))) is None


def test_artifact_reasons():
    assert L.artifact_reasons(0.1, 12) == []
    assert len(L.artifact_reasons(-1.0, 12)) == 1
    assert len(L.artifact_reasons(0.8, 1300)) == 2
