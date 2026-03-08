import math

import pandas as pd
from meterstick import metrics, operations
from scipy import stats

from brybrydataguy.meterstick_ext import InteractionEffect, MixEffect, WithinEffect


def _sample_df() -> pd.DataFrame:
    rows = []
    values = {
        1: {"A": {"s1": (20, 100), "s2": (10, 50)}, "B": {"s1": (27, 110), "s2": (9, 40)}},
        2: {"A": {"s1": (18, 90), "s2": (8, 60)}, "B": {"s1": (24, 100), "s2": (7, 50)}},
        3: {"A": {"s1": (25, 120), "s2": (11, 55)}, "B": {"s1": (30, 130), "s2": (12, 65)}},
        4: {"A": {"s1": (22, 110), "s2": (13, 70)}, "B": {"s1": (24, 115), "s2": (15, 75)}},
        5: {"A": {"s1": (19, 95), "s2": (9, 45)}, "B": {"s1": (23, 105), "s2": (11, 55)}},
        6: {"A": {"s1": (21, 105), "s2": (10, 65)}, "B": {"s1": (22, 100), "s2": (13, 70)}},
    }
    for unit, per_condition in values.items():
        for condition in ("A", "B"):
            for strata, (num, den) in per_condition[condition].items():
                rows.append(
                    {
                        "unit": unit,
                        "group": "g1" if unit <= 3 else "g2",
                        "condition": condition,
                        "strata": strata,
                        "num": num,
                        "den": den,
                    }
                )
    return pd.DataFrame(rows)


def _manual_decomposition_terms(df: pd.DataFrame) -> tuple[float, float, float]:
    grouped = df.groupby(["condition", "strata"], observed=True)[["num", "den"]].sum()
    n_a, d_a = grouped.loc["A", "num"], grouped.loc["A", "den"]
    n_b, d_b = grouped.loc["B", "num"], grouped.loc["B", "den"]
    r_a, r_b = n_a / d_a, n_b / d_b
    w_a, w_b = d_a / d_a.sum(), d_b / d_b.sum()
    delta_w, delta_r = w_b - w_a, r_b - r_a
    within = (w_a * delta_r).sum()
    mix = (delta_w * r_a).sum()
    interaction = (delta_w * delta_r).sum()
    return float(within), float(mix), float(interaction)


def _overall_rate_delta(df: pd.DataFrame) -> float:
    by_condition = df.groupby("condition", observed=True)[["num", "den"]].sum()
    rate_a = by_condition.loc["A", "num"] / by_condition.loc["A", "den"]
    rate_b = by_condition.loc["B", "num"] / by_condition.loc["B", "den"]
    return float(rate_b - rate_a)


def test_decomposition_terms_match_manual_by_group():
    df = _sample_df()
    ratio = metrics.Ratio("num", "den")
    split_by = ["group"]

    within = WithinEffect("condition", "A", "strata", metric=ratio).compute_on(df, split_by)
    mix = MixEffect("condition", "A", "strata", metric=ratio).compute_on(df, split_by)
    interaction = InteractionEffect("condition", "A", "strata", metric=ratio).compute_on(df, split_by)

    for group_name, group_df in df.groupby("group", observed=True):
        expected_within, expected_mix, expected_interaction = _manual_decomposition_terms(group_df)
        assert math.isclose(within.loc[(group_name, "B")].iloc[0], expected_within, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(mix.loc[(group_name, "B")].iloc[0], expected_mix, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(interaction.loc[(group_name, "B")].iloc[0], expected_interaction, rel_tol=0, abs_tol=1e-12)


def test_terms_sum_to_total_rate_change():
    df = _sample_df()
    ratio = metrics.Ratio("num", "den")
    within = WithinEffect("condition", "A", "strata", metric=ratio).compute_on(df).iloc[0, 0]
    mix = MixEffect("condition", "A", "strata", metric=ratio).compute_on(df).iloc[0, 0]
    interaction = InteractionEffect("condition", "A", "strata", metric=ratio).compute_on(df).iloc[0, 0]

    expected_delta = _overall_rate_delta(df)
    assert math.isclose(within + mix + interaction, expected_delta, rel_tol=0, abs_tol=1e-12)


def test_as_percent_scales_by_baseline_rate():
    df = _sample_df()
    ratio = metrics.Ratio("num", "den")
    within_abs = WithinEffect("condition", "A", "strata", metric=ratio).compute_on(df).iloc[0, 0]
    within_pct = WithinEffect("condition", "A", "strata", metric=ratio, as_percent=True).compute_on(df).iloc[0, 0]

    baseline = df[df["condition"] == "A"][["num", "den"]].sum()
    baseline_rate = baseline["num"] / baseline["den"]
    expected_pct = (within_abs / baseline_rate) * 100.0
    assert math.isclose(within_pct, expected_pct, rel_tol=0, abs_tol=1e-12)


def test_jackknife_ci_matches_manual_leave_one_out():
    df = _sample_df()
    ratio = metrics.Ratio("num", "den")
    metric = WithinEffect("condition", "A", "strata", metric=ratio)
    confidence = 0.95

    jk = operations.Jackknife("unit", metric, confidence=confidence)
    res = jk.compute_on(df, melted=True)
    observed = res.iloc[0]
    observed_value = float(observed["Value"])
    observed_lower = float(observed["Jackknife CI-lower"])
    observed_upper = float(observed["Jackknife CI-upper"])

    point, _, _ = _manual_decomposition_terms(df)
    loo = []
    for unit in sorted(df["unit"].unique()):
        loo_point, _, _ = _manual_decomposition_terms(df[df["unit"] != unit])
        loo.append(loo_point)
    loo = pd.Series(loo)
    dof = len(loo) - 1
    stderr = float(loo.std(ddof=1) * dof / math.sqrt(dof + 1))
    half_width = float(stderr * stats.t.ppf((1 + confidence) / 2, dof))
    expected_lower = point - half_width
    expected_upper = point + half_width

    assert math.isclose(observed_value, point, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(observed_lower, expected_lower, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(observed_upper, expected_upper, rel_tol=0, abs_tol=1e-12)
    assert observed_lower <= observed_value <= observed_upper
