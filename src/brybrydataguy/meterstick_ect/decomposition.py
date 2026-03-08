"""Decomposition operators for meterstick ratio metrics."""

from __future__ import annotations

import pandas as pd
from meterstick import metrics, operations


class _DecompositionBase(operations.Comparison):
    """
    Base for Within/Mix/Interaction.

    Structured identically to MH so Jackknife precomputation can be reused.
    """

    _term_name = "Decomposition"

    def __init__(
        self,
        condition_column,
        baseline_key,
        stratified_by,
        metric=None,
        as_percent=False,
        include_base=False,
        name_tmpl=None,
        **kwargs,
    ):
        stratified_by = stratified_by if isinstance(stratified_by, list) else [stratified_by]
        condition_column = [condition_column] if isinstance(condition_column, str) else condition_column
        self.as_percent = as_percent
        suffix = " %" if as_percent else ""
        name_tmpl = name_tmpl or ("{} " + self._term_name + suffix)

        super().__init__(
            condition_column + stratified_by,
            baseline_key,
            metric,
            include_base,
            name_tmpl,
            extra_index=condition_column,
            additional_fingerprint_attrs=["as_percent"],
            **kwargs,
        )
        self.precomputable_in_jk_bs = True

    @property
    def stratified_by(self):
        return self.extra_split_by[len(self.extra_index) :]

    def check_is_ratio(self, metric):
        if isinstance(metric, metrics.MetricList):
            for child_metric in metric:
                self.check_is_ratio(child_metric)
        elif not isinstance(metric, (metrics.CompositeMetric, metrics.Ratio)):
            raise ValueError(f"{self._term_name} requires a Ratio metric.")
        elif metric.op(2.0, 2) != 1:
            raise ValueError(f"{self._term_name} requires a Ratio metric (a/b).")

    def compute_children(self, df, split_by=None, melted=False, return_dataframe=True, cache_key=None):
        child = self.children[0]
        self.check_is_ratio(child)
        if isinstance(child, metrics.MetricList):
            children = []
            for child_metric in child.children:
                util_metric = metrics.MetricList(
                    [metrics.MetricList(child_metric.children, where=child_metric.where_)],
                    where=child.where_,
                )
                children.append(self.compute_util_metric_on(util_metric, df, split_by, cache_key=cache_key))
            return children
        util_metric = metrics.MetricList(child.children, where=child.where_)
        return self.compute_util_metric_on(util_metric, df, split_by, cache_key=cache_key)

    def compute_on_children(self, children, split_by):
        child = self.children[0]
        if isinstance(child, metrics.MetricList):
            res = [self._compute_term(metric_child, data_child, split_by) for metric_child, data_child in zip(child.children, children)]
            return pd.concat(res, axis=1, sort=False)
        return self._compute_term(child, children, split_by)

    def _compute_term(self, child, df_all, split_by):
        """
        df_all: DataFrame indexed by (...split_by, condition, strata)
        with columns [sum(numer), sum(denom)].
        """
        level = self.extra_index[0] if len(self.extra_index) == 1 else self.extra_index
        numer = child.children[0].name
        denom = child.children[1].name

        df_baseline = df_all.xs(self.baseline_key, level=level)
        suffix = "_base"
        df_joined = df_all.join(df_baseline, rsuffix=suffix)

        n_b, d_b = df_joined[numer], df_joined[denom]
        n_a, d_a = df_joined[numer + suffix], df_joined[denom + suffix]

        r_a = n_a / d_a
        r_b = n_b / d_b

        to_split = [idx_name for idx_name in n_a.index.names if idx_name not in self.stratified_by]

        d_total_a = d_a.groupby(to_split, observed=True).transform("sum")
        d_total_b = d_b.groupby(to_split, observed=True).transform("sum")
        w_a = d_a / d_total_a
        w_b = d_b / d_total_b

        delta_w = w_b - w_a
        delta_r = r_b - r_a

        per_stratum = self._term_formula(w_a, w_b, r_a, r_b, delta_w, delta_r)
        res = per_stratum.groupby(to_split, observed=True).sum()

        if self.as_percent:
            baseline_rate = n_a.groupby(to_split, observed=True).sum() / d_total_a.groupby(to_split, observed=True).first()
            res = (res / baseline_rate) * 100

        res.name = child.name

        to_split_outer = [idx_name for idx_name in to_split if idx_name not in self.extra_index]
        if to_split_outer:
            split_by = split_by or []
            extra_idx = [idx_name for idx_name in to_split_outer if idx_name not in split_by]
            res = res.reorder_levels(split_by + self.extra_index + extra_idx)

        if not self.include_base:
            to_drop = [idx_name for idx_name in res.index.names if idx_name not in self.extra_index]
            idx_to_match = res.index.droplevel(to_drop) if to_drop else res.index
            res = res[~idx_to_match.isin([self.baseline_key])]

        return pd.DataFrame(res.sort_index(level=(split_by or []) + self.extra_index))

    def _term_formula(self, w_a, w_b, r_a, r_b, delta_w, delta_r):
        raise NotImplementedError


class WithinEffect(_DecompositionBase):
    """Σ w_{i,A} * Δr_i: rate changes at fixed baseline mix."""

    _term_name = "Within"

    def _term_formula(self, w_a, w_b, r_a, r_b, delta_w, delta_r):
        return w_a * delta_r


class MixEffect(_DecompositionBase):
    """Σ Δw_i * r_{i,A}: mix shift at fixed baseline rates."""

    _term_name = "Mix"

    def _term_formula(self, w_a, w_b, r_a, r_b, delta_w, delta_r):
        return delta_w * r_a


class InteractionEffect(_DecompositionBase):
    """Σ Δw_i * Δr_i: correlated changes in mix and rate."""

    _term_name = "Interaction"

    def _term_formula(self, w_a, w_b, r_a, r_b, delta_w, delta_r):
        return delta_w * delta_r

