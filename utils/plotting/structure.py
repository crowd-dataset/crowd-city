"""Figures for the city-level structure diagnostics.

Each figure draws values that ``utils.analytics.structure`` has already
computed and logged, so a figure and the log always agree.

The cluster figure deliberately shows the principal-component cloud without a
partition overlay. Drawing k-means colours on a continuum would suggest groups
that the gap statistic, silhouette and Hopkins tests do not support; the test
values are printed on the figure instead.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

import common
from custom_logger import CustomLogger
from utils.plotting.io import IO


logger = CustomLogger(__name__)
plots_io_class = IO()

# Colour-blind safe qualitative palette (Okabe and Ito).
CONTINENT_COLOURS = {
    "Africa": "#E69F00",
    "Asia": "#0072B2",
    "Europe": "#009E73",
    "North America": "#D55E00",
    "Oceania": "#CC79A7",
    "South America": "#56B4E9",
}
FALLBACK_COLOUR = "#666666"
POSITIVE_COLOUR = "#0072B2"
NEGATIVE_COLOUR = "#D55E00"
NEUTRAL_COLOUR = "#BBBBBB"


def _font(extra: int = 0) -> dict:
    return {
        "family": common.get_configs("font_family"),
        "size": common.get_configs("font_size") + extra,
    }


def _stars(p_value: float) -> str:
    """Significance marker appended to a predictor label."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _metric_label() -> str:
    return (
        "Crossing speed (m/s)"
        if os.environ.get("CROWD_CROSSING_SPEED_UNIT") == "m/s"
        else "Crossing speed"
    )


class StructurePlots:
    """Plotly figures for cluster tendency, gradient and reliability."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # A) Principal-component cloud
    # ------------------------------------------------------------------
    def pca_cloud(
        self,
        tendency: Dict[str, Any],
        city_table,
        file_name: str = "structure_pca_cloud",
        save_file: bool = True,
    ) -> None:
        """Scatter of PC1 against PC2, coloured by continent, no partition."""
        if tendency.get("status") != "complete":
            logger.warning("[structure] Skipping PCA cloud: no tendency result.")
            return

        scores = np.asarray(tendency["scores"], dtype=float)
        if scores.shape[1] < 2:
            logger.warning("[structure] Skipping PCA cloud: fewer than 2 components.")
            return

        features = list(tendency["features"])
        complete = city_table.drop_nulls(features)
        continents = (
            complete["continent"].to_list()
            if "continent" in complete.columns
            else ["Unknown"] * scores.shape[0]
        )
        labels = (
            complete["locality"].to_list()
            if "locality" in complete.columns
            else [""] * scores.shape[0]
        )
        countries = (
            complete["country"].to_list()
            if "country" in complete.columns
            else [""] * scores.shape[0]
        )

        explained = tendency["explained_variance"]
        fig = go.Figure()

        for continent in sorted(set(continents)):
            mask = [value == continent for value in continents]
            fig.add_trace(
                go.Scatter(
                    x=scores[mask, 0],
                    y=scores[mask, 1],
                    mode="markers",
                    name=str(continent),
                    marker=dict(
                        size=9,
                        color=CONTINENT_COLOURS.get(continent, FALLBACK_COLOUR),
                        opacity=0.80,
                        line=dict(width=0.5, color="white"),
                    ),
                    customdata=[
                        [labels[i], countries[i]]
                        for i, keep in enumerate(mask)
                        if keep
                    ],
                    hovertemplate=(
                        "%{customdata[0]}, %{customdata[1]}<br>"
                        "PC1=%{x:.2f}  PC2=%{y:.2f}<extra></extra>"
                    ),
                )
            )

        # Feature loadings as a biplot overlay, scaled to the score cloud.
        loadings = np.asarray(tendency["loadings"], dtype=float)
        scale = 0.85 * max(
            float(np.abs(scores[:, 0]).max()),
            float(np.abs(scores[:, 1]).max()),
        )
        for index, feature in enumerate(features):
            x_end = loadings[index, 0] * scale
            y_end = loadings[index, 1] * scale
            fig.add_annotation(
                x=x_end, y=y_end, ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.1,
                arrowwidth=2, arrowcolor="#333333", opacity=0.85,
            )
            fig.add_annotation(
                x=x_end * 1.10, y=y_end * 1.10,
                text=f"<b>{feature}</b>",
                showarrow=False,
                font={**_font(-2), "color": "#333333"},
            )

        verdict = (
            "cluster structure supported"
            if tendency.get("clustered")
            else "no discrete clusters: single continuum"
        )
        gap_line = (
            "gap statistic: no elbow (k = 1)"
            if tendency["gap"].get("no_interior_maximum")
            else f"gap statistic: optimal k = {tendency['gap']['optimal_k']}"
        )
        fig.add_annotation(
            x=0.01, y=0.99, xref="paper", yref="paper",
            xanchor="left", yanchor="top", align="left", showarrow=False,
            text="<br>".join([
                f"<b>{verdict}</b>",
                f"Hopkins = {tendency['hopkins']:.3f} (0.50 = uniform)",
                gap_line,
                f"best silhouette = {tendency['best_silhouette']:.3f} "
                f"(k = {tendency['best_silhouette_k']})",
                f"n = {tendency['n_cities']} cities",
            ]),
            font=_font(-2),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#999999", borderwidth=1, borderpad=6,
        )

        fig.update_layout(
            template=common.get_configs("plotly_template"),
            xaxis_title=f"PC1 ({explained[0] * 100:.1f}% of variance)",
            yaxis_title=f"PC2 ({explained[1] * 100:.1f}% of variance)",
            font=_font(),
            legend=dict(
                title="", x=0.99, y=0.01, xanchor="right", yanchor="bottom",
                bgcolor="rgba(255,255,255,0.80)",
                bordercolor="rgba(0,0,0,0.15)", borderwidth=1,
            ),
            margin=dict(l=80, r=30, t=30, b=70),
        )
        fig.add_hline(y=0, line=dict(color="#DDDDDD", width=1))
        fig.add_vline(x=0, line=dict(color="#DDDDDD", width=1))

        self._emit(fig, file_name, save_file, width=1200, height=900)

    # ------------------------------------------------------------------
    # B) Cluster-tendency diagnostics
    # ------------------------------------------------------------------
    def cluster_diagnostics(
        self,
        tendency: Dict[str, Any],
        file_name: str = "structure_cluster_diagnostics",
        save_file: bool = True,
    ) -> None:
        """Silhouette against k, and the gap statistic with its error bars."""
        if tendency.get("status") != "complete":
            return

        partitions = tendency["partitions"]
        gap_rows = tendency["gap"]["rows"]

        fig = make_subplots(
            rows=1, cols=2, horizontal_spacing=0.12,
            subplot_titles=(
                "Silhouette coefficient by k",
                "Gap statistic vs uniform reference",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=[row["k"] for row in partitions],
                y=[row["silhouette"] for row in partitions],
                mode="lines+markers",
                marker=dict(size=11, color=POSITIVE_COLOUR),
                line=dict(width=2.5, color=POSITIVE_COLOUR),
                name="silhouette",
                showlegend=False,
                hovertemplate="k=%{x}<br>silhouette=%{y:.3f}<extra></extra>",
            ),
            row=1, col=1,
        )
        # 0.50 is the conventional threshold for "reasonable structure";
        # both labels are anchored left so they cannot sit on the data line.
        fig.add_hline(
            y=0.50, line=dict(color=NEGATIVE_COLOUR, dash="dash", width=2),
            annotation_text="0.50 = reasonable structure",
            annotation_position="top left",
            annotation_font=_font(-4),
            row=1, col=1,
        )
        fig.add_hline(
            y=0.25, line=dict(color="#999999", dash="dot", width=1.5),
            annotation_text="0.25 = weak / artificial",
            annotation_position="top left",
            annotation_font=_font(-4),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[row["k"] for row in gap_rows],
                y=[row["gap"] for row in gap_rows],
                error_y=dict(
                    type="data",
                    array=[row["standard_error"] for row in gap_rows],
                    visible=True, thickness=1.5, width=6,
                ),
                mode="lines+markers",
                marker=dict(size=11, color=POSITIVE_COLOUR),
                line=dict(width=2.5, color=POSITIVE_COLOUR),
                name="gap",
                showlegend=False,
                hovertemplate="k=%{x}<br>gap=%{y:.4f}<extra></extra>",
            ),
            row=1, col=2,
        )

        if tendency["gap"].get("no_interior_maximum"):
            fig.add_annotation(
                x=0.5, y=0.10, xref="x2 domain", yref="y2 domain",
                xanchor="center", showarrow=False, align="center",
                font=_font(-3),
                bgcolor="rgba(255,255,255,0.88)", bordercolor="#999999",
                borderwidth=1, borderpad=5,
                text=(
                    "Gap never turns over within the<br>"
                    "searched range: no interior maximum,<br>"
                    "so no optimal k &gt; 1"
                ),
            )
        else:
            optimal = tendency["gap"]["optimal_k"]
            fig.add_vline(
                x=optimal,
                line=dict(color=NEGATIVE_COLOUR, dash="dash", width=2),
                annotation_text=f"optimal k = {optimal}",
                annotation_position="top",
                annotation_font=_font(-3),
                row=1, col=2,
            )

        fig.update_yaxes(range=[0, 0.65], title_text="Silhouette", row=1, col=1)
        fig.update_xaxes(title_text="Number of clusters k", dtick=1, row=1, col=1)
        fig.update_yaxes(title_text="Gap", row=1, col=2)
        fig.update_xaxes(title_text="Number of clusters k", dtick=1, row=1, col=2)
        fig.update_annotations(font=_font(-1))
        fig.update_layout(
            template=common.get_configs("plotly_template"),
            font=_font(),
            margin=dict(l=70, r=30, t=60, b=70),
        )

        self._emit(fig, file_name, save_file, width=1600, height=700)

    # ------------------------------------------------------------------
    # C) Gradient correlations
    # ------------------------------------------------------------------
    def gradient_correlations(
        self,
        gradient: Dict[str, Any],
        file_name: str = "structure_gradient_correlations",
        save_file: bool = True,
    ) -> None:
        """Spearman rho for each predictor, one panel per headline metric."""
        by_metric = gradient.get("by_metric", {})
        panels = [
            (name, label)
            for name, label in (
                ("speed", _metric_label()),
                ("time", "Crossing initiation time"),
            )
            if by_metric.get(name)
        ]
        if not panels:
            return

        fig = make_subplots(
            rows=1, cols=len(panels), horizontal_spacing=0.22,
            subplot_titles=[label for _, label in panels],
        )

        for column, (metric, _) in enumerate(panels, start=1):
            rows = sorted(by_metric[metric], key=lambda row: row["rho"])
            names = [
                f"{row['predictor']}{_stars(row['p_value'])}"
                for row in rows
            ]
            values = [row["rho"] for row in rows]
            colours = [
                NEUTRAL_COLOUR
                if row["p_value"] >= 0.05
                else (POSITIVE_COLOUR if row["rho"] > 0 else NEGATIVE_COLOUR)
                for row in rows
            ]
            fig.add_trace(
                go.Bar(
                    x=values, y=names, orientation="h",
                    marker=dict(color=colours),
                    customdata=[[row["p_value"], row["n"]] for row in rows],
                    hovertemplate=(
                        "%{y}<br>rho=%{x:.3f}<br>"
                        "p=%{customdata[0]:.4f}  n=%{customdata[1]}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1, col=column,
            )
            fig.add_vline(
                x=0, line=dict(color="#333333", width=1.5),
                row=1, col=column,
            )
            fig.update_xaxes(
                title_text="Spearman rho", range=[-0.55, 0.55],
                zeroline=False, row=1, col=column,
            )
            fig.update_yaxes(tickfont=_font(-4), row=1, col=column)

        fig.add_annotation(
            x=0.5, y=-0.18, xref="paper", yref="paper",
            xanchor="center", showarrow=False, align="center",
            font=_font(-4),
            text=(
                "Grey = not significant at p &lt; 0.05. "
                "* p &lt; 0.05, ** p &lt; 0.01, *** p &lt; 0.001.<br>"
                "Predictors are mutually correlated, so these are convergent "
                "indicators rather than separable effects."
            ),
        )
        fig.update_annotations(font=_font(-1))
        fig.update_layout(
            template=common.get_configs("plotly_template"),
            font=_font(),
            margin=dict(l=200, r=60, t=60, b=150),
        )

        self._emit(fig, file_name, save_file, width=1700, height=850)

    # ------------------------------------------------------------------
    # D) Reliability
    # ------------------------------------------------------------------
    def reliability(
        self,
        reliability: Dict[str, Any],
        file_name: str = "structure_reliability",
        save_file: bool = True,
    ) -> None:
        """Split-half reliability against the minimum-track threshold."""
        series = [
            ("speed", _metric_label(), POSITIVE_COLOUR),
            ("time", "Crossing initiation time", NEGATIVE_COLOUR),
        ]
        if not any(reliability.get(name) for name, _, _ in series):
            return

        fig = go.Figure()
        for metric, label, colour in series:
            rows = reliability.get(metric) or []
            if not rows:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[row["threshold"] for row in rows],
                    y=[row["reliability"] for row in rows],
                    error_y=dict(
                        type="data",
                        array=[row["sd"] for row in rows],
                        visible=True, thickness=1.4, width=6,
                    ),
                    mode="lines+markers+text",
                    text=[f"n={row['n_cities']}" for row in rows],
                    textposition="bottom center",
                    textfont=_font(-6),
                    marker=dict(size=11, color=colour),
                    line=dict(width=2.5, color=colour),
                    name=label,
                    hovertemplate=(
                        "min %{x} tracks/city<br>"
                        "reliability=%{y:.3f}<extra></extra>"
                    ),
                )
            )

        for level, text, colour in (
            (0.80, "0.80 = good for group comparison", "#009E73"),
            (0.50, "0.50 = interpret rankings with caution", "#999999"),
        ):
            # Anchored right: the motion series starts near 0.80 on the left,
            # so a left-anchored label would sit on top of it.
            fig.add_hline(
                y=level, line=dict(color=colour, dash="dash", width=1.6),
                annotation_text=text, annotation_position="top right",
                annotation_font=_font(-4),
            )

        fig.update_layout(
            template=common.get_configs("plotly_template"),
            xaxis_title="Minimum analysed tracks per city",
            yaxis_title="Split-half reliability (Spearman-Brown corrected)",
            yaxis=dict(range=[0, 1.0]),
            font=_font(),
            legend=dict(
                title="", x=0.99, y=0.02, xanchor="right", yanchor="bottom",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.15)", borderwidth=1,
            ),
            margin=dict(l=90, r=40, t=40, b=80),
        )

        self._emit(fig, file_name, save_file, width=1300, height=800)

    # ------------------------------------------------------------------
    # E) Continent distributions
    # ------------------------------------------------------------------
    def continent_distributions(
        self,
        city_table,
        tests: Optional[Dict[str, Any]] = None,
        file_name: str = "structure_continent_distributions",
        save_file: bool = True,
    ) -> None:
        """Box plus point distributions per continent for each metric."""
        panels = [
            (name, label)
            for name, label in (
                ("speed", _metric_label()),
                ("time", "Crossing initiation time (s)"),
                ("pct_unsignalised", "Crossings without traffic signals (%)"),
            )
            if name in city_table.columns
        ]
        if not panels or "continent" not in city_table.columns:
            return

        tests = tests or {}
        order = (
            city_table
            .drop_nulls(["continent"])
            .group_by("continent")
            .agg([pl.len().alias("n")])
            .sort("n", descending=True)["continent"]
            .to_list()
        )

        titles = []
        for name, label in panels:
            test = tests.get(name)
            titles.append(
                f"{label}<br><sub>Kruskal-Wallis H={test['H']:.2f}, "
                f"p={test['p_value']:.3g}</sub>"
                if test
                else label
            )

        fig = make_subplots(
            rows=1, cols=len(panels),
            horizontal_spacing=0.07, subplot_titles=titles,
        )

        for column, (metric, _) in enumerate(panels, start=1):
            for continent in order:
                values = (
                    city_table
                    .filter(pl.col("continent") == continent)
                    .drop_nulls([metric])[metric]
                    .to_list()
                )
                if not values:
                    continue
                fig.add_trace(
                    go.Box(
                        y=values, name=str(continent),
                        marker=dict(
                            color=CONTINENT_COLOURS.get(continent, FALLBACK_COLOUR),
                            size=5, opacity=0.65,
                        ),
                        line=dict(width=1.6),
                        boxpoints="all", jitter=0.45, pointpos=0,
                        showlegend=False,
                        hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
                    ),
                    row=1, col=column,
                )
            fig.update_xaxes(tickangle=-30, tickfont=_font(-5), row=1, col=column)

        fig.update_annotations(font=_font(-2))
        fig.update_layout(
            template=common.get_configs("plotly_template"),
            font=_font(),
            margin=dict(l=70, r=40, t=100, b=110),
        )

        self._emit(fig, file_name, save_file, width=1800, height=800)

    # ------------------------------------------------------------------
    # F) Initiation-time floor
    # ------------------------------------------------------------------
    def initiation_time_floor(
        self,
        floor: Dict[str, Any],
        file_name: str = "structure_time_floor",
        save_file: bool = True,
    ) -> None:
        """Histogram of per-track initiation times showing the estimator floor."""
        if not floor or floor.get("n_tracks", 0) == 0:
            return

        seconds = np.asarray(floor["seconds"], dtype=float)
        step = floor["step_seconds"]
        floor_seconds = floor["floor_seconds"]
        upper = float(np.percentile(seconds, 99))

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=seconds,
                xbins=dict(start=0, end=max(upper, floor_seconds + step), size=step),
                marker=dict(color=POSITIVE_COLOUR, line=dict(width=0)),
                name="tracks",
                hovertemplate="%{x:.2f} s<br>%{y} tracks<extra></extra>",
            )
        )
        fig.add_vline(
            x=floor_seconds,
            line=dict(color=NEGATIVE_COLOUR, width=2.5, dash="dash"),
            annotation_text=(
                f"estimator floor {floor_seconds:.2f} s "
                f"({floor['fraction_at_floor'] * 100:.1f}% of tracks)"
            ),
            annotation_position="top right",
            annotation_font=_font(-3),
        )
        fig.add_annotation(
            x=0.99, y=0.80, xref="paper", yref="paper",
            xanchor="right", align="right", showarrow=False,
            font=_font(-3),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#999999", borderwidth=1, borderpad=6,
            text=(
                f"{floor['n_tracks']:,} tracks<br>"
                f"{floor['distinct_values']} distinct values<br>"
                f"quantised in steps of {step:.3f} s"
            ),
        )
        fig.update_layout(
            template=common.get_configs("plotly_template"),
            xaxis_title="Crossing initiation time (s)",
            yaxis_title="Number of analysed tracks",
            font=_font(),
            bargap=0.02,
            margin=dict(l=90, r=40, t=40, b=80),
        )

        self._emit(fig, file_name, save_file, width=1300, height=800)

    # ------------------------------------------------------------------
    # G) Day versus night
    # ------------------------------------------------------------------
    def day_night(
        self,
        day_night: Dict[str, Any],
        file_name: str = "structure_day_night",
        save_file: bool = True,
    ) -> None:
        """Day/night difference against the minimum track support."""
        panels = [
            (name, label)
            for name, label in (
                ("speed", _metric_label()),
                ("time", "Crossing initiation time (s)"),
            )
            if day_night.get(name)
        ]
        if not panels:
            return

        fig = make_subplots(
            rows=1, cols=len(panels), horizontal_spacing=0.14,
            subplot_titles=[label for _, label in panels],
        )

        for column, (metric, _) in enumerate(panels, start=1):
            rows = day_night[metric]
            categories = [
                f"≥{row['threshold']} tracks<br>n={row['n']}" for row in rows
            ]
            for name, key, colour in (
                ("Day", "day", "#E69F00"),
                ("Night", "night", "#0072B2"),
            ):
                fig.add_trace(
                    go.Bar(
                        x=categories, y=[row[key] for row in rows],
                        name=name, marker=dict(color=colour),
                        showlegend=(column == 1),
                        hovertemplate=f"{name}: %{{y:.3f}}<extra></extra>",
                    ),
                    row=1, col=column,
                )
            for index, row in enumerate(rows):
                marker = (
                    "***" if row["p_value"] < 0.001
                    else "**" if row["p_value"] < 0.01
                    else "*" if row["p_value"] < 0.05
                    else "n.s."
                )
                fig.add_annotation(
                    x=categories[index],
                    y=max(row["day"], row["night"]) * 1.06,
                    text=f"{marker}<br><sub>p={row['p_value']:.3g}</sub>",
                    showarrow=False, font=_font(-4),
                    row=1, col=column,
                )
            fig.update_yaxes(title_text="City mean", row=1, col=column)
            fig.update_xaxes(tickfont=_font(-5), row=1, col=column)

        fig.add_annotation(
            x=0.5, y=-0.32, xref="paper", yref="paper",
            xanchor="center", showarrow=False, align="center",
            font=_font(-4),
            text=(
                "Each group restricts the paired test to cities with at least "
                "that many analysed tracks in both conditions.<br>"
                "An effect that survives only the leftmost group is driven by "
                "thin night-time sampling."
            ),
        )
        fig.update_annotations(font=_font(-1))
        # The legend is placed below the panels: inside the plotting area it
        # would collide with the significance markers above the tallest bars.
        fig.update_layout(
            template=common.get_configs("plotly_template"),
            barmode="group", font=_font(),
            legend=dict(
                title="", orientation="h",
                x=0.5, y=-0.11, xanchor="center", yanchor="top",
                bgcolor="rgba(255,255,255,0.85)",
            ),
            margin=dict(l=90, r=60, t=70, b=190),
        )

        self._emit(fig, file_name, save_file, width=1500, height=800)

    # ------------------------------------------------------------------
    # Shared output
    # ------------------------------------------------------------------
    @staticmethod
    def _emit(fig, file_name: str, save_file: bool, width: int, height: int) -> None:
        if save_file:
            plots_io_class.save_plotly_figure(
                fig, file_name, width=width, height=height, save_final=True
            )
        else:
            fig.show()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def plot_all(self, result: Dict[str, Any], save_file: bool = True) -> None:
        """Draw every structure figure from one ``analyse_structure`` result."""
        if result.get("status") != "complete":
            logger.warning("[structure] No structure result to plot.")
            return

        city_table = result["city_table"]
        figures = [
            ("PCA cloud", lambda: self.pca_cloud(
                result["cluster_tendency"], city_table, save_file=save_file)),
            ("cluster diagnostics", lambda: self.cluster_diagnostics(
                result["cluster_tendency"], save_file=save_file)),
            ("gradient correlations", lambda: self.gradient_correlations(
                result["gradient"], save_file=save_file)),
            ("reliability", lambda: self.reliability(
                result["reliability"], save_file=save_file)),
            ("continent distributions", lambda: self.continent_distributions(
                city_table, result.get("continents", {}).get("tests"),
                save_file=save_file)),
            ("initiation-time floor", lambda: self.initiation_time_floor(
                result["time_floor"], save_file=save_file)),
            ("day versus night", lambda: self.day_night(
                result["day_night"], save_file=save_file)),
        ]

        for name, draw in figures:
            try:
                draw()
            except Exception as error:
                logger.error(
                    "[structure] Could not draw the {} figure: {}", name, error
                )
