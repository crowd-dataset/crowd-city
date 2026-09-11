"""City-level structure diagnostics for the CROWD crossing metrics.

This module answers one question that the descriptive rollups cannot: do the
analysed cities fall into discrete behavioural types, or do they sit on a
continuum?

Three independent cluster-tendency tests are reported, because a k-means
partition always returns k groups regardless of whether groups exist:

* The Hopkins statistic compares nearest-neighbour distances of the observed
  cities against uniform samples drawn from the same bounding box. A value
  near 0.5 means the configuration is indistinguishable from uniform noise.
* The gap statistic (Tibshirani, Walther and Hastie, 2001) compares the
  within-cluster dispersion against a PCA-aligned uniform reference. An
  optimal ``k`` of 1 means no cluster structure.
* Silhouette coefficients quantify separation for each candidate ``k``.

The module also reports the split-half reliability of every city estimate.
Reliability bounds the correlations that can be observed at all, so it is
required before any city ranking is interpreted.

Everything computed here is written to the log through ``CustomLogger`` so the
values can be quoted directly, and returned as a dictionary so the plotting
layer can draw the same numbers without recomputing them.

No dependency outside numpy, scipy and polars is used, so ``uv sync --frozen``
remains valid.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
from scipy import stats

from custom_logger import CustomLogger


logger = CustomLogger(__name__)

# Behavioural features used for the cluster-tendency tests. These are the four
# city-level quantities the analysis produces independently of one another.
CLUSTER_FEATURES: Tuple[str, ...] = (
    "speed",
    "time",
    "pct_unsignalised",
    "crossings_per_hour",
)

# Contextual predictors screened against the two headline metrics.
GRADIENT_PREDICTORS: Tuple[str, ...] = (
    "avg_height",
    "pct_unsignalised",
    "med_age",
    "traffic_mortality",
    "gmp_per_capita",
    "literacy_rate",
    "gini",
    "traffic_index",
    "population_locality",
    "crossings_per_hour",
    "n_speed_tracks",
    "hours",
)

# Minimum-support thresholds used for the reliability and day/night sweeps.
SUPPORT_THRESHOLDS: Tuple[int, ...] = (1, 10, 20, 30, 50)

DEFAULT_RANDOM_SEED = 13


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _significance(p_value: float) -> str:
    """Return the conventional significance marker for a p value."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _split_locality_condition(key: str) -> Optional[Tuple[str, str]]:
    """Split ``{locality}_{lat}_{lon}_{condition}`` from the right.

    rsplit is required because a locality name may itself contain an
    underscore. Returns ``(base_key, condition)`` or None when malformed.
    """
    parts = str(key).rsplit("_", 3)
    if len(parts) != 4:
        return None
    locality, latitude, longitude, condition = parts
    try:
        float(latitude)
        float(longitude)
    except (TypeError, ValueError):
        return None
    return f"{locality}_{latitude}_{longitude}", str(condition)


def _finite(values: Sequence[Any]) -> np.ndarray:
    """Return the finite float values of a sequence."""
    output: List[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            output.append(numeric)
    return np.asarray(output, dtype=float)


def _rank_z(matrix: np.ndarray) -> np.ndarray:
    """Rank-transform then standardise each column.

    The gradient analysis is reported with Spearman correlations, so the
    cluster tests are run in the same rank space. This also removes the heavy
    right tail of the crossings-per-hour distribution.
    """
    return np.column_stack([
        stats.zscore(stats.rankdata(matrix[:, column]))
        for column in range(matrix.shape[1])
    ])


# ---------------------------------------------------------------------
# City table
# ---------------------------------------------------------------------

def build_city_table(
    df_mapping: pl.DataFrame,
    avg_speed_locality: dict,
    avg_time_locality: dict,
    pedestrian_cross_locality: dict,
    crossings_with_traffic_equipment_locality: dict,
    crossings_without_traffic_equipment_locality: dict,
    all_speed_locality: dict,
    all_time_locality: dict,
) -> pl.DataFrame:
    """Assemble one row per city from the locality-condition dictionaries.

    Day (condition 0) and night (condition 1) values are kept in their own
    columns and also averaged over whichever conditions are present, so a city
    observed only during the day is not penalised.
    """
    per_city: Dict[str, Dict[str, float]] = {}

    scalar_sources = [
        (avg_speed_locality, "speed"),
        (avg_time_locality, "time"),
        (pedestrian_cross_locality, "cross"),
        (crossings_with_traffic_equipment_locality, "with_eq"),
        (crossings_without_traffic_equipment_locality, "no_eq"),
    ]
    for source, name in scalar_sources:
        for key, value in (source or {}).items():
            parsed = _split_locality_condition(key)
            if parsed is None:
                continue
            base, condition = parsed
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                per_city.setdefault(base, {})[f"{name}_{condition}"] = numeric

    # Track counts behind each city estimate, used for the support sweeps.
    count_sources = [
        (all_speed_locality, "n_speed"),
        (all_time_locality, "n_time"),
    ]
    for source, name in count_sources:
        for key, values in (source or {}).items():
            parsed = _split_locality_condition(key)
            if parsed is None:
                continue
            base, condition = parsed
            try:
                count = len(values)
            except TypeError:
                continue
            per_city.setdefault(base, {})[f"{name}_{condition}"] = float(count)

    if not per_city:
        logger.warning(
            "[structure] No locality-condition values were available; "
            "the city table is empty."
        )
        return pl.DataFrame()

    records: List[Dict[str, Any]] = []
    for base, values in per_city.items():
        locality, latitude, longitude = base.rsplit("_", 2)
        record: Dict[str, Any] = {
            "locality": locality,
            "lat": float(latitude),
            "lon": float(longitude),
        }
        record.update(values)
        records.append(record)

    tidy = pl.DataFrame(records, infer_schema_length=None)

    # Guarantee every day/night column exists even when one condition is
    # entirely absent from this run.
    for name in ("speed", "time", "cross", "with_eq", "no_eq", "n_speed", "n_time"):
        for condition in ("0", "1"):
            column = f"{name}_{condition}"
            if column not in tidy.columns:
                tidy = tidy.with_columns(
                    pl.lit(None).cast(pl.Float64).alias(column)
                )

    context_columns = [
        column
        for column in (
            "locality", "lat", "lon", "country", "iso3", "continent",
            "population_locality", "population_country", "gmp",
            "traffic_mortality", "literacy_rate", "gini", "med_age",
            "traffic_index", "avg_height", "total_time", "total_videos",
        )
        if column in df_mapping.columns
    ]
    table = tidy.join(
        df_mapping.select(context_columns),
        on=["locality", "lat", "lon"],
        how="inner",
    )

    if table.height == 0:
        logger.warning(
            "[structure] No city rows survived the join with the mapping."
        )
        return table

    def mean_of_conditions(name: str) -> pl.Expr:
        day = pl.col(f"{name}_0")
        night = pl.col(f"{name}_1")
        present = (
            day.is_not_null().cast(pl.Int8) + night.is_not_null().cast(pl.Int8)
        ).cast(pl.Float64)
        return (
            pl.when(present > 0)
            .then((day.fill_null(0.0) + night.fill_null(0.0)) / present)
            .otherwise(None)
            .alias(name)
        )

    table = table.with_columns([
        mean_of_conditions("speed"),
        mean_of_conditions("time"),
        (pl.col("cross_0").fill_null(0.0)
         + pl.col("cross_1").fill_null(0.0)).alias("crossings"),
        (pl.col("with_eq_0").fill_null(0.0)
         + pl.col("with_eq_1").fill_null(0.0)).alias("with_eq"),
        (pl.col("no_eq_0").fill_null(0.0)
         + pl.col("no_eq_1").fill_null(0.0)).alias("no_eq"),
        (pl.col("n_speed_0").fill_null(0.0)
         + pl.col("n_speed_1").fill_null(0.0)).alias("n_speed_tracks"),
        (pl.col("n_time_0").fill_null(0.0)
         + pl.col("n_time_1").fill_null(0.0)).alias("n_time_tracks"),
    ])

    # A gross metropolitan product of zero means "not available" in the
    # mapping file rather than an economy of zero size. Treating it as a real
    # value would drag the GMP correlation towards an artefact.
    if "gmp" in table.columns:
        table = table.with_columns(
            pl.when(pl.col("gmp") > 0)
            .then(pl.col("gmp"))
            .otherwise(None)
            .alias("gmp")
        )
        table = table.with_columns(
            (pl.col("gmp") * 1e9 / pl.col("population_locality"))
            .alias("gmp_per_capita")
        )
    else:
        table = table.with_columns(
            pl.lit(None).cast(pl.Float64).alias("gmp_per_capita")
        )

    equipment_total = pl.col("with_eq") + pl.col("no_eq")
    footage_hours = pl.col("total_time").cast(pl.Float64, strict=False) / 3600.0

    table = table.with_columns([
        pl.when(equipment_total > 0)
        .then(pl.col("no_eq") / equipment_total * 100.0)
        .otherwise(None)
        .alias("pct_unsignalised"),
        footage_hours.alias("hours"),
    ])
    table = table.with_columns(
        pl.when(pl.col("hours") > 0)
        .then(pl.col("crossings") / pl.col("hours"))
        .otherwise(None)
        .alias("crossings_per_hour")
    )

    # Any NaN produced by an upstream division is treated as missing so the
    # statistical routines can drop it cleanly.
    float_columns = [
        column
        for column, dtype in zip(table.columns, table.dtypes)
        if dtype in (pl.Float64, pl.Float32)
    ]
    table = table.with_columns([
        pl.when(pl.col(column).is_nan())
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
        for column in float_columns
    ])

    logger.info(
        "[structure] Built city table: {} cities, {} with a crossing speed, "
        "{} with an initiation time.",
        table.height,
        table["speed"].drop_nulls().len(),
        table["time"].drop_nulls().len(),
    )
    return table


# ---------------------------------------------------------------------
# Cluster tendency
# ---------------------------------------------------------------------

def _kmeans(
    points: np.ndarray,
    n_clusters: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Minimal k-means++ implementation returning labels, centres, inertia."""
    generator = np.random.default_rng(seed)
    n_points = len(points)
    centres = [points[generator.integers(n_points)]]

    for _ in range(n_clusters - 1):
        distances = np.min(
            ((points[:, None, :] - np.asarray(centres)[None]) ** 2).sum(axis=2),
            axis=1,
        )
        total = distances.sum()
        if total <= 0:
            centres.append(points[generator.integers(n_points)])
            continue
        centres.append(points[generator.choice(n_points, p=distances / total)])

    centre_array = np.asarray(centres, dtype=float)
    labels = np.zeros(n_points, dtype=int)

    for _ in range(200):
        labels = np.argmin(
            ((points[:, None, :] - centre_array[None]) ** 2).sum(axis=2),
            axis=1,
        )
        updated = np.asarray([
            points[labels == index].mean(axis=0)
            if (labels == index).any()
            else centre_array[index]
            for index in range(n_clusters)
        ])
        if np.allclose(updated, centre_array):
            break
        centre_array = updated

    inertia = float(((points - centre_array[labels]) ** 2).sum())
    return labels, centre_array, inertia


def _best_kmeans(
    points: np.ndarray,
    n_clusters: int,
    restarts: int,
) -> Tuple[np.ndarray, float]:
    """Return the lowest-inertia k-means solution over several restarts."""
    best_labels: Optional[np.ndarray] = None
    best_inertia = math.inf
    for seed in range(restarts):
        labels, _, inertia = _kmeans(points, n_clusters, seed)
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    assert best_labels is not None
    return best_labels, best_inertia


def _silhouette(points: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient over all points."""
    unique = set(labels.tolist())
    if len(unique) < 2:
        return 0.0

    distances = np.sqrt(
        ((points[:, None, :] - points[None]) ** 2).sum(axis=2)
    )
    scores: List[float] = []
    for index in range(len(points)):
        same = labels == labels[index]
        same[index] = False
        if not same.any():
            scores.append(0.0)
            continue
        cohesion = distances[index, same].mean()
        separation = min(
            distances[index, labels == other].mean()
            for other in unique
            if other != labels[index]
        )
        scores.append(
            (separation - cohesion) / max(cohesion, separation)
        )
    return float(np.mean(scores))


def _hopkins(points: np.ndarray, seed: int, repeats: int = 50) -> float:
    """Hopkins statistic averaged over several random subsamples."""
    n_points, n_dimensions = points.shape
    sample_size = max(5, int(0.1 * n_points))
    if sample_size >= n_points:
        return float("nan")

    lower, upper = points.min(axis=0), points.max(axis=0)
    values: List[float] = []

    for offset in range(repeats):
        generator = np.random.default_rng(seed + offset)
        chosen = generator.choice(n_points, sample_size, replace=False)
        uniform = generator.uniform(
            lower,
            upper,
            size=(sample_size, n_dimensions),
        )

        uniform_distance = np.sqrt(
            ((uniform[:, None, :] - points[None]) ** 2).sum(axis=2)
        ).min(axis=1)
        # The nearest neighbour of an observed point is itself, so take the
        # second smallest distance.
        observed_distance = np.sort(
            np.sqrt(
                ((points[chosen][:, None, :] - points[None]) ** 2).sum(axis=2)
            ),
            axis=1,
        )[:, 1]

        denominator = uniform_distance.sum() + observed_distance.sum()
        if denominator > 0:
            values.append(float(uniform_distance.sum() / denominator))

    return float(np.mean(values)) if values else float("nan")


def _gap_statistic(
    points: np.ndarray,
    max_clusters: int,
    reference_sets: int,
    restarts: int,
    seed: int,
) -> Dict[str, Any]:
    """Gap statistic against a PCA-aligned uniform reference distribution."""
    n_points, n_dimensions = points.shape
    centred = points - points.mean(axis=0)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ components.T
    lower, upper = projected.min(axis=0), projected.max(axis=0)
    generator = np.random.default_rng(seed)

    def dispersion(sample: np.ndarray, n_clusters: int) -> float:
        if n_clusters == 1:
            return float(((sample - sample.mean(axis=0)) ** 2).sum())
        return _best_kmeans(sample, n_clusters, restarts)[1]

    rows: List[Dict[str, float]] = []
    for n_clusters in range(1, max_clusters + 1):
        observed = math.log(max(dispersion(points, n_clusters), 1e-12))
        reference = []
        for _ in range(reference_sets):
            uniform = generator.uniform(
                lower,
                upper,
                size=(n_points, n_dimensions),
            ) @ components + points.mean(axis=0)
            reference.append(
                math.log(max(dispersion(uniform, n_clusters), 1e-12))
            )
        reference_array = np.asarray(reference, dtype=float)
        rows.append({
            "k": float(n_clusters),
            "log_dispersion": observed,
            "reference_log_dispersion": float(reference_array.mean()),
            "gap": float(reference_array.mean() - observed),
            "standard_error": float(
                reference_array.std() * math.sqrt(1.0 + 1.0 / reference_sets)
            ),
        })

    # Tibshirani's rule: the smallest k whose gap is not bettered by k+1
    # beyond one standard error.
    optimal = len(rows)
    for index in range(len(rows) - 1):
        if rows[index]["gap"] >= rows[index + 1]["gap"] - rows[index + 1]["standard_error"]:
            optimal = index + 1
            break

    # The first-crossing rule assumes the gap curve turns over somewhere in
    # the searched range. When the gap is still largest at the last k there is
    # no interior maximum: observed dispersion falls no faster with k than it
    # does for uniform noise, which is itself the signature of an absence of
    # clusters. In that regime the crossing point lands on Monte-Carlo noise
    # and flips between runs, so it is reported as "no elbow" instead of being
    # presented as an estimate of k.
    gaps = [row["gap"] for row in rows]
    argmax_k = int(rows[int(np.argmax(gaps))]["k"])
    no_interior_maximum = argmax_k >= len(rows)
    if no_interior_maximum:
        optimal = 1

    return {
        "rows": rows,
        "optimal_k": int(optimal),
        "argmax_k": argmax_k,
        "no_interior_maximum": bool(no_interior_maximum),
    }


def cluster_tendency(
    city_table: pl.DataFrame,
    features: Sequence[str] = CLUSTER_FEATURES,
    max_clusters: int = 6,
    reference_sets: int = 40,
    restarts: int = 25,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """Test whether cities form discrete groups, and log the verdict."""
    usable = [name for name in features if name in city_table.columns]
    subset = city_table.select(list(usable)).drop_nulls()

    logger.info("\n=== [structure] A) Cluster tendency of city behaviour ===")

    if subset.height < 20 or len(usable) < 2:
        logger.warning(
            "[structure] Only {} complete cities across {} features; "
            "cluster tendency was not tested.",
            subset.height,
            len(usable),
        )
        return {"status": "insufficient_data", "n_cities": subset.height}

    matrix = subset.to_numpy().astype(float)
    ranked = _rank_z(matrix)

    hopkins = _hopkins(ranked, seed)

    covariance = np.cov(ranked, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    explained = eigenvalues / eigenvalues.sum()
    scores = ranked @ eigenvectors

    partitions: List[Dict[str, Any]] = []
    for n_clusters in range(2, max_clusters + 1):
        labels, inertia = _best_kmeans(ranked, n_clusters, restarts)
        partitions.append({
            "k": n_clusters,
            "silhouette": _silhouette(ranked, labels),
            "inertia": inertia,
            "sizes": np.bincount(labels, minlength=n_clusters).tolist(),
        })

    gap = _gap_statistic(
        ranked,
        max_clusters=max_clusters,
        reference_sets=reference_sets,
        restarts=max(10, restarts // 2),
        seed=seed,
    )

    best_partition = max(partitions, key=lambda row: row["silhouette"])
    clustered = (
        gap["optimal_k"] > 1
        and best_partition["silhouette"] >= 0.50
        and hopkins >= 0.75
    )

    logger.info(
        "[structure] Features: {}; complete cities: {}.",
        ", ".join(usable),
        subset.height,
    )
    logger.info(
        "[structure] Hopkins statistic = {:.3f} "
        "(0.50 = indistinguishable from uniform; >=0.75 = clustered).",
        hopkins,
    )
    logger.info(
        "[structure] PCA explained variance: {}.",
        ", ".join(f"PC{i + 1}={value:.3f}" for i, value in enumerate(explained)),
    )
    logger.info("[structure] PCA loadings:")
    for index, name in enumerate(usable):
        logger.info(
            "[structure]   {:22s} {}",
            name,
            ", ".join(
                f"PC{axis + 1}={eigenvectors[index, axis]:+.3f}"
                for axis in range(len(usable))
            ),
        )
    logger.info("[structure] k-means partitions in rank space:")
    for row in partitions:
        logger.info(
            "[structure]   k={}  silhouette={:.3f}  inertia={:.1f}  sizes={}",
            row["k"],
            row["silhouette"],
            row["inertia"],
            row["sizes"],
        )
    logger.info(
        "[structure] Gap statistic ({} uniform reference sets):",
        reference_sets,
    )
    for row in gap["rows"]:
        logger.info(
            "[structure]   k={:.0f}  log(W)={:.4f}  E*[log(W)]={:.4f}  "
            "gap={:.4f}  s_k={:.4f}",
            row["k"],
            row["log_dispersion"],
            row["reference_log_dispersion"],
            row["gap"],
            row["standard_error"],
        )
    if gap["no_interior_maximum"]:
        logger.info(
            "[structure] The gap is still largest at k={}, the top of the "
            "searched range, so the curve never turns over and the "
            "Tibshirani first-crossing rule has no elbow to find: "
            "within-cluster dispersion falls no faster than for uniform "
            "noise. Reported as k = 1.",
            max_clusters,
        )
    else:
        logger.info(
            "[structure] Tibshirani criterion selects k = {} "
            "(gap is maximised at k = {}).",
            gap["optimal_k"],
            gap["argmax_k"],
        )

    if clustered:
        logger.info(
            "[structure] VERDICT: discrete cluster structure is supported "
            "(k={}, silhouette={:.3f}, Hopkins={:.3f}).",
            best_partition["k"],
            best_partition["silhouette"],
            hopkins,
        )
    else:
        logger.info(
            "[structure] VERDICT: no discrete cluster structure. {} The best "
            "silhouette is only {:.3f} (k={}), and Hopkins={:.3f} is close to "
            "the uniform value of 0.50. Cities should be described as a "
            "continuous gradient, not as a typology; any k-means partition "
            "drawn on these data cuts a continuum rather than separating "
            "groups.",
            "The gap curve has no interior maximum (k = 1)."
            if gap["no_interior_maximum"]
            else f"The gap statistic selects k={gap['optimal_k']}.",
            best_partition["silhouette"],
            best_partition["k"],
            hopkins,
        )

    return {
        "status": "complete",
        "features": list(usable),
        "n_cities": subset.height,
        "hopkins": hopkins,
        "explained_variance": explained.tolist(),
        "loadings": eigenvectors.tolist(),
        "scores": scores,
        "partitions": partitions,
        "gap": gap,
        "best_silhouette": best_partition["silhouette"],
        "best_silhouette_k": best_partition["k"],
        "clustered": bool(clustered),
        "row_index": subset,
    }


# ---------------------------------------------------------------------
# Gradient correlations
# ---------------------------------------------------------------------

def gradient_correlations(
    city_table: pl.DataFrame,
    predictors: Sequence[str] = GRADIENT_PREDICTORS,
    metrics: Sequence[Tuple[str, str]] = (
        ("speed", "crossing speed"),
        ("time", "crossing initiation time"),
    ),
) -> Dict[str, Any]:
    """Spearman-screen contextual predictors against the headline metrics."""
    logger.info("\n=== [structure] B) Gradient correlations (Spearman) ===")

    results: Dict[str, List[Dict[str, Any]]] = {}
    for metric, label in metrics:
        if metric not in city_table.columns:
            continue
        rows: List[Dict[str, Any]] = []
        logger.info("[structure] {}:", label)
        for predictor in predictors:
            if predictor not in city_table.columns:
                continue
            pair = city_table.select([predictor, metric]).drop_nulls()
            if pair.height < 10:
                continue
            rho, p_value = stats.spearmanr(pair[predictor], pair[metric])
            if not math.isfinite(rho):
                continue
            rows.append({
                "predictor": predictor,
                "rho": float(rho),
                "p_value": float(p_value),
                "n": int(pair.height),
            })
            logger.info(
                "[structure]   {:22s} rho={:+.3f}  p={:.4f}{:<3s} n={}",
                predictor,
                rho,
                p_value,
                _significance(p_value),
                pair.height,
            )
        rows.sort(key=lambda row: abs(row["rho"]), reverse=True)
        results[metric] = rows

    pair = city_table.select(["speed", "time"]).drop_nulls()
    metric_pair: Dict[str, Any] = {}
    if pair.height >= 10:
        rho, p_value = stats.spearmanr(pair["speed"], pair["time"])
        metric_pair = {
            "rho": float(rho),
            "p_value": float(p_value),
            "n": int(pair.height),
        }
        logger.info(
            "[structure] Crossing speed vs initiation time: rho={:+.3f} "
            "p={:.4f}{} n={}. {}",
            rho,
            p_value,
            _significance(p_value),
            pair.height,
            "The two metrics are statistically independent, so they capture "
            "distinct behavioural dimensions."
            if p_value >= 0.05
            else "The two metrics share variance.",
        )

    return {"by_metric": results, "metric_pair": metric_pair}


def continent_contrasts(
    city_table: pl.DataFrame,
    metrics: Sequence[Tuple[str, str]] = (
        ("speed", "crossing speed"),
        ("time", "crossing initiation time"),
        ("pct_unsignalised", "crossings without traffic signals (%)"),
    ),
    minimum_group: int = 5,
) -> Dict[str, Any]:
    """Describe and test each metric across continents."""
    logger.info("\n=== [structure] C) Continental contrasts ===")

    if "continent" not in city_table.columns:
        return {}

    summaries: Dict[str, Any] = {}
    available = [name for name, _ in metrics if name in city_table.columns]

    table = (
        city_table
        .filter(pl.col("continent").is_not_null() & (pl.col("continent") != ""))
        .group_by("continent")
        .agg(
            [pl.len().alias("n")]
            + [
                expression
                for name in available
                for expression in (
                    pl.col(name).mean().alias(f"{name}_mean"),
                    pl.col(name).std().alias(f"{name}_sd"),
                    pl.col(name).median().alias(f"{name}_median"),
                )
            ]
        )
        .sort("n", descending=True)
    )
    for row in table.iter_rows(named=True):
        logger.info(
            "[structure]   {:16s} n={:3d}  {}",
            str(row["continent"]),
            int(row["n"]),
            "  ".join(
                f"{name}={row[f'{name}_mean']:.3f}+-{row[f'{name}_sd']:.3f}"
                if row[f"{name}_mean"] is not None
                and row[f"{name}_sd"] is not None
                else f"{name}=NA"
                for name in available
            ),
        )

    for name, label in metrics:
        if name not in city_table.columns:
            continue
        groups = [
            _finite(group[name].drop_nulls().to_list())
            for _, group in city_table.group_by("continent")
        ]
        groups = [group for group in groups if len(group) >= minimum_group]
        if len(groups) < 2:
            continue
        statistic, p_value = stats.kruskal(*groups)
        summaries[name] = {
            "H": float(statistic),
            "p_value": float(p_value),
            "groups": len(groups),
        }
        logger.info(
            "[structure] Kruskal-Wallis {:38s} H={:.2f}  p={:.4g}{}",
            label,
            statistic,
            p_value,
            _significance(p_value),
        )

    return {"summary_table": table, "tests": summaries}


# ---------------------------------------------------------------------
# Reliability, support and measurement artefacts
# ---------------------------------------------------------------------

def _tracks_per_city(nested: dict, scale: float = 1.0) -> Dict[str, List[float]]:
    """Flatten ``{locality_condition: {video: {track: value}}}`` per city."""
    output: Dict[str, List[float]] = {}
    for key, videos in (nested or {}).items():
        parsed = _split_locality_condition(key)
        if parsed is None:
            continue
        base, _ = parsed
        if not isinstance(videos, dict):
            continue
        for tracks in videos.values():
            if not isinstance(tracks, dict):
                continue
            for value in tracks.values():
                try:
                    numeric = float(value) * scale
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric):
                    output.setdefault(base, []).append(numeric)
    return output


def split_half_reliability(
    all_speed: dict,
    all_time: dict,
    checks_per_second: float = 3.0,
    thresholds: Sequence[int] = SUPPORT_THRESHOLDS,
    repeats: int = 200,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """Estimate how repeatable each city estimate is.

    Every city's tracks are shuffled and split into halves; the Spearman
    correlation of the two half-means across cities, corrected with the
    Spearman-Brown formula, estimates the reliability of the full estimate.
    Reliability caps the correlation any predictor can show with the metric,
    so a low value means city rankings should not be interpreted.
    """
    logger.info("\n=== [structure] D) Split-half reliability of city estimates ===")

    time_scale = 1.0 / checks_per_second if checks_per_second else 1.0
    sources = [
        ("speed", "crossing speed", _tracks_per_city(all_speed)),
        ("time", "crossing initiation time (s)", _tracks_per_city(all_time, time_scale)),
    ]

    results: Dict[str, Any] = {}
    for metric, label, per_city in sources:
        if not per_city:
            continue
        rows: List[Dict[str, Any]] = []
        logger.info("[structure] {}:", label)
        for threshold in thresholds:
            eligible = {
                city: values
                for city, values in per_city.items()
                if len(values) >= max(2, threshold)
            }
            if len(eligible) < 10:
                continue

            corrected: List[float] = []
            for repeat in range(repeats):
                generator = np.random.default_rng(seed + repeat)
                first, second = [], []
                for values in eligible.values():
                    sample = np.asarray(values, dtype=float)
                    generator.shuffle(sample)
                    half = len(sample) // 2
                    first.append(float(sample[:half].mean()))
                    second.append(float(sample[half:2 * half].mean()))
                rho = stats.spearmanr(first, second).statistic
                if math.isfinite(rho) and rho > -1.0:
                    corrected.append(2.0 * rho / (1.0 + rho))

            if not corrected:
                continue
            mean_reliability = float(np.mean(corrected))
            rows.append({
                "threshold": int(threshold),
                "n_cities": len(eligible),
                "reliability": mean_reliability,
                "sd": float(np.std(corrected)),
            })
            logger.info(
                "[structure]   >= {:3d} tracks/city -> {:3d} cities, "
                "split-half reliability = {:.3f} (SD {:.3f}); "
                "maximum observable correlation ~ {:.2f}",
                threshold,
                len(eligible),
                mean_reliability,
                float(np.std(corrected)),
                math.sqrt(max(mean_reliability, 0.0)),
            )
        results[metric] = rows

        if rows:
            headline = rows[0]["reliability"]
            if headline < 0.60:
                logger.warning(
                    "[structure] {} has low reliability ({:.3f}) at the "
                    "unrestricted threshold. City-level rankings of this "
                    "metric are dominated by sampling noise and should be "
                    "reported only at an aggregated level or with an "
                    "explicit minimum-track rule.",
                    label,
                    headline,
                )

        support = sorted(len(values) for values in per_city.values())
        if support:
            logger.info(
                "[structure]   track support per city: median={}, "
                "p25={}, p75={}, min={}, max={}",
                int(np.median(support)),
                int(np.percentile(support, 25)),
                int(np.percentile(support, 75)),
                support[0],
                support[-1],
            )

    return results


def initiation_time_floor(
    all_time: dict,
    checks_per_second: float = 3.0,
    minimum_stable_samples: int = 3,
) -> Dict[str, Any]:
    """Quantify the floor and quantisation of the initiation-time metric.

    ``time_to_start_cross`` accepts a track only after ``minimum_stable_samples``
    consecutive stable samples and reports the count divided by
    ``check_per_sec_time``. The metric therefore cannot go below
    ``minimum_stable_samples / checks_per_second`` seconds and is quantised in
    steps of ``1 / checks_per_second``.
    """
    logger.info("\n=== [structure] E) Initiation-time floor and quantisation ===")

    samples = _finite([
        value
        for videos in (all_time or {}).values()
        if isinstance(videos, dict)
        for tracks in videos.values()
        if isinstance(tracks, dict)
        for value in tracks.values()
    ])
    if samples.size == 0:
        return {}

    floor_seconds = minimum_stable_samples / checks_per_second
    step_seconds = 1.0 / checks_per_second
    at_floor = float(np.mean(samples <= minimum_stable_samples))

    unique, counts = np.unique(samples, return_counts=True)
    top = sorted(zip(unique, counts), key=lambda item: -item[1])[:8]

    logger.info(
        "[structure] {} tracks, {} distinct values; the metric is quantised "
        "in steps of {:.3f} s and cannot fall below {:.3f} s.",
        samples.size,
        unique.size,
        step_seconds,
        floor_seconds,
    )
    logger.info("[structure] Most frequent values:")
    for value, count in top:
        logger.info(
            "[structure]   {:5.0f} stable samples = {:5.2f} s  "
            "{:7d} tracks ({:5.2f}%)",
            value,
            value / checks_per_second,
            int(count),
            100.0 * count / samples.size,
        )
    logger.info(
        "[structure] {:.1f}% of all tracks sit exactly on the {:.2f} s floor. "
        "This is a property of the estimator, not of pedestrian behaviour, "
        "and must be disclosed alongside any initiation-time distribution.",
        100.0 * at_floor,
        floor_seconds,
    )

    return {
        "n_tracks": int(samples.size),
        "distinct_values": int(unique.size),
        "floor_seconds": floor_seconds,
        "step_seconds": step_seconds,
        "fraction_at_floor": at_floor,
        "seconds": samples / checks_per_second,
        "top_values": [
            {
                "samples": float(value),
                "seconds": float(value) / checks_per_second,
                "count": int(count),
                "share": float(count) / samples.size,
            }
            for value, count in top
        ],
    }


def crossing_speed_scale(all_speed: dict) -> Dict[str, Any]:
    """Describe the per-track crossing speed and state its normalisation."""
    logger.info("\n=== [structure] F) Scale of the crossing speed metric ===")

    values = _finite([
        value
        for videos in (all_speed or {}).values()
        if isinstance(videos, dict)
        for tracks in videos.values()
        if isinstance(tracks, dict)
        for value in tracks.values()
    ])
    if values.size == 0:
        return {}

    metric_unit = "m/s" if __import__("os").environ.get(
        "CROWD_CROSSING_SPEED_UNIT"
    ) == "m/s" else "relative index"

    summary = {
        "n_tracks": int(values.size),
        "unit": metric_unit,
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "values": values,
    }
    logger.info(
        "[structure] {} tracks; unit = {}; mean={:.3f} median={:.3f} "
        "p05={:.3f} p95={:.3f}.",
        summary["n_tracks"],
        metric_unit,
        summary["mean"],
        summary["median"],
        summary["p05"],
        summary["p95"],
    )
    if metric_unit != "m/s":
        logger.info(
            "[structure] The Waymo metric speed model did not qualify for "
            "this run, so values are a within-video relative index: each "
            "track's motion proxy divided by the median proxy of every "
            "eligible person track in the same clip, not only the "
            "crossing ones. A value of 1.0 means 'as fast as a typical "
            "detected pedestrian in that same video', so crossing tracks "
            "sit below 1.0 as a rule. Between-city "
            "comparison assumes the within-clip reference populations are "
            "comparable, and the values must never be reported as m/s."
        )
    return summary


def day_night_contrast(
    city_table: pl.DataFrame,
    thresholds: Sequence[int] = (1, 10, 25),
) -> Dict[str, Any]:
    """Compare day and night per city, sweeping the minimum track support.

    Night-time footage is far thinner than daytime footage, so an effect that
    only appears at the unrestricted threshold is unlikely to be robust.
    """
    logger.info("\n=== [structure] G) Day versus night, by track support ===")

    results: Dict[str, List[Dict[str, Any]]] = {}
    metrics = [
        ("speed", "n_speed", "crossing speed"),
        ("time", "n_time", "crossing initiation time (s)"),
    ]

    for metric, count_prefix, label in metrics:
        day_column, night_column = f"{metric}_0", f"{metric}_1"
        if day_column not in city_table.columns or night_column not in city_table.columns:
            continue
        rows: List[Dict[str, Any]] = []
        logger.info("[structure] {}:", label)
        for threshold in thresholds:
            subset = city_table
            for condition in ("0", "1"):
                column = f"{count_prefix}_{condition}"
                if column in city_table.columns:
                    subset = subset.filter(
                        pl.col(column).fill_null(0) >= threshold
                    )
            subset = subset.select([day_column, night_column]).drop_nulls()
            if subset.height < 6:
                logger.info(
                    "[structure]   >= {:2d} tracks/condition -> only {} "
                    "paired cities; not tested.",
                    threshold,
                    subset.height,
                )
                continue
            day = subset[day_column].to_numpy().astype(float)
            night = subset[night_column].to_numpy().astype(float)
            try:
                p_value = float(stats.wilcoxon(day, night).pvalue)
            except ValueError:
                continue
            rows.append({
                "threshold": int(threshold),
                "n": int(subset.height),
                "day": float(day.mean()),
                "night": float(night.mean()),
                "delta": float(night.mean() - day.mean()),
                "p_value": p_value,
            })
            logger.info(
                "[structure]   >= {:2d} tracks/condition  n={:3d}  "
                "day={:.3f}  night={:.3f}  delta={:+.3f}  "
                "Wilcoxon p={:.4g}{}",
                threshold,
                subset.height,
                day.mean(),
                night.mean(),
                night.mean() - day.mean(),
                p_value,
                _significance(p_value),
            )
        results[metric] = rows

        significant = [row for row in rows if row["p_value"] < 0.05]
        if rows and significant and len(significant) < len(rows):
            logger.warning(
                "[structure] The day/night difference in {} is not stable "
                "across support thresholds. Report the support-gated result "
                "and state the night-time track counts.",
                label,
            )

    return results


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def analyse_structure(
    df_mapping: pl.DataFrame,
    avg_speed_locality: dict,
    avg_time_locality: dict,
    pedestrian_cross_locality: dict,
    crossings_with_traffic_equipment_locality: dict,
    crossings_without_traffic_equipment_locality: dict,
    all_speed: dict,
    all_time: dict,
    all_speed_locality: dict,
    all_time_locality: dict,
    checks_per_second: float = 3.0,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """Run every structure diagnostic and log the results."""
    logger.info("\n=== [structure] City-level structure diagnostics ===")

    city_table = build_city_table(
        df_mapping=df_mapping,
        avg_speed_locality=avg_speed_locality,
        avg_time_locality=avg_time_locality,
        pedestrian_cross_locality=pedestrian_cross_locality,
        crossings_with_traffic_equipment_locality=(
            crossings_with_traffic_equipment_locality
        ),
        crossings_without_traffic_equipment_locality=(
            crossings_without_traffic_equipment_locality
        ),
        all_speed_locality=all_speed_locality,
        all_time_locality=all_time_locality,
    )
    if city_table.height == 0:
        return {"status": "no_data"}

    return {
        "status": "complete",
        "city_table": city_table,
        "cluster_tendency": cluster_tendency(city_table, seed=seed),
        "gradient": gradient_correlations(city_table),
        "continents": continent_contrasts(city_table),
        "reliability": split_half_reliability(
            all_speed,
            all_time,
            checks_per_second=checks_per_second,
            seed=seed,
        ),
        "time_floor": initiation_time_floor(
            all_time,
            checks_per_second=checks_per_second,
        ),
        "motion_scale": crossing_speed_scale(all_speed),
        "day_night": day_night_contrast(city_table),
    }
