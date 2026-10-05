#!/usr/bin/env python3
# network_parser/hierarchy_separation_visualization.py
"""
Hierarchy + sample/strain separation visualizations for NetworkParser.

Purpose
-------
Write a combined interpretation report that shows both:

1. the user-defined metadata hierarchy, for example Country -> Lineage -> Pheno;
2. how the retained samples/strains separate in the final selected genomic
   feature space used for model training.

This module is descriptive only. It does not change statistical filtering,
model training, decision-tree construction, bootstrapping, confidence scoring,
or query inference.

Important wording
-----------------
The dendrograms produced here are feature-space hierarchical clustering views.
They should not be described as phylogenetic trees.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    plt = None  # type: ignore
    HAVE_MATPLOTLIB = False

try:
    from sklearn.metrics import silhouette_score

    HAVE_SKLEARN_SILHOUETTE = True
except ImportError:  # pragma: no cover
    silhouette_score = None  # type: ignore
    HAVE_SKLEARN_SILHOUETTE = False

try:
    from network_parser.result_visualization import (
        build_aligned_metadata_hierarchy_frame,
        parse_visualization_label_columns,
        write_sample_hierarchy_visualizations,
    )
except ImportError:  # pragma: no cover - supports direct source-tree execution
    from result_visualization import (  # type: ignore
        build_aligned_metadata_hierarchy_frame,
        parse_visualization_label_columns,
        write_sample_hierarchy_visualizations,
    )

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _coerce_binary_matrix(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X_local = X.copy()
    X_local.index = X_local.index.astype(str)
    # Preserve non-callable NaN for pairwise-complete distances.
    X_num = X_local.apply(pd.to_numeric, errors="coerce")
    # Drop samples/features that have no evidence at all. Remaining NaNs are
    # handled pairwise in ``build_feature_space_distances``.
    X_num = X_num.dropna(axis=0, how="all").dropna(axis=1, how="all")
    row_ok = X_num.notna().mean(axis=1) >= 0.5
    X_num = X_num.loc[row_ok]
    return X_num.where(X_num.isna(), (X_num > 0).astype(float))


def _leaf_font_size(n_samples: int) -> float:
    if n_samples <= 25:
        return 9.0
    if n_samples <= 60:
        return 8.0
    if n_samples <= 120:
        return 7.0
    return 6.0


def _compose_leaf_label(row: pd.Series, label_columns: Sequence[str]) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    parts = [sample_id]
    for col in label_columns:
        if col not in row.index:
            continue
        value = str(row.get(col, "")).strip()
        if value:
            parts.append(f"{col}={value}")
    return " | ".join(parts)


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
        if np.isfinite(f):
            return f
    except Exception:
        pass
    return None


def _prepare_leaf_metadata(
    *,
    meta: pd.DataFrame,
    sample_ids: Sequence[Any],
    hierarchy_columns: Sequence[str],
    include_missing: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    hierarchy_df, alignment_summary = build_aligned_metadata_hierarchy_frame(
        meta=meta,
        sample_ids=sample_ids,
        hierarchy_columns=hierarchy_columns,
        include_missing=include_missing,
    )
    visualized_columns = list(alignment_summary.get("visualized_columns", []))

    ordered_ids = [str(s) for s in sample_ids]
    hierarchy_df = (
        hierarchy_df.drop_duplicates(subset=["sample_id"], keep="first")
        .set_index("sample_id", drop=False)
        .reindex(ordered_ids)
    )

    for sample_id in ordered_ids:
        if pd.isna(hierarchy_df.loc[sample_id, "sample_id"]):
            hierarchy_df.loc[sample_id, "sample_id"] = sample_id

    hierarchy_df["leaf_label_full_hierarchy"] = hierarchy_df.apply(
        lambda row: _compose_leaf_label(row, visualized_columns),
        axis=1,
    )

    return hierarchy_df.reset_index(drop=True), alignment_summary


# -----------------------------------------------------------------------------
# Distances + separation metrics
# -----------------------------------------------------------------------------


def build_feature_space_distances(
    X: pd.DataFrame,
    metric: str = "jaccard",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (square distance dataframe, condensed distance vector, linkage matrix input-ready matrix)."""
    metric = str(metric).lower().strip()
    if metric not in {"jaccard", "hamming"}:
        raise ValueError("sample separation metric must be 'jaccard' or 'hamming'")

    X_bin = _coerce_binary_matrix(X)
    sample_ids = X_bin.index.astype(str).tolist()

    if X_bin.shape[0] < 2:
        raise ValueError(
            "At least two samples are required for sample separation distances."
        )

    values = X_bin.to_numpy(dtype=float)
    pairwise: List[float] = []
    for left in range(values.shape[0] - 1):
        for right in range(left + 1, values.shape[0]):
            jointly_callable = np.isfinite(values[left]) & np.isfinite(values[right])
            if not jointly_callable.any():
                pairwise.append(float("nan"))
                continue
            a = values[left, jointly_callable].astype(bool)
            b = values[right, jointly_callable].astype(bool)
            if metric == "hamming":
                pairwise.append(float(np.mean(a != b)))
            else:
                union = np.logical_or(a, b)
                pairwise.append(
                    0.0
                    if not union.any()
                    else float(np.logical_and(a != b, union).sum() / union.sum())
                )
    condensed = np.asarray(pairwise, dtype=float)
    if not np.isfinite(condensed).all():
        raise ValueError(
            "At least one sample pair has no jointly callable selected features; "
            "refusing to treat missing states as reference for clustering."
        )
    square = squareform(condensed)
    distance_df = pd.DataFrame(square, index=sample_ids, columns=sample_ids)
    return distance_df, condensed, X_bin.to_numpy(dtype=float)


def _nearest_neighbor_agreement(
    distance_array: np.ndarray, labels: Sequence[str]
) -> Optional[float]:
    labels_arr = np.asarray([str(x) for x in labels], dtype=object)
    n = int(distance_array.shape[0])
    if n < 2 or len(set(labels_arr.tolist())) < 2:
        return None

    masked = distance_array.copy()
    np.fill_diagonal(masked, np.inf)
    nearest = np.argmin(masked, axis=1)
    agreement = labels_arr[nearest] == labels_arr
    return float(np.mean(agreement))


def _within_between_distance_summary(
    distance_array: np.ndarray,
    labels: Sequence[str],
) -> Dict[str, Optional[float]]:
    labels_arr = np.asarray([str(x) for x in labels], dtype=object)
    n = int(distance_array.shape[0])
    if n < 2 or len(set(labels_arr.tolist())) < 2:
        return {
            "mean_within_distance": None,
            "mean_between_distance": None,
            "between_within_distance_ratio": None,
        }

    tri_i, tri_j = np.triu_indices(n, k=1)
    d = distance_array[tri_i, tri_j]
    same = labels_arr[tri_i] == labels_arr[tri_j]

    within = d[same]
    between = d[~same]

    mean_within = float(np.mean(within)) if within.size else None
    mean_between = float(np.mean(between)) if between.size else None

    if mean_within is None or mean_within <= 0 or mean_between is None:
        ratio = None
    else:
        ratio = float(mean_between / mean_within)

    return {
        "mean_within_distance": mean_within,
        "mean_between_distance": mean_between,
        "between_within_distance_ratio": ratio,
    }


def _silhouette_from_precomputed(
    distance_array: np.ndarray,
    labels: Sequence[str],
) -> Optional[float]:
    if not HAVE_SKLEARN_SILHOUETTE:
        return None

    labels_arr = np.asarray([str(x) for x in labels], dtype=object)
    unique, counts = np.unique(labels_arr, return_counts=True)

    # silhouette_score requires at least two labels and fewer labels than samples.
    if unique.size < 2 or unique.size >= labels_arr.size:
        return None
    if np.any(counts < 2):
        return None

    try:
        value = silhouette_score(distance_array, labels_arr, metric="precomputed")
        return float(value) if np.isfinite(value) else None
    except Exception:
        return None


def build_per_level_separation_summary(
    *,
    distance_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    hierarchy_columns: Sequence[str],
    metric: str,
    linkage_method: str,
) -> pd.DataFrame:
    distance_array = distance_df.to_numpy(dtype=float, copy=True)
    rows: List[Dict[str, Any]] = []

    for level_index, col in enumerate(hierarchy_columns, start=1):
        if col not in labels_df.columns:
            continue

        labels = labels_df[col].fillna("[missing]").astype(str).tolist()
        counts = pd.Series(labels).value_counts(dropna=False)
        sep = _within_between_distance_summary(distance_array, labels)

        rows.append(
            {
                "level_index": int(level_index),
                "level_name": str(col),
                "n_samples": int(len(labels)),
                "n_labels": int(counts.shape[0]),
                "min_label_count": int(counts.min()) if not counts.empty else 0,
                "max_label_count": int(counts.max()) if not counts.empty else 0,
                "nearest_neighbor_label_agreement": _nearest_neighbor_agreement(
                    distance_array, labels
                ),
                "silhouette_score_precomputed_distance": _silhouette_from_precomputed(
                    distance_array, labels
                ),
                "mean_within_label_distance": sep["mean_within_distance"],
                "mean_between_label_distance": sep["mean_between_distance"],
                "between_within_distance_ratio": sep["between_within_distance_ratio"],
                "distance_metric": str(metric),
                "linkage_method": str(linkage_method),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Dendrogram rendering
# -----------------------------------------------------------------------------


def _write_dendrogram_image(
    *,
    linkage_matrix: np.ndarray,
    leaf_labels: Sequence[str],
    output_png: Path,
    output_svg: Path,
    title: str,
    distance_label: str,
    figure_width: float,
    row_height: float,
) -> Dict[str, Optional[str]]:
    if not HAVE_MATPLOTLIB:
        return {"png": None, "svg": None}

    n_samples = len(leaf_labels)
    fig_height = max(6.0, min(80.0, 2.0 + float(row_height) * max(1, n_samples)))

    fig, ax = plt.subplots(figsize=(float(figure_width), fig_height))
    dendrogram(
        linkage_matrix,
        labels=list(map(str, leaf_labels)),
        orientation="left",
        leaf_font_size=_leaf_font_size(n_samples),
        ax=ax,
        color_threshold=None,
    )
    ax.set_xlabel(distance_label)
    ax.set_ylabel("Samples / strains")
    fig.tight_layout()
    fig.text(0.5, -0.02, title, ha="center", va="top", fontsize=11)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)

    return {"png": str(output_png), "svg": str(output_svg)}


def write_sample_separation_dendrograms(
    *,
    X: pd.DataFrame,
    labels_df: pd.DataFrame,
    hierarchy_columns: Sequence[str],
    output_dir: Path,
    prefix: str = "final_sample",
    metric: str = "jaccard",
    linkage_method: str = "average",
    figure_width: float = 14.0,
    row_height: float = 0.24,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Write one full-hierarchy dendrogram plus one per metadata level."""
    out = ensure_dir(Path(output_dir))
    X_bin = _coerce_binary_matrix(X)

    if max_samples is not None:
        max_samples_int = int(max_samples)
        if max_samples_int > 0 and X_bin.shape[0] > max_samples_int:
            # Deterministic downsample for visual readability only.  The matrix
            # exported elsewhere remains complete.
            X_bin = X_bin.iloc[:max_samples_int, :].copy()
            labels_df = labels_df.iloc[:max_samples_int, :].copy()

    n_samples = int(X_bin.shape[0])
    n_features = int(X_bin.shape[1])

    if n_samples < 2:
        summary = {
            "status": "skipped",
            "reason": "Need at least two retained samples to build sample-separation dendrograms.",
            "n_samples": n_samples,
            "n_features": n_features,
        }
        write_json(summary, out / f"{prefix}_sample_separation_summary.json")
        return summary

    distance_df, condensed, _ = build_feature_space_distances(X_bin, metric=metric)
    Z = linkage(condensed, method=linkage_method)
    leaf_order = leaves_list(Z)

    ordered_sample_ids = X_bin.index.astype(str).tolist()
    labels_df = labels_df.copy()
    labels_df["sample_id"] = labels_df["sample_id"].astype(str)
    labels_df = (
        labels_df.set_index("sample_id", drop=False)
        .reindex(ordered_sample_ids)
        .reset_index(drop=True)
    )

    visualized_columns = [col for col in hierarchy_columns if col in labels_df.columns]
    labels_df["leaf_label_full_hierarchy"] = labels_df.apply(
        lambda row: _compose_leaf_label(row, visualized_columns),
        axis=1,
    )

    # Core tables
    labels_path = out / f"{prefix}_labels.tsv"
    distance_path = out / f"{prefix}_distance_matrix.tsv"
    linkage_path = out / f"{prefix}_linkage.tsv"
    leaf_order_path = out / f"{prefix}_leaf_order.tsv"

    labels_df.to_csv(labels_path, sep="\t", index=False)
    distance_df.to_csv(distance_path, sep="\t", index=True)

    linkage_df = pd.DataFrame(
        Z,
        columns=["left_cluster", "right_cluster", "distance", "n_leaves_merged"],
    )
    linkage_df.insert(0, "merge_step", range(1, int(linkage_df.shape[0]) + 1))
    linkage_df.to_csv(linkage_path, sep="\t", index=False)

    leaf_order_df = pd.DataFrame(
        {
            "leaf_rank": range(1, len(leaf_order) + 1),
            "sample_id": [ordered_sample_ids[i] for i in leaf_order],
            "leaf_label_full_hierarchy": [
                str(labels_df.iloc[i]["leaf_label_full_hierarchy"]) for i in leaf_order
            ],
        }
    )
    leaf_order_df.to_csv(leaf_order_path, sep="\t", index=False)

    # Full-hierarchy dendrogram.
    artifacts: Dict[str, Any] = {
        "labels_tsv": str(labels_path),
        "distance_matrix_tsv": str(distance_path),
        "linkage_tsv": str(linkage_path),
        "leaf_order_tsv": str(leaf_order_path),
        "full_hierarchy_dendrogram": {},
        "per_level_dendrograms": {},
    }

    full_png = out / f"{prefix}_dendrogram_full_hierarchy.png"
    full_svg = out / f"{prefix}_dendrogram_full_hierarchy.svg"
    artifacts["full_hierarchy_dendrogram"] = _write_dendrogram_image(
        linkage_matrix=Z,
        leaf_labels=labels_df["leaf_label_full_hierarchy"].astype(str).tolist(),
        output_png=full_png,
        output_svg=full_svg,
        title="Final retained samples: full metadata hierarchy",
        distance_label=f"{str(metric).capitalize()} distance",
        figure_width=figure_width,
        row_height=row_height,
    )

    # One dendrogram per hierarchy level.  These make it easier to see which
    # levels separate cleanly in the final selected feature space.
    for col in visualized_columns:
        safe_col = (
            "".join(
                ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(col)
            ).strip("_")
            or "level"
        )
        level_png = out / f"{prefix}_dendrogram_by_{safe_col}.png"
        level_svg = out / f"{prefix}_dendrogram_by_{safe_col}.svg"
        level_labels = (
            labels_df["sample_id"].astype(str)
            + " | "
            + str(col)
            + "="
            + labels_df[col].astype(str)
        ).tolist()
        artifacts["per_level_dendrograms"][str(col)] = _write_dendrogram_image(
            linkage_matrix=Z,
            leaf_labels=level_labels,
            output_png=level_png,
            output_svg=level_svg,
            title=f"Final retained samples annotated by {col}",
            distance_label=f"{str(metric).capitalize()} distance",
            figure_width=figure_width,
            row_height=row_height,
        )

    per_level_summary = build_per_level_separation_summary(
        distance_df=distance_df,
        labels_df=labels_df,
        hierarchy_columns=visualized_columns,
        metric=metric,
        linkage_method=linkage_method,
    )
    per_level_summary_path = out / f"{prefix}_per_level_separation_summary.tsv"
    per_level_summary.to_csv(per_level_summary_path, sep="\t", index=False)
    artifacts["per_level_separation_summary_tsv"] = str(per_level_summary_path)

    summary = {
        "status": "success" if HAVE_MATPLOTLIB else "partial_success",
        "n_samples": n_samples,
        "n_features": n_features,
        "visualized_columns": visualized_columns,
        "distance_metric": str(metric),
        "linkage_method": str(linkage_method),
        "matrix_type": "final_selected_binary_feature_matrix",
        "matplotlib_available": bool(HAVE_MATPLOTLIB),
        "sklearn_silhouette_available": bool(HAVE_SKLEARN_SILHOUETTE),
        "max_samples_for_visualization": int(max_samples)
        if max_samples is not None
        else None,
        "artifacts": artifacts,
    }

    write_json(summary, out / f"{prefix}_sample_separation_summary.json")

    logger.info(
        "Hierarchy sample-separation dendrograms written | samples=%d | features=%d | levels=%d | metric=%s",
        n_samples,
        n_features,
        len(visualized_columns),
        str(metric),
    )
    return summary


# -----------------------------------------------------------------------------
# HTML report
# -----------------------------------------------------------------------------


def _format_metric(value: Any) -> str:
    f = _safe_float(value)
    if f is None:
        return "NA"
    return f"{f:.4g}"


def _table_to_html(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df is None or df.empty:
        return "<p class='muted'>No records available.</p>"

    shown = df.head(max_rows).copy()
    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in shown.columns)
    rows: List[str] = []
    for _, row in shown.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.tolist())
        rows.append(f"<tr>{cells}</tr>")

    suffix = ""
    if int(df.shape[0]) > max_rows:
        suffix = (
            f"<p class='muted'>Showing first {max_rows} of {int(df.shape[0])} rows.</p>"
        )
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>{suffix}"


def write_combined_hierarchy_separation_html(
    *,
    output_path: Path,
    hierarchy_summary: Dict[str, Any],
    separation_summary: Dict[str, Any],
    per_level_summary: pd.DataFrame,
) -> None:
    visualized_columns = separation_summary.get("visualized_columns", [])
    hierarchy_text = " → ".join(html.escape(str(x)) for x in visualized_columns)

    full_png = (
        separation_summary.get("artifacts", {})
        .get("full_hierarchy_dendrogram", {})
        .get("png")
    )
    full_png_rel = Path(full_png).name if full_png else None

    per_level_cards: List[str] = []
    per_level_artifacts = (
        separation_summary.get("artifacts", {}).get("per_level_dendrograms", {})
        if isinstance(separation_summary.get("artifacts", {}), dict)
        else {}
    )

    for _, row in per_level_summary.iterrows() if not per_level_summary.empty else []:
        level = str(row.get("level_name", ""))
        png = None
        if isinstance(per_level_artifacts, dict):
            png = (
                per_level_artifacts.get(level, {}).get("png")
                if isinstance(per_level_artifacts.get(level, {}), dict)
                else None
            )
        img = (
            f"<img src='{html.escape(Path(png).name)}' alt='Dendrogram by {html.escape(level)}' />"
            if png
            else "<p class='muted'>Image unavailable.</p>"
        )
        per_level_cards.append(
            "<section class='card'>"
            f"<h2>Level {html.escape(str(row.get('level_index', '')))}: {html.escape(level)}</h2>"
            "<div class='metric-grid'>"
            f"<div><strong>{html.escape(str(row.get('n_labels', 'NA')))}</strong><span>labels</span></div>"
            f"<div><strong>{html.escape(_format_metric(row.get('nearest_neighbor_label_agreement')))}</strong><span>nearest-neighbour agreement</span></div>"
            f"<div><strong>{html.escape(_format_metric(row.get('silhouette_score_precomputed_distance')))}</strong><span>silhouette</span></div>"
            f"<div><strong>{html.escape(_format_metric(row.get('between_within_distance_ratio')))}</strong><span>between/within distance</span></div>"
            "</div>"
            f"{img}"
            "</section>"
        )

    full_img = (
        f"<img src='{html.escape(full_png_rel)}' alt='Full hierarchy dendrogram' />"
        if full_png_rel
        else "<p class='muted'>Full dendrogram image unavailable. Check the TSV/JSON artifacts.</p>"
    )

    content = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>NetworkParser hierarchy sample-separation report</title>
<style>
:root {{
  --bg: #f7f7fb;
  --card: #ffffff;
  --text: #1f2937;
  --muted: #6b7280;
  --line: #e5e7eb;
}}
body {{
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif;
  background: var(--bg);
  color: var(--text);
}}
header {{
  padding: 28px 34px;
  background: linear-gradient(135deg, #111827, #4b5563);
  color: white;
}}
main {{
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
}}
.card {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 18px;
  margin-bottom: 18px;
  box-shadow: 0 8px 24px rgba(31,41,55,0.06);
}}
.muted {{ color: var(--muted); }}
.metric-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}}
.metric-grid div {{
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px;
  background: #f9fafb;
}}
.metric-grid strong {{ display: block; font-size: 1.25rem; }}
.metric-grid span {{ color: var(--muted); }}
img {{ max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 12px; background: white; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }}
th {{ background: #f9fafb; }}
code {{ background: #f3f4f6; padding: 2px 5px; border-radius: 5px; }}
</style>
</head>
<body>
<header>
  <h1>NetworkParser hierarchy sample-separation report</h1>
  <p>Metadata hierarchy: <strong>{hierarchy_text}</strong></p>
  <p>This report links label structure to sample/strain clustering in the final selected genomic feature space.</p>
</header>
<main>
  <section class=\"card\">
    <h2>What this shows</h2>
    <p>The label hierarchy view describes how samples are distributed across metadata levels. The sample-separation dendrogram clusters retained samples using distances computed from the final selected binary genomic feature matrix. This is a feature-space clustering view, not a phylogenetic tree.</p>
  </section>

  <section class=\"card\">
    <h2>Full hierarchy dendrogram</h2>
    <p class=\"muted\">Leaves are labelled with sample identifiers and all selected hierarchy levels.</p>
    {full_img}
  </section>

  {''.join(per_level_cards)}

  <section class=\"card\">
    <h2>Per-level separation summary</h2>
    {_table_to_html(per_level_summary, max_rows=200)}
  </section>
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


# -----------------------------------------------------------------------------
# Public orchestration entry point
# -----------------------------------------------------------------------------


def write_hierarchy_sample_separation_report(
    *,
    X_selected: pd.DataFrame,
    meta: pd.DataFrame,
    sample_ids: Sequence[Any],
    hierarchy_columns: Sequence[str],
    output_dir: Path,
    include_missing: bool = True,
    max_categories_per_level: int = 30,
    metric: str = "jaccard",
    linkage_method: str = "average",
    figure_width: float = 14.0,
    row_height: float = 0.24,
    max_samples_for_dendrogram: Optional[int] = None,
) -> Dict[str, Any]:
    """Write both label-hierarchy and sample-separation artifacts."""
    root = ensure_dir(Path(output_dir))
    hierarchy_dir = ensure_dir(root / "hierarchy_labels")
    separation_dir = ensure_dir(root / "sample_separation")
    summary_dir = ensure_dir(root / "separation_summary")

    parsed_columns = parse_visualization_label_columns(hierarchy_columns)
    if not parsed_columns:
        raise ValueError("At least one hierarchy label column is required.")

    # Label hierarchy: count/path/edge/Graphviz/HTML view.
    hierarchy_summary = write_sample_hierarchy_visualizations(
        meta=meta,
        sample_ids=sample_ids,
        hierarchy_columns=parsed_columns,
        output_dir=hierarchy_dir,
        prefix="label_hierarchy",
        include_missing=include_missing,
        max_categories_per_level=max_categories_per_level,
    )
    visualized_columns = list(
        hierarchy_summary.get("visualized_columns", parsed_columns)
    )

    # Metadata aligned to selected matrix order for sample separation.
    X_selected = X_selected.copy()
    X_selected.index = X_selected.index.astype(str)
    selected_sample_ids = X_selected.index.astype(str).tolist()
    labels_df, label_alignment_summary = _prepare_leaf_metadata(
        meta=meta,
        sample_ids=selected_sample_ids,
        hierarchy_columns=visualized_columns,
        include_missing=include_missing,
    )

    # Sample/strain separation in final selected feature space.
    separation_summary = write_sample_separation_dendrograms(
        X=X_selected,
        labels_df=labels_df,
        hierarchy_columns=visualized_columns,
        output_dir=separation_dir,
        prefix="final_sample",
        metric=metric,
        linkage_method=linkage_method,
        figure_width=figure_width,
        row_height=row_height,
        max_samples=max_samples_for_dendrogram,
    )

    per_level_summary_path = (
        separation_summary.get("artifacts", {}).get("per_level_separation_summary_tsv")
        if isinstance(separation_summary.get("artifacts", {}), dict)
        else None
    )
    if per_level_summary_path and Path(per_level_summary_path).exists():
        per_level_summary = pd.read_csv(per_level_summary_path, sep="\t")
    else:
        per_level_summary = pd.DataFrame()

    # Copy the per-level summary into the top-level summary folder as well.
    top_per_level_path = summary_dir / "per_level_separation_summary.tsv"
    per_level_summary.to_csv(top_per_level_path, sep="\t", index=False)

    combined_html_path = root / "hierarchy_sample_separation_report.html"
    write_combined_hierarchy_separation_html(
        output_path=combined_html_path,
        hierarchy_summary=hierarchy_summary,
        separation_summary=separation_summary,
        per_level_summary=per_level_summary,
    )

    summary = {
        "status": "success"
        if separation_summary.get("status") in {"success", "partial_success"}
        else separation_summary.get("status", "unknown"),
        "description": "Combined label hierarchy and final selected-feature sample/strain separation report.",
        "visualized_columns": visualized_columns,
        "label_alignment": label_alignment_summary,
        "hierarchy_labels": hierarchy_summary,
        "sample_separation": separation_summary,
        "artifacts": {
            "combined_html": str(combined_html_path),
            "per_level_separation_summary_tsv": str(top_per_level_path),
            "hierarchy_labels_dir": str(hierarchy_dir),
            "sample_separation_dir": str(separation_dir),
            "summary_json": str(summary_dir / "hierarchy_visualization_summary.json"),
        },
    }

    write_json(summary, summary_dir / "hierarchy_visualization_summary.json")
    logger.info(
        "Hierarchy sample-separation report written | levels=%d | html=%s",
        len(visualized_columns),
        str(combined_html_path),
    )
    return summary
