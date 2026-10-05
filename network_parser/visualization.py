#!/usr/bin/env python3
# network_parser/visualization.py
"""
NetworkParser visualization utilities
=====================================

Purpose
-------
Generate publication-friendly, generic visual summaries from already completed
NetworkParser training and query outputs.

This module is deliberately downstream-only. It does not rerun statistical
filtering, feature-panel selection, model training, decision-tree construction,
bootstrapping, or confidence scoring. It reads existing artifacts and produces
figures/tables that help inspect:

    1. query predictions and query marker recovery/evidence
    2. final selected markers used at each trained level/node
    3. sample clustering over the final selected marker space

Main outputs
------------
Training registry visualizations:
    visualizations/training/registry_marker_level_summary.tsv
    visualizations/training/registry_marker_feature_membership.tsv
    visualizations/training/final_marker_counts_by_level.png/.svg
    visualizations/training/final_marker_overlap_jaccard_heatmap.png/.svg
    visualizations/training/final_marker_level_graph.graphml
    visualizations/training/final_marker_level_graph.png/.svg
    visualizations/training/per_level/<model_id>/selected_marker_dendrogram.png/.svg
    visualizations/training/per_level/<model_id>/selected_marker_heatmap.png/.svg

Query visualizations:
    visualizations/query/query_prediction_routes.tsv
    visualizations/query/query_prediction_route_counts.png/.svg
    visualizations/query/query_support_summary.png/.svg
    visualizations/query/query_marker_quality.png/.svg
    visualizations/query/query_route_graph.graphml
    visualizations/query/query_route_graph.png/.svg
    visualizations/query/query_selected_feature_dendrogram.png/.svg
    visualizations/query/query_selected_feature_heatmap.png/.svg

CLI examples
------------
python -m network_parser.visualization training \
  --registry path/to/hierarchical_model_registry.json \
  --output_dir path/to/results/visualizations

python -m network_parser.visualization query \
  --query_dir path/to/query_results \
  --output_dir path/to/query_results/visualizations

python -m network_parser.visualization all \
  --registry path/to/hierarchical_model_registry.json \
  --query_dir path/to/query_results \
  --output_dir path/to/visualizations
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None  # type: ignore

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
except Exception:  # pragma: no cover
    dendrogram = None  # type: ignore
    linkage = None  # type: ignore

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Small IO helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(payload: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)
        handle.write("\n")
    return path


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path_value: Optional[Any], base_dir: str | Path) -> Optional[Path]:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    path = Path(text)
    if path.is_absolute():
        return path if path.exists() else path
    candidate = Path(base_dir) / path
    if candidate.exists():
        return candidate
    return path


def read_csv_or_tsv(
    path: str | Path, *, index_col: Optional[int] = None
) -> pd.DataFrame:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".tsv") or suffixes.endswith(".txt"):
        return pd.read_csv(path, sep="\t", index_col=index_col)
    return pd.read_csv(path, index_col=index_col)


def _safe_token(value: Any, max_len: int = 80) -> str:
    raw = str(value)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    if not cleaned:
        cleaned = "item"
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip("_")


def _set_bottom_title(fig: plt.Figure, title: str | None) -> None:
    """Place the figure title below the axes (publication-style caption)."""
    if not title:
        return
    fig.text(0.5, -0.02, title, ha="center", va="top", fontsize=11)


def _save_figure(
    fig: plt.Figure,
    path: str | Path,
    *,
    svg: bool = True,
    title: str | None = None,
) -> Dict[str, str]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    _set_bottom_title(fig, title)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    outputs = {"png": str(path)}
    if svg:
        svg_path = path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight")
        outputs["svg"] = str(svg_path)
    plt.close(fig)
    return outputs


def _write_skip(path: str | Path, reason: str, **extra: Any) -> Dict[str, Any]:
    payload = {"status": "skipped", "reason": reason, **extra}
    write_json(payload, path)
    return payload


def _coerce_numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str)
    out.index = out.index.astype(str)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _pairwise_complete_distances(X: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """Condensed distances using only jointly callable feature states."""
    arr = X.to_numpy(dtype=float)
    observed = arr[np.isfinite(arr)]
    binary = set(np.unique(observed)).issubset({0.0, 1.0}) if observed.size else True
    distances: List[float] = []
    for left in range(arr.shape[0] - 1):
        for right in range(left + 1, arr.shape[0]):
            jointly_callable = np.isfinite(arr[left]) & np.isfinite(arr[right])
            if not jointly_callable.any():
                distances.append(float("nan"))
                continue
            delta = arr[left, jointly_callable] - arr[right, jointly_callable]
            if binary:
                distances.append(float(np.mean(delta != 0.0)))
            else:
                # Root-mean-square distance keeps pairs comparable when the
                # number of jointly callable markers differs.
                distances.append(float(np.sqrt(np.mean(np.square(delta)))))
    return np.asarray(distances, dtype=float), (
        "pairwise_complete_hamming" if binary else "pairwise_complete_rms"
    )


def _limit_features_by_variance(X: pd.DataFrame, max_features: int) -> pd.DataFrame:
    if X.shape[1] <= max_features:
        return X
    variances = X.var(axis=0).sort_values(ascending=False)
    keep = variances.head(max_features).index.tolist()
    return X.loc[:, keep]


# -----------------------------------------------------------------------------
# Registry traversal
# -----------------------------------------------------------------------------


@dataclass
class MarkerLevel:
    model_id: str
    display_name: str
    label_column: str
    status: str
    features: List[str]
    manifest_file: Optional[str] = None
    matrix_file: Optional[str] = None
    parent_path: str = ""
    model_file: Optional[str] = None
    source: str = "registry"


def is_hierarchical_registry(registry: Dict[str, Any]) -> bool:
    protocol = str(registry.get("protocol", "")).strip().lower()
    hierarchy = registry.get("hierarchy", {})
    return protocol == "multi_level_hierarchy_protocol" or (
        isinstance(hierarchy, dict) and isinstance(hierarchy.get("root"), dict)
    )


def _get_nested(mapping: Dict[str, Any], tokens: Sequence[Any]) -> Any:
    current: Any = mapping
    for token in tokens:
        if not isinstance(current, dict):
            return None
        current = current.get(token)
    return current


def _payload_manifest(payload: Dict[str, Any], base_dir: Path) -> Optional[str]:
    manifest = payload.get("feature_manifest") if isinstance(payload, dict) else None
    if isinstance(manifest, dict):
        path = resolve_path(manifest.get("manifest_file"), base_dir)
        if path is not None:
            return str(path)
    return None


def _payload_matrix(payload: Dict[str, Any], base_dir: Path) -> Optional[str]:
    """Best-effort matrix path from feature-panel artifacts."""
    if not isinstance(payload, dict):
        return None
    panel = payload.get("feature_panel_separability")
    if isinstance(panel, dict):
        artifacts = panel.get("artifacts")
        if isinstance(artifacts, dict):
            for key in (
                "selected_panel_matrix",
                "selected_matrix",
                "selected_panel_matrix_csv",
            ):
                path = resolve_path(artifacts.get(key), base_dir)
                if path is not None and path.exists():
                    return str(path)
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        for key in ("selected_panel_matrix", "filtered_matrix", "matrix"):
            path = resolve_path(artifacts.get(key), base_dir)
            if path is not None and path.exists():
                return str(path)
    return None


def _level_from_payload(
    *,
    model_id: str,
    display_name: str,
    payload: Dict[str, Any],
    base_dir: Path,
    label_column: str = "",
    parent_path: str = "",
    source: str = "registry",
) -> Optional[MarkerLevel]:
    if not isinstance(payload, dict):
        return None
    features = [str(f) for f in payload.get("features", []) or [] if str(f)]
    status = str(payload.get("status", "success" if features else "unavailable"))
    label_column = str(payload.get("label_column", label_column) or label_column or "")
    model_file = resolve_path(payload.get("model_file"), base_dir)
    manifest_file = _payload_manifest(payload, base_dir)
    matrix_file = _payload_matrix(payload, base_dir)
    return MarkerLevel(
        model_id=model_id,
        display_name=display_name,
        label_column=label_column,
        status=status,
        features=features,
        manifest_file=manifest_file,
        matrix_file=matrix_file,
        parent_path=parent_path,
        model_file=str(model_file) if model_file is not None else None,
        source=source,
    )


def _iter_hierarchy_levels(
    node: Dict[str, Any], base_dir: Path, path_tokens: List[str]
) -> Iterator[MarkerLevel]:
    if not isinstance(node, dict):
        return

    level_number = str(node.get("level_number", "level"))
    label_column = str(node.get("label_column", "label"))
    path_values = []
    for item in node.get("path", []) or []:
        if isinstance(item, dict):
            value = item.get("value", "")
            if str(value):
                path_values.append(str(value))
    parent_path = " / ".join(path_values)
    model_id = "hierarchy_" + "_".join(
        [_safe_token(f"level_{level_number}"), _safe_token(label_column)]
        + [_safe_token(value, max_len=30) for value in path_values]
    )
    display_name = f"Hierarchy level {level_number}: {label_column}"
    if parent_path:
        display_name += f" | {parent_path}"

    level = _level_from_payload(
        model_id=model_id,
        display_name=display_name,
        payload=node,
        base_dir=base_dir,
        label_column=label_column,
        parent_path=parent_path,
        source="hierarchy",
    )
    if level is not None:
        yield level

    children = node.get("children", {})
    if isinstance(children, dict):
        for child_key, child in children.items():
            if isinstance(child, dict):
                yield from _iter_hierarchy_levels(
                    child, base_dir, path_tokens + [str(child_key)]
                )


def iter_marker_levels(
    registry: Dict[str, Any], base_dir: str | Path
) -> List[MarkerLevel]:
    """Return trainable and deterministic level/node sections from a registry."""
    base = Path(base_dir)
    levels: List[MarkerLevel] = []

    if is_hierarchical_registry(registry):
        hierarchy = registry.get("hierarchy", {}) if isinstance(registry, dict) else {}
        root = hierarchy.get("root", {}) if isinstance(hierarchy, dict) else {}
        if isinstance(root, dict):
            levels.extend(list(_iter_hierarchy_levels(root, base, [])))
        return levels

    level1 = registry.get("level1", {}) if isinstance(registry, dict) else {}
    if isinstance(level1, dict):
        level = _level_from_payload(
            model_id="level1",
            display_name="Level 1 selected markers",
            payload=level1,
            base_dir=base,
            label_column=str(level1.get("label_column", "")),
            source="hierarchy",
        )
        if level is not None:
            levels.append(level)

    level2 = registry.get("level2", {}) if isinstance(registry, dict) else {}
    if isinstance(level2, dict):
        label_column = str(level2.get("label_column", ""))
        for key, display in (
            ("global_fallback", "Level 2 global fallback selected markers"),
            (
                "global_binary_fallback",
                "Level 2 global binary fallback selected markers",
            ),
        ):
            payload = level2.get(key, {})
            if isinstance(payload, dict):
                level = _level_from_payload(
                    model_id=f"level2_{key}",
                    display_name=display,
                    payload=payload,
                    base_dir=base,
                    label_column=label_column,
                    source="hierarchy",
                )
                if level is not None:
                    levels.append(level)

        by_group = level2.get("by_level1_group", {})
        if isinstance(by_group, dict):
            for group, payload in by_group.items():
                if not isinstance(payload, dict):
                    continue
                group_token = _safe_token(group, max_len=40)
                level = _level_from_payload(
                    model_id=f"level2_group_{group_token}",
                    display_name="Level 2 group-specific selected markers",
                    payload=payload,
                    base_dir=base,
                    label_column=label_column,
                    parent_path=str(group),
                    source="hierarchy",
                )
                if level is not None:
                    levels.append(level)

    return levels


def registry_training_matrix(
    registry: Dict[str, Any], base_dir: str | Path
) -> Optional[Path]:
    base = Path(base_dir)
    training = registry.get("training_matrix", {}) if isinstance(registry, dict) else {}
    if isinstance(training, dict):
        path = resolve_path(training.get("aligned_matrix_csv"), base)
        if path is not None and path.exists():
            return path
    fallback = base / "aligned_two_level_matrix.csv"
    if fallback.exists():
        return fallback
    return None


# -----------------------------------------------------------------------------
# Plotting primitives
# -----------------------------------------------------------------------------


def plot_sample_dendrogram(
    X: pd.DataFrame,
    out_png: str | Path,
    *,
    title: str,
    max_features: int = 400,
) -> Dict[str, Any]:
    """Plot a sample dendrogram using the final selected genomic feature matrix."""
    out_png = Path(out_png)
    if dendrogram is None or linkage is None:
        return _write_skip(
            out_png.with_suffix(".skip.json"), "scipy_hierarchy_unavailable"
        )

    X = _coerce_numeric_matrix(X)
    if X.shape[0] < 2:
        return _write_skip(
            out_png.with_suffix(".skip.json"),
            "at_least_two_samples_required",
            samples=int(X.shape[0]),
        )
    if X.shape[1] < 1:
        return _write_skip(
            out_png.with_suffix(".skip.json"),
            "at_least_one_feature_required",
            features=int(X.shape[1]),
        )

    X_plot = _limit_features_by_variance(X, max_features=max_features)
    if X_plot.shape[1] < 1:
        return _write_skip(
            out_png.with_suffix(".skip.json"), "no_features_after_variance_selection"
        )

    distances, metric = _pairwise_complete_distances(X_plot)
    if len(distances) == 0 or not np.isfinite(distances).all():
        return _write_skip(
            out_png.with_suffix(".skip.json"), "nonfinite_pairwise_distances"
        )

    Z = linkage(distances, method="average")
    fig_height = max(5.0, min(28.0, 0.28 * X_plot.shape[0] + 1.5))
    fig_width = max(7.5, min(18.0, 0.035 * X_plot.shape[1] + 7.5))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    dendrogram(
        Z,
        labels=[str(x) for x in X_plot.index],
        orientation="right",
        leaf_font_size=7 if X_plot.shape[0] <= 80 else 5,
        ax=ax,
    )
    ax.set_xlabel(f"Average-linkage distance ({metric})")
    ax.set_ylabel("Samples")
    outputs = _save_figure(fig, out_png, title=title)
    return {
        "status": "generated",
        "artifacts": outputs,
        "samples": int(X_plot.shape[0]),
        "features_used_for_clustering": int(X_plot.shape[1]),
        "distance_metric": metric,
        "linkage_method": "average",
    }


def plot_matrix_heatmap(
    X: pd.DataFrame,
    out_png: str | Path,
    *,
    title: str,
    max_features: int = 80,
) -> Dict[str, Any]:
    out_png = Path(out_png)
    X = _coerce_numeric_matrix(X)
    if X.shape[0] < 1 or X.shape[1] < 1:
        return _write_skip(
            out_png.with_suffix(".skip.json"),
            "nonempty_sample_by_feature_matrix_required",
            samples=int(X.shape[0]),
            features=int(X.shape[1]),
        )
    X_plot = _limit_features_by_variance(X, max_features=max_features)
    fig_width = max(8.0, min(18.0, 0.18 * X_plot.shape[1] + 4.0))
    fig_height = max(4.0, min(24.0, 0.25 * X_plot.shape[0] + 2.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(X_plot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
    ax.set_ylabel("Samples")
    ax.set_xlabel("Selected genomic features")
    ax.set_yticks(range(X_plot.shape[0]))
    ax.set_yticklabels(
        [str(x) for x in X_plot.index], fontsize=7 if X_plot.shape[0] <= 80 else 5
    )
    if X_plot.shape[1] <= 40:
        ax.set_xticks(range(X_plot.shape[1]))
        ax.set_xticklabels([str(x) for x in X_plot.columns], rotation=90, fontsize=5)
    else:
        ax.set_xticks([])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Encoded marker state")
    outputs = _save_figure(fig, out_png, title=title)
    return {
        "status": "generated",
        "artifacts": outputs,
        "samples": int(X_plot.shape[0]),
        "features_plotted": int(X_plot.shape[1]),
        "feature_selection_for_display": "highest_variance"
        if X.shape[1] > X_plot.shape[1]
        else "all_features",
    }


def plot_marker_counts(summary_df: pd.DataFrame, out_png: str | Path) -> Dict[str, Any]:
    if summary_df.empty:
        return _write_skip(
            Path(out_png).with_suffix(".skip.json"), "empty_marker_level_summary"
        )
    df = summary_df.copy()
    df["n_selected_features"] = pd.to_numeric(
        df["n_selected_features"], errors="coerce"
    ).fillna(0)
    df = df.sort_values("n_selected_features", ascending=True)
    fig_height = max(4.0, min(24.0, 0.35 * len(df) + 1.5))
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    ax.barh(df["model_id"].astype(str), df["n_selected_features"].astype(float))
    ax.set_xlabel("Selected genomic features")
    ax.set_ylabel("Trained level/node")
    outputs = _save_figure(
        fig, out_png, title="Final selected marker counts by trained level/node"
    )
    return {"status": "generated", "artifacts": outputs, "levels": int(len(df))}


def plot_jaccard_heatmap(
    levels: Sequence[MarkerLevel], out_png: str | Path
) -> Dict[str, Any]:
    valid = [level for level in levels if level.features]
    if len(valid) < 2:
        return _write_skip(
            Path(out_png).with_suffix(".skip.json"),
            "at_least_two_feature_sets_required",
        )

    names = [level.model_id for level in valid]
    feature_sets = [set(level.features) for level in valid]
    matrix = np.zeros((len(valid), len(valid)), dtype=float)
    for i, a in enumerate(feature_sets):
        for j, b in enumerate(feature_sets):
            denom = len(a | b)
            matrix[i, j] = len(a & b) / denom if denom else 0.0

    fig_size = max(5.5, min(16.0, 0.5 * len(valid) + 3.5))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7 if len(names) <= 30 else 5)
    ax.set_yticklabels(names, fontsize=7 if len(names) <= 30 else 5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard overlap")
    outputs = _save_figure(
        fig, out_png, title="Selected-marker overlap between trained levels/nodes"
    )
    return {"status": "generated", "artifacts": outputs, "levels": int(len(valid))}


def plot_route_counts(
    predictions: pd.DataFrame, out_png: str | Path
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    routes = build_query_routes(predictions)
    if routes.empty:
        return routes, _write_skip(
            Path(out_png).with_suffix(".skip.json"), "no_query_routes_detected"
        )
    counts = routes["route"].value_counts().reset_index()
    counts.columns = ["route", "n_samples"]
    counts = counts.sort_values("n_samples", ascending=True)
    fig_height = max(4.0, min(24.0, 0.40 * len(counts) + 1.5))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.barh(counts["route"].astype(str), counts["n_samples"].astype(float))
    ax.set_xlabel("Query samples")
    ax.set_ylabel("Predicted route")
    outputs = _save_figure(fig, out_png, title="Query prediction route counts")
    return counts, {
        "status": "generated",
        "artifacts": outputs,
        "routes": int(len(counts)),
    }


def plot_numeric_summary(
    df: pd.DataFrame,
    columns: Sequence[str],
    out_png: str | Path,
    *,
    title: str,
    ylabel: str,
) -> Dict[str, Any]:
    cols = [col for col in columns if col in df.columns]
    if not cols:
        return _write_skip(
            Path(out_png).with_suffix(".skip.json"), "no_numeric_columns_available"
        )
    plot_df = df.copy()
    for col in cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = (
        plot_df.set_index("sample_id") if "sample_id" in plot_df.columns else plot_df
    )
    plot_df = plot_df.loc[:, cols].dropna(how="all")
    if plot_df.empty:
        return _write_skip(
            Path(out_png).with_suffix(".skip.json"), "numeric_columns_are_empty"
        )
    fig_width = max(8.0, min(20.0, 0.4 * plot_df.shape[0] + 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Query sample")
    ax.tick_params(axis="x", labelrotation=90)
    outputs = _save_figure(fig, out_png, title=title)
    return {
        "status": "generated",
        "artifacts": outputs,
        "samples": int(plot_df.shape[0]),
        "columns": cols,
    }


# -----------------------------------------------------------------------------
# Network graph helpers
# -----------------------------------------------------------------------------


def write_marker_level_graph(
    levels: Sequence[MarkerLevel],
    out_prefix: str | Path,
    *,
    max_features_per_level: int = 200,
) -> Dict[str, Any]:
    out_prefix = Path(out_prefix)
    if nx is None:
        return _write_skip(out_prefix.with_suffix(".skip.json"), "networkx_unavailable")

    G = nx.Graph()
    for level in levels:
        level_node = f"level::{level.model_id}"
        G.add_node(
            level_node,
            node_type="level",
            label=level.display_name,
            model_id=level.model_id,
            status=level.status,
            n_selected_features=int(len(level.features)),
        )
        features = level.features[:max_features_per_level]
        for feature in features:
            feature_node = f"feature::{feature}"
            if not G.has_node(feature_node):
                G.add_node(feature_node, node_type="feature", label=str(feature))
            G.add_edge(level_node, feature_node)
        if len(level.features) > max_features_per_level:
            omitted = f"omitted::{level.model_id}"
            G.add_node(
                omitted,
                node_type="omitted_features",
                label=f"omitted_features_plus_{len(level.features) - max_features_per_level}",
            )
            G.add_edge(level_node, omitted)

    graphml = out_prefix.with_suffix(".graphml")
    graphml.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, graphml)

    fig, ax = plt.subplots(figsize=(12, 9))
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, seed=42, k=None)
        level_nodes = [
            n for n, d in G.nodes(data=True) if d.get("node_type") == "level"
        ]
        other_nodes = [n for n in G.nodes if n not in level_nodes]
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.5, alpha=0.35)
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_size=12, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=level_nodes, node_size=260, ax=ax)
        labels = {n: G.nodes[n].get("model_id", n) for n in level_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)
    ax.axis("off")
    outputs = _save_figure(
        fig, out_prefix.with_suffix(".png"), title="Selected-marker level graph"
    )
    return {
        "status": "generated",
        "graphml": str(graphml),
        "artifacts": outputs,
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "max_features_per_level_in_graph": int(max_features_per_level),
    }


def write_query_route_graph(
    predictions: pd.DataFrame, out_prefix: str | Path
) -> Dict[str, Any]:
    out_prefix = Path(out_prefix)
    if nx is None:
        return _write_skip(out_prefix.with_suffix(".skip.json"), "networkx_unavailable")
    routes = build_query_routes(predictions)
    if routes.empty:
        return _write_skip(
            out_prefix.with_suffix(".skip.json"), "no_query_routes_detected"
        )

    G = nx.DiGraph()
    for _, row in routes.iterrows():
        sample_id = str(row.get("sample_id", "sample"))
        route_parts = [
            part.strip()
            for part in str(row.get("route", "")).split(" -> ")
            if part.strip()
        ]
        previous = f"sample::{sample_id}"
        G.add_node(previous, node_type="sample", label=sample_id)
        for depth, part in enumerate(route_parts, start=1):
            node = f"route::{depth}::{part}"
            G.add_node(node, node_type="prediction_step", label=part, depth=depth)
            if G.has_edge(previous, node):
                G[previous][node]["weight"] = (
                    int(G[previous][node].get("weight", 1)) + 1
                )
            else:
                G.add_edge(previous, node, weight=1)
            previous = node

    graphml = out_prefix.with_suffix(".graphml")
    graphml.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, graphml)

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, width=0.8, alpha=0.45)
    sample_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "sample"]
    route_nodes = [n for n in G.nodes if n not in sample_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=sample_nodes, node_size=80, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=route_nodes, node_size=220, ax=ax)
    labels = {
        n: str(d.get("label", n))
        for n, d in G.nodes(data=True)
        if d.get("node_type") != "sample"
    }
    if len(sample_nodes) <= 40:
        labels.update({n: str(G.nodes[n].get("label", n)) for n in sample_nodes})
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)
    ax.axis("off")
    outputs = _save_figure(
        fig, out_prefix.with_suffix(".png"), title="Query route graph"
    )
    return {
        "status": "generated",
        "graphml": str(graphml),
        "artifacts": outputs,
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
    }


# -----------------------------------------------------------------------------
# Training registry visualizations
# -----------------------------------------------------------------------------


def _summary_frame(levels: Sequence[MarkerLevel]) -> pd.DataFrame:
    rows = []
    for level in levels:
        rows.append(
            {
                "model_id": level.model_id,
                "display_name": level.display_name,
                "label_column": level.label_column,
                "status": level.status,
                "parent_path": level.parent_path,
                "source": level.source,
                "n_selected_features": int(len(level.features)),
                "manifest_file": level.manifest_file or "",
                "matrix_file": level.matrix_file or "",
                "model_file": level.model_file or "",
            }
        )
    return pd.DataFrame(rows)


def _membership_frame(levels: Sequence[MarkerLevel]) -> pd.DataFrame:
    rows = []
    for level in levels:
        for rank, feature in enumerate(level.features, start=1):
            rows.append(
                {
                    "model_id": level.model_id,
                    "label_column": level.label_column,
                    "parent_path": level.parent_path,
                    "feature_rank_in_model": int(rank),
                    "feature_id": str(feature),
                }
            )
    return pd.DataFrame(rows)


def _read_training_matrix_for_level(
    level: MarkerLevel, registry_matrix: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    if level.matrix_file:
        path = Path(level.matrix_file)
        if path.exists():
            try:
                return read_csv_or_tsv(path, index_col=0)
            except Exception as exc:
                logger.warning("Could not read level matrix %s: %s", path, exc)
    if registry_matrix is not None and level.features:
        available = [
            f
            for f in level.features
            if f in registry_matrix.columns.astype(str).tolist()
        ]
        if available:
            X = registry_matrix.copy()
            X.columns = X.columns.astype(str)
            return X.loc[:, available].copy()
    return None


def visualize_training_registry(
    *,
    registry_path: str | Path,
    output_dir: str | Path,
    max_features_heatmap: int = 80,
    max_features_dendrogram: int = 400,
    max_features_per_level_graph: int = 200,
) -> Dict[str, Any]:
    registry_path = Path(registry_path)
    base_dir = registry_path.parent
    out = ensure_dir(Path(output_dir) / "training")
    registry = load_json(registry_path)
    levels = iter_marker_levels(registry, base_dir)

    summary_df = _summary_frame(levels)
    membership_df = _membership_frame(levels)

    summary_path = out / "registry_marker_level_summary.tsv"
    membership_path = out / "registry_marker_feature_membership.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    membership_df.to_csv(membership_path, sep="\t", index=False)

    artifacts: Dict[str, Any] = {
        "registry_marker_level_summary_tsv": str(summary_path),
        "registry_marker_feature_membership_tsv": str(membership_path),
    }

    artifacts["marker_counts"] = plot_marker_counts(
        summary_df, out / "final_marker_counts_by_level.png"
    )
    artifacts["marker_overlap"] = plot_jaccard_heatmap(
        levels, out / "final_marker_overlap_jaccard_heatmap.png"
    )
    artifacts["marker_level_graph"] = write_marker_level_graph(
        levels,
        out / "final_marker_level_graph",
        max_features_per_level=max_features_per_level_graph,
    )

    matrix_path = registry_training_matrix(registry, base_dir)
    registry_matrix: Optional[pd.DataFrame] = None
    if matrix_path is not None and matrix_path.exists():
        registry_matrix = read_csv_or_tsv(matrix_path, index_col=0)
        registry_matrix.columns = registry_matrix.columns.astype(str)

    per_level: Dict[str, Any] = {}
    per_level_dir = ensure_dir(out / "per_level")
    for level in levels:
        level_out = ensure_dir(per_level_dir / _safe_token(level.model_id))
        X_level = _read_training_matrix_for_level(level, registry_matrix)
        if X_level is None or X_level.empty:
            per_level[level.model_id] = {
                "status": "skipped",
                "reason": "no_readable_sample_by_selected_marker_matrix",
                "n_selected_features": int(len(level.features)),
            }
            write_json(per_level[level.model_id], level_out / "visualization_skip.json")
            continue
        X_level = _coerce_numeric_matrix(X_level)
        per_level[level.model_id] = {
            "status": "generated_or_partially_generated",
            "dendrogram": plot_sample_dendrogram(
                X_level,
                level_out / "selected_marker_dendrogram.png",
                title=f"Sample clustering over final selected markers — {level.model_id}",
                max_features=max_features_dendrogram,
            ),
            "heatmap": plot_matrix_heatmap(
                X_level,
                level_out / "selected_marker_heatmap.png",
                title=f"Final selected-marker matrix — {level.model_id}",
                max_features=max_features_heatmap,
            ),
            "samples": int(X_level.shape[0]),
            "features": int(X_level.shape[1]),
        }
        write_json(per_level[level.model_id], level_out / "visualization_summary.json")

    artifacts["per_level"] = per_level
    result = {
        "status": "complete",
        "mode": "training_registry_visualization",
        "registry_path": str(registry_path),
        "output_dir": str(out),
        "n_levels_or_nodes": int(len(levels)),
        "training_matrix": str(matrix_path) if matrix_path is not None else None,
        "artifacts": artifacts,
        "notes": [
            "Visualization is downstream-only and does not change marker selection or model fitting.",
            "Sample dendrograms are computed only from the saved final selected-marker spaces.",
            "When a model-specific selected panel matrix is not found, the aligned training matrix is subset to that model's saved feature list.",
        ],
    }
    write_json(result, out / "training_visualization_summary.json")
    return result


# -----------------------------------------------------------------------------
# Query visualizations
# -----------------------------------------------------------------------------


def build_query_routes(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    rows = []
    for _, row in predictions.iterrows():
        sample_id = str(row.get("sample_id", row.name))
        if (
            "predicted_hierarchy_path" in predictions.columns
            and str(row.get("predicted_hierarchy_path", "")).strip()
        ):
            route = str(row.get("predicted_hierarchy_path"))
        else:
            parts: List[str] = []
            if "predicted_level1_identity" in predictions.columns:
                parts.append(
                    "Level1=" + str(row.get("predicted_level1_identity", "unavailable"))
                )
            if "predicted_level2_identity" in predictions.columns:
                parts.append(
                    "Level2=" + str(row.get("predicted_level2_identity", "unavailable"))
                )
            if not parts and "predicted_terminal_label" in predictions.columns:
                parts.append(
                    "Terminal="
                    + str(row.get("predicted_terminal_label", "unavailable"))
                )
            route = " -> ".join(parts) if parts else "unavailable"
        rows.append({"sample_id": sample_id, "route": route})
    return pd.DataFrame(rows)


def _find_query_matrix(query_dir: Path) -> Optional[Path]:
    candidates = [
        query_dir / "fasta_query_encoding" / "fasta_selected_feature_matrix.csv",
        query_dir / "vcf_query_encoding" / "vcf_selected_feature_matrix.csv",
        query_dir
        / "raw_sequence_query_encoding"
        / "raw_sequence_selected_feature_matrix.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    recursive = sorted(query_dir.rglob("*_selected_feature_matrix.csv"))
    if recursive:
        return recursive[0]
    return None


def _find_query_calls(query_dir: Path) -> List[Path]:
    patterns = ["*_feature_calls.tsv"]
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(sorted(query_dir.rglob(pattern)))
    return paths


def _plot_call_status_counts(
    call_paths: Sequence[Path], out_png: Path
) -> Dict[str, Any]:
    frames = []
    for path in call_paths:
        try:
            df = pd.read_csv(path, sep="\t")
            if not df.empty:
                df["_source_file"] = str(path)
                frames.append(df)
        except Exception as exc:
            logger.warning("Could not read query call table %s: %s", path, exc)
    if not frames:
        return _write_skip(
            out_png.with_suffix(".skip.json"), "no_readable_query_call_tables"
        )
    calls = pd.concat(frames, ignore_index=True)
    col = None
    for candidate in ("allele_call", "mapping_status", "mapping_quality"):
        if candidate in calls.columns:
            col = candidate
            break
    if col is None:
        return _write_skip(
            out_png.with_suffix(".skip.json"), "no_call_status_columns_available"
        )
    counts = calls[col].astype(str).value_counts().sort_values(ascending=True)
    fig_height = max(4.0, min(18.0, 0.35 * len(counts) + 1.5))
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    ax.barh(counts.index.astype(str), counts.values.astype(float))
    ax.set_xlabel("Feature calls")
    ax.set_ylabel(col)
    outputs = _save_figure(
        fig, out_png, title=f"Query feature-call summary by {col}"
    )
    counts_path = out_png.with_suffix(".tsv")
    counts.rename("n_feature_calls").reset_index().rename(
        columns={"index": col}
    ).to_csv(counts_path, sep="\t", index=False)
    return {
        "status": "generated",
        "artifacts": {**outputs, "counts_tsv": str(counts_path)},
        "status_column": col,
    }


def visualize_query_results(
    *,
    query_dir: str | Path,
    output_dir: str | Path,
    max_features_heatmap: int = 80,
    max_features_dendrogram: int = 400,
) -> Dict[str, Any]:
    query_dir = Path(query_dir)
    out = ensure_dir(Path(output_dir) / "query")
    predictions_path = query_dir / "query_predictions.csv"
    if not predictions_path.exists():
        alt = query_dir / "two_level_predictions.csv"
        if alt.exists():
            predictions_path = alt
        else:
            raise FileNotFoundError(f"No query_predictions.csv found under {query_dir}")

    predictions = pd.read_csv(predictions_path)
    routes = build_query_routes(predictions)
    routes_path = out / "query_prediction_routes.tsv"
    routes.to_csv(routes_path, sep="\t", index=False)

    artifacts: Dict[str, Any] = {
        "query_predictions_csv": str(predictions_path),
        "query_prediction_routes_tsv": str(routes_path),
    }
    counts, route_plot = plot_route_counts(
        predictions, out / "query_prediction_route_counts.png"
    )
    route_counts_path = out / "query_prediction_route_counts.tsv"
    counts.to_csv(route_counts_path, sep="\t", index=False)
    artifacts["query_prediction_route_counts_tsv"] = str(route_counts_path)
    artifacts["route_counts_plot"] = route_plot

    support_cols = [
        "level1_support",
        "level2_support",
    ] + [
        col
        for col in predictions.columns
        if col.endswith("_support") and col not in {"level1_support", "level2_support"}
    ]
    artifacts["support_summary_plot"] = plot_numeric_summary(
        predictions,
        support_cols,
        out / "query_support_summary.png",
        title="Query prediction support summary",
        ylabel="Support",
    )

    quality_cols = [
        "query_unique_mapped_fraction",
        "query_active_feature_fraction",
        "query_resolved_feature_fraction",
        "query_resolved_baseline_feature_fraction",
        "level1_resolved_feature_fraction",
        "level2_resolved_feature_fraction",
    ]
    artifacts["marker_quality_plot"] = plot_numeric_summary(
        predictions,
        quality_cols,
        out / "query_marker_quality.png",
        title="Query marker recovery and evidence fractions",
        ylabel="Fraction",
    )

    artifacts["query_route_graph"] = write_query_route_graph(
        predictions, out / "query_route_graph"
    )

    matrix_path = _find_query_matrix(query_dir)
    if matrix_path is not None and matrix_path.exists():
        X_query = read_csv_or_tsv(matrix_path, index_col=0)
        X_query = _coerce_numeric_matrix(X_query)
        artifacts["query_selected_feature_matrix"] = str(matrix_path)
        artifacts["query_selected_feature_dendrogram"] = plot_sample_dendrogram(
            X_query,
            out / "query_selected_feature_dendrogram.png",
            title="Query sample clustering over trained selected markers",
            max_features=max_features_dendrogram,
        )
        artifacts["query_selected_feature_heatmap"] = plot_matrix_heatmap(
            X_query,
            out / "query_selected_feature_heatmap.png",
            title="Query selected-marker matrix",
            max_features=max_features_heatmap,
        )
    else:
        artifacts["query_selected_feature_matrix"] = None
        artifacts["query_selected_feature_dendrogram"] = _write_skip(
            out / "query_selected_feature_dendrogram.skip.json",
            "no_query_selected_feature_matrix_found",
        )
        artifacts["query_selected_feature_heatmap"] = _write_skip(
            out / "query_selected_feature_heatmap.skip.json",
            "no_query_selected_feature_matrix_found",
        )

    call_paths = _find_query_calls(query_dir)
    artifacts["query_call_status_plot"] = _plot_call_status_counts(
        call_paths, out / "query_feature_call_status_counts.png"
    )

    result = {
        "status": "complete",
        "mode": "query_visualization",
        "query_dir": str(query_dir),
        "output_dir": str(out),
        "n_query_samples": int(predictions.shape[0]),
        "artifacts": artifacts,
        "notes": [
            "Visualization is downstream-only and does not rerun model inference or marker encoding.",
            "The query dendrogram uses the saved selected-feature matrix when available.",
            "Baseline states encoded as 0 remain part of the query marker pattern; nonbaseline and caution calls are summarized separately where call tables are available.",
        ],
    }
    write_json(result, out / "query_visualization_summary.json")
    return result


# -----------------------------------------------------------------------------
# Combined runner and CLI
# -----------------------------------------------------------------------------


def visualize_all(
    *,
    registry_path: str | Path,
    query_dir: Optional[str | Path],
    output_dir: str | Path,
    max_features_heatmap: int = 80,
    max_features_dendrogram: int = 400,
    max_features_per_level_graph: int = 200,
) -> Dict[str, Any]:
    out = ensure_dir(output_dir)
    training = visualize_training_registry(
        registry_path=registry_path,
        output_dir=out,
        max_features_heatmap=max_features_heatmap,
        max_features_dendrogram=max_features_dendrogram,
        max_features_per_level_graph=max_features_per_level_graph,
    )
    query = None
    if query_dir is not None:
        query = visualize_query_results(
            query_dir=query_dir,
            output_dir=out,
            max_features_heatmap=max_features_heatmap,
            max_features_dendrogram=max_features_dendrogram,
        )
    result = {"status": "complete", "training": training, "query": query}
    write_json(result, Path(out) / "visualization_summary.json")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate NetworkParser training/query visualizations from existing artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--output_dir",
            required=True,
            help="Directory where visualization files will be written.",
        )
        p.add_argument(
            "--max_features_heatmap",
            type=int,
            default=80,
            help="Maximum feature columns shown in heatmaps.",
        )
        p.add_argument(
            "--max_features_dendrogram",
            type=int,
            default=400,
            help="Maximum features used for dendrogram clustering.",
        )
        p.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    p_train = sub.add_parser(
        "training", help="Visualize final selected markers in a trained registry."
    )
    p_train.add_argument(
        "--registry",
        required=True,
        help="Path to hierarchy/two-level/hierarchical model registry JSON.",
    )
    p_train.add_argument(
        "--max_features_per_level_graph",
        type=int,
        default=200,
        help="Cap feature nodes per level in the GraphML/PNG graph.",
    )
    add_common(p_train)

    p_query = sub.add_parser(
        "query", help="Visualize completed query output directory."
    )
    p_query.add_argument(
        "--query_dir",
        required=True,
        help="Directory containing query_predictions.csv and query artifacts.",
    )
    add_common(p_query)

    p_all = sub.add_parser(
        "all", help="Visualize both training registry and query results."
    )
    p_all.add_argument(
        "--registry",
        required=True,
        help="Path to hierarchy/two-level/hierarchical model registry JSON.",
    )
    p_all.add_argument(
        "--query_dir",
        default=None,
        help="Optional directory containing query_predictions.csv and query artifacts.",
    )
    p_all.add_argument(
        "--max_features_per_level_graph",
        type=int,
        default=200,
        help="Cap feature nodes per level in the GraphML/PNG graph.",
    )
    add_common(p_all)
    return parser


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(bool(getattr(args, "verbose", False)))

    if args.command == "training":
        result = visualize_training_registry(
            registry_path=args.registry,
            output_dir=args.output_dir,
            max_features_heatmap=int(args.max_features_heatmap),
            max_features_dendrogram=int(args.max_features_dendrogram),
            max_features_per_level_graph=int(args.max_features_per_level_graph),
        )
    elif args.command == "query":
        result = visualize_query_results(
            query_dir=args.query_dir,
            output_dir=args.output_dir,
            max_features_heatmap=int(args.max_features_heatmap),
            max_features_dendrogram=int(args.max_features_dendrogram),
        )
    elif args.command == "all":
        result = visualize_all(
            registry_path=args.registry,
            query_dir=args.query_dir,
            output_dir=args.output_dir,
            max_features_heatmap=int(args.max_features_heatmap),
            max_features_dendrogram=int(args.max_features_dendrogram),
            max_features_per_level_graph=int(args.max_features_per_level_graph),
        )
    else:  # pragma: no cover
        parser.print_help()
        return 2

    logger.info(
        "Visualization complete | output_dir=%s",
        result.get("output_dir", args.output_dir),
    )
    print(json.dumps(result, indent=2, default=json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
