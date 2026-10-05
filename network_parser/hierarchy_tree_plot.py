#!/usr/bin/env python3
"""Dendrogram-style hierarchy tree plot from hierarchical_model_registry.json."""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

STATUS_EDGE = {
    "success": "#2E7D32",
    "constant": "#F9A825",
    "skipped": "#C62828",
    "failed": "#C62828",
}


@dataclass
class HNode:
    label_column: str
    route_label: str
    status: str
    algorithm: Optional[str]
    n_train: Optional[int]
    level_number: int
    children: List["HNode"] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0


def _algo(node: Dict[str, Any]) -> Optional[str]:
    model = node.get("model")
    if isinstance(model, dict):
        for k in ("selected_algorithm", "algorithm", "ml_algorithm"):
            if model.get(k):
                return str(model[k]).upper()
    mf = str(node.get("model_file") or "")
    for a in ("LR", "RF", "DT", "SVC", "MLP"):
        if f"/{a}_" in mf:
            return a
    if str(node.get("status")) == "constant":
        return "CONST"
    return None


def build_tree(node: Dict[str, Any], route_label: str = "ROOT", depth: int = 0) -> HNode:
    status = str(node.get("status") or "unknown")
    h = HNode(
        label_column=str(node.get("label_column") or f"level_{depth}"),
        route_label=str(route_label),
        status=status,
        algorithm=_algo(node),
        n_train=int(node["n_training_samples"])
        if node.get("n_training_samples") is not None
        else (int(node["n_samples"]) if node.get("n_samples") is not None else None),
        level_number=int(node.get("level_number") or depth + 1),
    )
    children = node.get("children") or {}
    if isinstance(children, dict):
        for k, child in children.items():
            if isinstance(child, dict):
                h.children.append(build_tree(child, route_label=str(k), depth=depth + 1))
    return h


def layout_tree(root: HNode, x_spacing: float = 2.4, y_spacing: float = 1.55) -> None:
    def place(n: HNode, depth: int, y0: float) -> float:
        n.x = depth * x_spacing
        if not n.children:
            n.y = y0
            return y0 + y_spacing
        y = y0
        ys = []
        for c in n.children:
            y = place(c, depth + 1, y)
            ys.append(c.y)
        n.y = sum(ys) / len(ys)
        return y

    place(root, 0, 0.0)


def short_label(text: str, width: int = 18) -> str:
    t = (
        str(text)
        .replace("test_", "")
        .replace("_parent", "")
        .replace("Lineage_Supergroup", "Supergroup")
        .replace("Lineage_clean", "Lineage")
        .replace("Lineage_family", "Family")
        .replace("AMR_binary", "AMR")
        .replace("Resistance_Profile_Collapsed", "Profile")
        .replace("Resistance_Profile", "Profile")
    )
    return "\n".join(textwrap.wrap(t, width=width)) if len(t) > width else t


def draw_registry_tree(
    registry_path: Path,
    out_png: Path,
    out_pdf: Path,
    *,
    figure_number: int = 6,
    title: Optional[str] = None,
) -> None:
    reg = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    hierarchy = reg.get("hierarchy") or {}
    root_dict = hierarchy.get("root")
    if not isinstance(root_dict, dict):
        raise ValueError("registry has no hierarchy.root")
    labels = hierarchy.get("label_columns") or []
    root = build_tree(root_dict)
    layout_tree(root)

    nodes: List[HNode] = []

    def collect(n: HNode) -> None:
        nodes.append(n)
        for c in n.children:
            collect(c)

    collect(root)
    ys = [n.y for n in nodes]
    xs = [n.x for n in nodes]
    y_min, y_max = min(ys), max(ys)
    x_min, x_max = min(xs), max(xs)
    fig_h = max(6.5, 1.1 * (y_max - y_min + 2))
    fig_w = max(10.0, 2.8 * (x_max - x_min + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    box_w, box_h = 1.85, 1.05

    for n in nodes:
        for c in n.children:
            x0, y0 = n.x + box_w / 2, n.y
            x1, y1 = c.x - box_w / 2, c.y
            mid = (x0 + x1) / 2
            ax.plot([x0, mid, mid, x1], [y0, y0, y1, y1], color="#90A4AE", lw=1.6, zorder=1)
            ax.text(
                mid + 0.05,
                (y0 + y1) / 2,
                short_label(c.route_label, 14),
                fontsize=8,
                color="#37474F",
                va="center",
                ha="left",
                fontstyle="italic",
                zorder=2,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
            )

    for n in nodes:
        algo = (n.algorithm or "—").upper()
        face = {
            "LR": "#E3F2FD",
            "RF": "#FFF3E0",
            "DT": "#E8F5E9",
            "CONST": "#FFF8E1",
        }.get(algo, "#ECEFF1")
        edge = STATUS_EDGE.get(n.status, "#607D8B")
        if n.status == "skipped":
            face = "#FFEBEE"
        ax.add_patch(
            FancyBboxPatch(
                (n.x - box_w / 2, n.y - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=face,
                edgecolor=edge,
                linewidth=2.0 if n.status == "success" else 1.5,
                zorder=3,
            )
        )
        header = "ROOT" if n.route_label == "ROOT" else short_label(n.route_label, 16)
        n_txt = f"n={n.n_train}" if n.n_train is not None else "n=—"
        body = f"{short_label(n.label_column, 16)}\n{algo if algo != 'CONST' else 'const'} · {n.status}\n{n_txt}"
        ax.text(n.x, n.y + 0.18, header, ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=4)
        ax.text(n.x, n.y - 0.18, body, ha="center", va="center", fontsize=7.5, color="#424242", zorder=4)

    depth_task = {}
    for n in nodes:
        depth_task.setdefault(int(round(n.x / 2.4)), n.label_column)
    for d, task in sorted(depth_task.items()):
        ax.text(d * 2.4, y_max + 1.0, f"L{d+1}: {short_label(task, 22)}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#455A64")

    ax.set_xlim(x_min - 1.2, x_max + 1.2)
    ax.set_ylim(y_min - 1.0, y_max + 1.5)
    path = " → ".join(str(x) for x in labels) if labels else "hierarchy"
    caption = title or f"Figure {figure_number}. Hierarchical model tree — {path}"
    fig.text(
        0.5,
        -0.02,
        caption,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.06,
        (
            "Edge labels = parent-class routes; boxes = task, algorithm, "
            "status, training n. From trained registry only."
        ),
        ha="center",
        va="top",
        fontsize=8.5,
        color="#546E7A",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
