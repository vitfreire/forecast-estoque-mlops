from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd


EPS = 1e-6


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


def _clip_prob(p: float) -> float:
    if p < EPS:
        return EPS
    if p > 1.0:
        return 1.0
    return p


def _psi_from_props(ref_props: List[float], cur_props: List[float]) -> float:
    if len(ref_props) != len(cur_props):
        raise ValueError("ref_props e cur_props devem ter o mesmo tamanho")

    psi = 0.0
    for r, c in zip(ref_props, cur_props):
        r = _clip_prob(float(r))
        c = _clip_prob(float(c))
        psi += (c - r) * math.log(c / r)
    return float(psi)


def _numeric_bin_edges_from_ref(x_ref: pd.Series, n_bins: int) -> List[float]:
    # bins por quantis no reference
    x = pd.to_numeric(x_ref, errors="coerce").dropna()
    if x.empty:
        # caso extremo: tudo NaN
        return [-np.inf, np.inf]

    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x.values, qs).astype(float).tolist()

    # garante monotonicidade e remove duplicatas
    edges = sorted(set(edges))

    if len(edges) < 2:
        v = float(x.iloc[0])
        return [-np.inf, v, np.inf]

    # estende com infinitos para pegar outliers no batch
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _numeric_props_from_edges(x: pd.Series, edges: List[float]) -> List[float]:
    # retorna proporções por bin (len = len(edges)-1)
    x_num = pd.to_numeric(x, errors="coerce")
    counts, _ = np.histogram(x_num.dropna().values,
                             bins=np.array(edges, dtype=float))
    total = counts.sum()
    if total == 0:
        # sem dados válidos -> tudo vai para 0
        return [0.0] * (len(edges) - 1)

    return [float(c) / float(total) for c in counts.tolist()]


def _cat_reference_distribution(x_ref: pd.Series, top_k: int) -> Dict[str, float]:
    # normaliza como string; missing explícito
    s = x_ref.astype("object").where(x_ref.notna(), "missing").astype(str)
    vc = s.value_counts(normalize=True)

    top = vc.head(top_k)
    out: Dict[str, float] = {k: float(v) for k, v in top.items()}

    other = float(1.0 - top.sum())
    out["__other__"] = max(0.0, other)
    return out


def _cat_current_distribution(
    x_cur: pd.Series,
    ref_dist: Dict[str, float],
) -> Dict[str, float]:
    # joga categorias que não estão no ref em "__other__"
    s = x_cur.astype("object").where(x_cur.notna(), "missing").astype(str)

    total = len(s)
    if total == 0:
        return {k: 0.0 for k in ref_dist.keys()}

    counts = s.value_counts()
    cur: Dict[str, float] = {k: 0.0 for k in ref_dist.keys()}

    known_keys = set(ref_dist.keys()) - {"__other__"}
    other_count = 0

    for cat, cnt in counts.items():
        if cat in known_keys:
            cur[cat] = float(cnt) / float(total)
        else:
            other_count += int(cnt)

    cur["__other__"] = float(other_count) / float(total)
    return cur


@dataclass
class DriftReference:
    feature_names: List[str]
    numeric_edges: Dict[str, List[float]]
    numeric_ref_props: Dict[str, List[float]]
    cat_ref_props: Dict[str, Dict[str, float]]
    n_ref: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "numeric_edges": self.numeric_edges,
            "numeric_ref_props": self.numeric_ref_props,
            "cat_ref_props": self.cat_ref_props,
            "n_ref": self.n_ref,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DriftReference":
        return DriftReference(
            feature_names=list(d["feature_names"]),
            numeric_edges={k: list(v) for k, v in d.get(
                "numeric_edges", {}).items()},
            numeric_ref_props={k: list(v) for k, v in d.get(
                "numeric_ref_props", {}).items()},
            cat_ref_props={k: dict(v) for k, v in d.get(
                "cat_ref_props", {}).items()},
            n_ref=int(d.get("n_ref", 0)),
        )


def build_drift_reference(
    X_ref: pd.DataFrame,
    feature_names: List[str],
    cat_cols: List[str],
    n_bins: int = 10,
    top_k: int = 30,
) -> DriftReference:
    # garante colunas e ordem
    X = X_ref.copy()
    X = X[feature_names].copy()

    numeric_edges: Dict[str, List[float]] = {}
    numeric_ref_props: Dict[str, List[float]] = {}
    cat_ref_props: Dict[str, Dict[str, float]] = {}

    for col in feature_names:
        if col in cat_cols:
            cat_ref_props[col] = _cat_reference_distribution(
                X[col], top_k=top_k)
        else:
            edges = _numeric_bin_edges_from_ref(X[col], n_bins=n_bins)
            props = _numeric_props_from_edges(X[col], edges=edges)
            numeric_edges[col] = edges
            numeric_ref_props[col] = props

    return DriftReference(
        feature_names=feature_names,
        numeric_edges=numeric_edges,
        numeric_ref_props=numeric_ref_props,
        cat_ref_props=cat_ref_props,
        n_ref=int(len(X)),
    )


def compute_drift_psi(
    X_cur: pd.DataFrame,
    ref: DriftReference,
    cat_cols: List[str],
) -> Dict[str, float]:
    X = X_cur.copy()
    X = X[ref.feature_names].copy()

    psi_by_feature: Dict[str, float] = {}

    for col in ref.feature_names:
        if col in cat_cols:
            ref_dist = ref.cat_ref_props.get(col, {})
            cur_dist = _cat_current_distribution(X[col], ref_dist=ref_dist)

            keys = list(ref_dist.keys())
            ref_props = [float(ref_dist[k]) for k in keys]
            cur_props = [float(cur_dist.get(k, 0.0)) for k in keys]
            psi_by_feature[col] = _psi_from_props(ref_props, cur_props)
        else:
            edges = ref.numeric_edges.get(col)
            ref_props = ref.numeric_ref_props.get(col)

            if edges is None or ref_props is None:
                # se faltar referência, não calcula
                continue

            cur_props = _numeric_props_from_edges(X[col], edges=edges)
            psi_by_feature[col] = _psi_from_props(ref_props, cur_props)

    return psi_by_feature


def summarize_drift(
    psi_by_feature: Dict[str, float],
    threshold: float = 0.2,
) -> Dict[str, Any]:
    if not psi_by_feature:
        return {
            "drift_detected": False,
            "drift_threshold": float(threshold),
            "max_psi": 0.0,
            "n_features_over_threshold": 0,
            "top_features": [],
        }

    items = sorted(psi_by_feature.items(), key=lambda x: x[1], reverse=True)
    max_psi = float(items[0][1])
    over = [(k, float(v)) for k, v in items if float(v) >= float(threshold)]

    return {
        "drift_detected": len(over) > 0,
        "drift_threshold": float(threshold),
        "max_psi": max_psi,
        "n_features_over_threshold": int(len(over)),
        "top_features": [(k, float(v)) for k, v in items[:10]],
    }


def save_reference_json(ref: DriftReference, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ref.to_dict(), f, ensure_ascii=False, indent=2)


def load_reference_json(path: str) -> DriftReference:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return DriftReference.from_dict(d)
