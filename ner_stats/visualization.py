"""Visualization helpers: PCA, t-SNE, UMAP and plotting utilities.

Designed to produce publication-ready figures for thesis use.
"""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


def _ensure_color_map(class_names: Sequence[str]) -> Dict[str, Tuple[float, float, float]]:
    cmap = plt.get_cmap("tab20")
    colors = {name: cmap(i % 20) for i, name in enumerate(sorted(class_names))}
    return colors


def pca_reduce(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(X)
    return proj, pca


def tsne_reduce(X: np.ndarray, n_components: int = 2, random_state: int = 42, perplexity: Optional[float] = None) -> np.ndarray:
    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for t-SNE.")
    if perplexity is None:
        # choose a safe perplexity: between 2 and 30, but less than n_samples
        p = max(2.0, min(30.0, float(max(2, n_samples // 3))))
    else:
        p = float(perplexity)
    tsne = TSNE(n_components=n_components, init="pca", random_state=random_state, perplexity=p)
    return tsne.fit_transform(X)


def umap_reduce(X: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    if not _HAS_UMAP:
        raise ImportError("UMAP is not installed. Install with `pip install umap-learn`.")
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    return reducer.fit_transform(X)


def plot_2d_scatter(
    X2: np.ndarray,
    labels: Sequence[str],
    out_path: str,
    title: Optional[str] = None,
    dpi: int = 300,
    figsize: Tuple[float, float] = (10, 8),
):
    class_names = sorted(set(labels))
    colors = _ensure_color_map(class_names)

    plt.figure(figsize=figsize)
    for cls in class_names:
        mask = [l == cls for l in labels]
        pts = X2[np.array(mask)]
        if pts.size == 0:
            continue
        plt.scatter(pts[:, 0], pts[:, 1], label=cls, alpha=0.7, s=12, c=[colors[cls]])

    plt.xlabel("PC1", fontsize=14)
    plt.ylabel("PC2", fontsize=14)
    if title:
        plt.title(title, fontsize=16)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_3d_scatter(
    X3: np.ndarray,
    labels: Sequence[str],
    out_path: str,
    title: Optional[str] = None,
    dpi: int = 300,
    figsize: Tuple[float, float] = (12, 9),
):
    class_names = sorted(set(labels))
    colors = _ensure_color_map(class_names)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    for cls in class_names:
        mask = [l == cls for l in labels]
        pts = X3[np.array(mask)]
        if pts.size == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label=cls, s=18, alpha=0.8, color=colors[cls])

    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_zlabel("PC3", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def explained_variance_text(pca: PCA) -> str:
    ratios = pca.explained_variance_ratio_
    text_lines = [f"PC{i+1}: {r:.4f}" for i, r in enumerate(ratios)]
    return "\n".join(text_lines)


def plot_prototypes_2d(
    X2: np.ndarray,
    labels: Sequence[str],
    prototypes: Dict[str, Sequence[float]],
    out_path: str,
    title: Optional[str] = None,
    dpi: int = 300,
    figsize: Tuple[float, float] = (10, 8),
    pca: Optional[PCA] = None,
):
    """Plot 2D projections of embeddings and overlay class prototypes."""
    class_names = sorted(set(labels) | set(prototypes.keys()))
    colors = _ensure_color_map(class_names)

    plt.figure(figsize=figsize)
    for cls in sorted(set(labels)):
        mask = [l == cls for l in labels]
        pts = X2[np.array(mask)]
        if pts.size == 0:
            continue
        plt.scatter(pts[:, 0], pts[:, 1], label=cls, alpha=0.6, s=10, c=[colors[cls]])

    # project prototypes: if a PCA is provided assume prototypes are original-dim and use it
    prot_keys = []
    prot_vals = []
    for k, v in prototypes.items():
        prot_keys.append(k)
        prot_vals.append(np.asarray(v, dtype=np.float64))
    if prot_vals:
        stacked = np.vstack(prot_vals)
        if pca is not None:
            prot_proj = pca.transform(stacked)
        else:
            # if prototypes appear 2D already, use them directly
            if stacked.shape[1] == X2.shape[1]:
                prot_proj = stacked
            else:
                # fall back: fit a PCA on prototypes only
                from sklearn.decomposition import PCA as _PCA

                _pca = _PCA(n_components=2)
                prot_proj = _pca.fit_transform(stacked)

        for k, p in zip(prot_keys, prot_proj):
            plt.scatter(p[0], p[1], marker="X", s=120, edgecolor="k", linewidth=0.8, c=[colors.get(k, (0, 0, 0))])
            plt.text(p[0], p[1], k, fontsize=9, weight="bold")

    plt.xlabel("PC1", fontsize=14)
    plt.ylabel("PC2", fontsize=14)
    if title:
        plt.title(title, fontsize=16)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
