"""Visualizations for prototype-based comparisons."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np
    import plotly.graph_objects as go
    from numpy.typing import NDArray

PlotBackend = Literal['plotly', 'matplotlib']

_PLOTLY_STYLE: dict = {
    'font': {'family': 'Times New Roman', 'size': 12},
    'paper_bgcolor': 'white',
    'plot_bgcolor': 'white',
}


def plot_prototype_heatmap(
    J: NDArray[np.float32],
    model_a: str,
    model_b: str,
    color_scale: str = 'Magma',
    zmin: float = 0.0,
    zmax: float = 1.0,
) -> go.Figure:
    """Create a heatmap visualization of the prototype Jaccard similarity matrix.

    Parameters
    ----------
    J : np.ndarray, shape (k, k)
        Jaccard similarity matrix.
    model_a : str
        Name of model A (reference).
    model_b : str
        Name of model B (target).
    color_scale : str
        Plotly color scale name. Default 'Viridis'.
    zmin : float
        Minimum value for color scale. Default 0.0.
    zmax : float
        Maximum value for color scale. Default 1.0.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive heatmap figure.
    """
    import plotly.express as px

    fig = px.imshow(
        J,
        color_continuous_scale=color_scale,
        zmin=zmin,
        zmax=zmax,
        labels={
            'x': model_b,
            'y': model_a,
            'color': 'Jaccard',
        },
        aspect='equal',
    )

    k = J.shape[0]
    tick_step = max(1, k // 10)

    fig.update_layout(
        **_PLOTLY_STYLE,
        height=500,
        margin={'t': 20, 'b': 50, 'l': 50, 'r': 80},
        coloraxis_colorbar={
            'title': 'Jaccard',
            'thickness': 14,
            'len': 0.75,
        },
        xaxis={
            'tickmode': 'linear',
            'tick0': 0,
            'dtick': tick_step,
            'title_standoff': 8,
        },
        yaxis={
            'tickmode': 'linear',
            'tick0': 0,
            'dtick': tick_step,
            'title_standoff': 8,
        },
    )

    return fig


def plot_similarity_profile(
    k_values: list[int],
    metrics: dict[str, list[float]],
    ylim: tuple[float, float] = (0, 1),
    ylim_entropy: tuple[float, float] | None = None,
) -> go.Figure:
    """Create a line plot showing similarity metrics across k values.

    Parameters
    ----------
    k_values : list[int]
        List of k (number of clusters) values.
    metrics : dict[str, list[float]]
        Dictionary mapping metric names to lists of values.
        Keys: 'f1', 'hungarian', 'entropy'.
    ylim : tuple[float, float]
        Y-axis limits for score metrics (f1, hungarian).
    ylim_entropy : tuple[float, float], optional
        Y-axis limits for entropy. If None, auto-scales.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive line plot figure with dual y-axis for entropy.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    k_values = list(k_values)
    has_entropy = 'entropy' in metrics
    fig = make_subplots(specs=[[{'secondary_y': True}]]) if has_entropy else go.Figure()

    colors = {
        'f1': '#00CC96',
        'hungarian': '#AB63FA',
        'entropy': '#FFA15A',
    }
    line_dashes = {
        'f1': 'solid',
        'hungarian': 'dot',
        'entropy': 'dot',
    }

    for metric_name, values in metrics.items():
        color = colors.get(metric_name, '#333333')
        dash = line_dashes.get(metric_name, 'solid')

        if has_entropy and metric_name == 'entropy':
            fig.add_trace(
                go.Scatter(
                    x=k_values,
                    y=values,
                    mode='lines+markers',
                    name=metric_name.capitalize(),
                    line={'color': color, 'dash': dash, 'width': 2},
                    marker={'size': 6, 'symbol': 'diamond'},
                ),
                secondary_y=True,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=k_values,
                    y=values,
                    mode='lines+markers',
                    name=metric_name.capitalize(),
                    line={'color': color, 'dash': dash, 'width': 2},
                    marker={'size': 6},
                ),
                secondary_y=False,
            )

    if has_entropy:
        fig.update_layout(
            **_PLOTLY_STYLE,
            xaxis_title='Prototypes',
            hovermode='x unified',
            height=400,
            margin={'t': 20, 'b': 50, 'l': 50, 'r': 50},
        )
        fig.update_yaxes(title_text='Score', range=list(ylim), secondary_y=False)
        ylim_ent = ylim_entropy or (None, None)
        fig.update_yaxes(title_text='Entropy', range=list(ylim_ent), secondary_y=True)
    else:
        fig.update_layout(
            **_PLOTLY_STYLE,
            xaxis_title='Prototypes',
            yaxis_title='Score',
            yaxis_range=list(ylim),
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'right',
                'x': 1,
            },
            hovermode='x unified',
            height=400,
            margin={'t': 20, 'b': 50, 'l': 50, 'r': 20},
        )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')

    return fig


def plot_dual_profile(
    k_values: list[int],
    primary_scores: list[float],
    secondary_scores: list[float],
    primary_label: str = 'Correspondence',
    secondary_label: str = 'Entropy',
    ylim_primary: tuple[float, float] = (0, 1),
    ylim_secondary: tuple[float, float] = (0, None),
) -> go.Figure:
    """Create a dual-axis plot for primary score and entropy.

    Parameters
    ----------
    k_values : list[int]
        List of k values.
    primary_scores : list[float]
        Primary metric scores (e.g., hungarian).
    secondary_scores : list[float]
        Secondary metric scores (e.g., entropy).
    primary_label : str
        Label for primary metric.
    secondary_label : str
        Label for secondary metric.
    ylim_primary : tuple[float, float]
        Y-axis limits for primary.
    ylim_secondary : tuple[float, float]
        Y-axis limits for secondary.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive dual-axis figure.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{'secondary_y': True}]])

    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=primary_scores,
            mode='lines+markers',
            name=primary_label,
            line={'color': '#636EFA', 'width': 2},
            marker={'size': 6},
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=secondary_scores,
            mode='lines+markers',
            name=secondary_label,
            line={'color': '#FFA15A', 'dash': 'dot', 'width': 2},
            marker={'size': 6, 'symbol': 'diamond'},
        ),
        secondary_y=True,
    )

    fig.update_layout(
        **_PLOTLY_STYLE,
        xaxis_title='Prototypes',
        hovermode='x unified',
        height=400,
        margin={'t': 20, 'b': 50, 'l': 50, 'r': 50},
    )

    fig.update_yaxes(
        title_text=primary_label,
        range=list(ylim_primary),
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text=secondary_label,
        range=list(ylim_secondary),
        secondary_y=True,
    )

    return fig


_GROUP_PALETTE: list[tuple[int, int, int]] = [
    (72, 120, 208),
    (238, 133, 74),
    (78, 190, 130),
    (196, 78, 82),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (188, 189, 34),
    (23, 190, 207),
    (127, 127, 127),
]


def _make_cloud_trace(points: list, r: int, g: int, b: int):
    """Filled Scatter trace enclosing *points* with smooth bezier-rounded edges."""
    import numpy as _np
    import plotly.graph_objects as go

    pts = _np.array(points, dtype=float)
    n = len(pts)
    centroid = pts.mean(axis=0)

    # Adaptive expand: ~15 % of the point-cloud span, at least a small circle
    span = float(max(_np.ptp(pts, axis=0).max(), 1e-2))
    expand = span * 0.18
    corner_r = span * 0.12

    if n == 1:
        theta = _np.linspace(0, 2 * _np.pi, 64)
        xy = centroid + expand * _np.column_stack([_np.cos(theta), _np.sin(theta)])
    elif n == 2:
        d = pts[1] - pts[0]
        length = _np.linalg.norm(d)
        if length < 1e-8:
            d, length = _np.array([1.0, 0.0]), 1e-8
        else:
            d = d / length
        perp = _np.array([-d[1], d[0]])
        theta = _np.linspace(0, 2 * _np.pi, 64)
        xy = (
            centroid
            + (length / 2 + expand) * _np.cos(theta)[:, None] * d
            + expand * _np.sin(theta)[:, None] * perp
        )
    else:
        try:
            from scipy.spatial import ConvexHull

            verts = pts[ConvexHull(pts).vertices]
        except Exception:
            verts = pts

        # Expand hull vertices outward from centroid
        vecs = verts - centroid
        norms = _np.linalg.norm(vecs, axis=1, keepdims=True)
        verts = verts + expand * vecs / _np.where(norms > 1e-8, norms, 1.0)

        # Smooth each corner with a quadratic Bézier arc
        nv = len(verts)
        curve = []
        for vi in range(nv):
            p0 = verts[(vi - 1) % nv]
            pc = verts[vi]
            p1 = verts[(vi + 1) % nv]

            d_in = pc - p0
            d_out = p1 - pc
            li = _np.linalg.norm(d_in)
            lo = _np.linalg.norm(d_out)
            if li < 1e-8 or lo < 1e-8:
                curve.append(pc)
                continue

            t = min(corner_r, li / 2.5, lo / 2.5)
            arc_s = pc - t * (d_in / li)
            arc_e = pc + t * (d_out / lo)

            # Quadratic Bézier: arc_s → pc (control) → arc_e
            for s in _np.linspace(0, 1, 12):
                pt = (1 - s) ** 2 * arc_s + 2 * (1 - s) * s * pc + s**2 * arc_e
                curve.append(pt)

        xy = _np.array(curve)

    xy = _np.vstack([xy, xy[0]])
    return go.Scatter(
        x=xy[:, 0].tolist(),
        y=xy[:, 1].tolist(),
        fill='toself',
        fillcolor=f'rgba({r},{g},{b},0.13)',
        line={'color': f'rgba({r},{g},{b},0.45)', 'width': 1.5},
        mode='lines',
        showlegend=False,
        hoverinfo='skip',
    )


def _edge_alpha(j: float) -> float:
    return 0.2 + 0.8 * j


def _edge_width(j: float) -> float:
    return max(0.5, j * 8)


def plot_bipartite_merge(
    J: NDArray[np.float32],
    model_a: str,
    model_b: str,
    threshold: float = 0.0,
    matched_pairs: set[tuple[int, int]] | None = None,
) -> go.Figure:
    """Bipartite layout: model A prototypes on the left, model B on the right.

    Edge opacity follows ``alpha = 0.2 + 0.8 * j`` and width ``max(0.5, j * 8)``.
    A Jaccard label is drawn at each edge midpoint with the same alpha.
    Pairs in ``matched_pairs`` are highlighted in red.

    Parameters
    ----------
    J : np.ndarray, shape (k_a, k_b)
        Jaccard similarity matrix.
    model_a, model_b : str
        Model names used as axis labels.
    threshold : float
        Minimum Jaccard value to draw an edge. Default 0.0.
    matched_pairs : set of (int, int), optional
        Pre-computed (a_idx, b_idx) pairs to highlight in red.
        If None, no edges are highlighted.
    """
    import numpy as _np
    import plotly.graph_objects as go

    k_a, k_b = J.shape
    y_a = _np.linspace(1, 0, k_a)
    y_b = _np.linspace(1, 0, k_b)

    matching_pairs: set[tuple[int, int]] = (
        matched_pairs if matched_pairs is not None else set()
    )

    fig = go.Figure()

    for i in range(k_a):
        for jj in range(k_b):
            w = float(J[i, jj])
            if w <= threshold:
                continue
            alpha = _edge_alpha(w)
            width = _edge_width(w)
            is_match = (i, jj) in matching_pairs
            color = (
                f'rgba(196,78,82,{alpha:.3f})'
                if is_match
                else f'rgba(100,100,100,{alpha:.3f})'
            )

            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[float(y_a[i]), float(y_b[jj])],
                    mode='lines',
                    line={'color': color, 'width': width},
                    showlegend=False,
                    hoverinfo='skip',
                )
            )
            fig.add_annotation(
                x=0.5,
                y=(float(y_a[i]) + float(y_b[jj])) / 2,
                text=f'{w:.2f}',
                showarrow=False,
                xref='x',
                yref='y',
                font={
                    'family': 'Times New Roman',
                    'size': 8,
                    'color': f'rgba(40,40,40,{alpha:.3f})',
                },
            )

    show_text = k_a <= 24
    node_mode = 'markers+text' if show_text else 'markers'
    fig.add_trace(
        go.Scatter(
            x=[0] * k_a,
            y=y_a.tolist(),
            mode=node_mode,
            text=[str(i) for i in range(k_a)] if show_text else [],
            textposition='middle left',
            marker={'size': 8, 'color': '#4878d0'},
            name=model_a,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[1] * k_b,
            y=y_b.tolist(),
            mode=node_mode,
            text=[str(jj) for jj in range(k_b)] if show_text else [],
            textposition='middle right',
            marker={'size': 8, 'color': '#ee854a'},
            name=model_b,
        )
    )

    height = min(max(400, k_a * 18), 1400)
    fig.update_layout(
        **_PLOTLY_STYLE,
        height=height,
        margin={'t': 40, 'b': 20, 'l': 80, 'r': 80},
        xaxis={
            'showticklabels': False,
            'showgrid': False,
            'zeroline': False,
            'range': [-0.25, 1.25],
        },
        yaxis={'showticklabels': False, 'showgrid': False, 'zeroline': False},
        legend={'orientation': 'h', 'y': 1.05, 'x': 0.5, 'xanchor': 'center'},
    )
    for x_pos, label in [(0, model_a), (1, model_b)]:
        fig.add_annotation(
            x=x_pos,
            y=1.02,
            text=f'<b>{label}</b>',
            showarrow=False,
            xref='x',
            yref='paper',
            xanchor='center',
            font={'family': 'Times New Roman', 'size': 11},
        )
    return fig


def plot_force_graph(
    J: NDArray[np.float32],
    model_a: str,
    model_b: str,
    threshold: float = 0.0,
    matched_pairs: set[tuple[int, int]] | None = None,
    groups: list | None = None,
    seed: int = 42,
) -> go.Figure:
    """Force-directed layout of prototype correspondences between two models.

    Uses igraph Fruchterman-Reingold layout. Edge opacity follows
    ``alpha = 0.2 + 0.8 * j`` and width ``max(0.5, j * 8)``.
    A Jaccard label is drawn at each edge midpoint with the same alpha.
    Pairs in ``matched_pairs`` are highlighted in red.
    When ``groups`` (list of MatchGroup) is supplied, a smooth bezier-rounded
    cloud is drawn behind the nodes of each group in a distinct colour.

    Parameters
    ----------
    J : np.ndarray, shape (k_a, k_b)
        Jaccard similarity matrix.
    model_a, model_b : str
        Model names used in the legend.
    threshold : float
        Minimum Jaccard value to draw an edge. Default 0.0.
    matched_pairs : set of (int, int), optional
        Pre-computed (a_idx, b_idx) pairs to highlight in red.
        If None, no edges are highlighted.
    groups : list of MatchGroup, optional
        Matched groups; one smooth cloud per group, drawn behind edges and nodes.
    seed : int
        Random seed for reproducible layout initialisation. Default 42.
    """
    import igraph as ig
    import numpy as _np
    import plotly.graph_objects as go

    k_a, k_b = J.shape
    n = k_a + k_b

    # Build igraph: nodes 0..k_a-1 = A side, k_a..n-1 = B side
    edges_ig, weights_ig = [], []
    for i in range(k_a):
        for j in range(k_b):
            w = float(J[i, j])
            if w > threshold:
                edges_ig.append((i, k_a + j))
                weights_ig.append(w)

    G = ig.Graph(n=n, edges=edges_ig, directed=False)
    has_edges = len(weights_ig) > 0
    if has_edges:
        G.es['weight'] = weights_ig

    # Reproducible layout via fixed random initialisation
    rng = _np.random.default_rng(seed)
    initial = rng.uniform(-1, 1, (n, 2)).tolist()
    layout = G.layout_fruchterman_reingold(
        weights='weight' if has_edges else None,
        niter=500,
        seed=initial,
    )
    coords = _np.array([layout[i] for i in range(n)])  # (n, 2)

    matching_pairs: set[tuple[int, int]] = (
        matched_pairs if matched_pairs is not None else set()
    )

    fig = go.Figure()

    # Smooth bezier clouds — rendered first so they sit behind everything
    if groups:
        for gi, group in enumerate(groups):
            rc, gc, bc = _GROUP_PALETTE[gi % len(_GROUP_PALETTE)]
            pts = [coords[ai].tolist() for ai in group.a_clusters if ai < k_a] + [
                coords[k_a + bi].tolist() for bi in group.b_clusters if bi < k_b
            ]
            if pts:
                fig.add_trace(_make_cloud_trace(pts, rc, gc, bc))

    # Edges
    for e in G.es:
        src, tgt = e.source, e.target
        w = float(e['weight']) if has_edges else 1.0
        alpha = _edge_alpha(w)
        width = _edge_width(w)

        a_idx = src if src < k_a else tgt
        b_idx = (tgt - k_a) if tgt >= k_a else (src - k_a)
        color = (
            f'rgba(196,78,82,{alpha:.3f})'
            if (a_idx, b_idx) in matching_pairs
            else f'rgba(100,100,100,{alpha:.3f})'
        )

        x0, y0 = coords[src]
        x1, y1 = coords[tgt]
        fig.add_trace(
            go.Scatter(
                x=[float(x0), float(x1)],
                y=[float(y0), float(y1)],
                mode='lines',
                line={'color': color, 'width': width},
                showlegend=False,
                hoverinfo='skip',
            )
        )
        fig.add_annotation(
            x=(float(x0) + float(x1)) / 2,
            y=(float(y0) + float(y1)) / 2,
            text=f'{w:.2f}',
            showarrow=False,
            font={
                'family': 'Times New Roman',
                'size': 8,
                'color': f'rgba(40,40,40,{alpha:.3f})',
            },
        )

    # Nodes
    show_text = n <= 48
    node_mode = 'markers+text' if show_text else 'markers'
    for side_start, side_end, color, name, labels in [
        (0, k_a, '#4878d0', model_a, [str(i) for i in range(k_a)]),
        (k_a, n, '#ee854a', model_b, [str(j) for j in range(k_b)]),
    ]:
        xy = coords[side_start:side_end]
        fig.add_trace(
            go.Scatter(
                x=xy[:, 0].tolist(),
                y=xy[:, 1].tolist(),
                mode=node_mode,
                text=labels if show_text else [],
                textposition='top center',
                marker={'size': 8, 'color': color},
                name=name,
            )
        )

    fig.update_layout(
        **_PLOTLY_STYLE,
        height=600,
        margin={'t': 20, 'b': 20, 'l': 20, 'r': 20},
        xaxis={'showticklabels': False, 'showgrid': False, 'zeroline': False},
        yaxis={'showticklabels': False, 'showgrid': False, 'zeroline': False},
        legend={'orientation': 'h', 'y': 1.05, 'x': 0.5, 'xanchor': 'center'},
    )
    return fig


def plot_cluster_pair_images(
    cluster_a_indices,
    cluster_b_indices,
    img_datasets: list,
    dataset_boundaries: list[int],
    n_samples: int = 8,
    model_a: str = '',
    model_b: str = '',
    jaccard: float = 0.0,
    cluster_a: str | int = 0,
    cluster_b: str | int = 0,
    seed: int = 42,
):
    """Sample images from a matched cluster pair and render them as a 2-row grid.

    Parameters
    ----------
    cluster_a_indices : array-like of int
        Global sample indices belonging to model-A cluster.
    cluster_b_indices : array-like of int
        Global sample indices belonging to model-B cluster.
    img_datasets : list of (dataset, img_col_name | None)
        Raw HuggingFace datasets paired with their image column name.
    dataset_boundaries : list of int
        Cumulative sample counts, e.g. [0, 50000, 60000, ...].
    n_samples : int
        Number of images to sample per cluster row.
    model_a, model_b : str
        Model names used as y-axis labels.
    jaccard : float
        Jaccard similarity for this pair (used in title).
    cluster_a, cluster_b : int
        Cluster indices (used in title).
    seed : int
        Random seed for reproducible sampling.
    """
    import matplotlib as _mpl
    import matplotlib.pyplot as plt
    import numpy as _np

    _mpl.rcParams['font.family'] = 'Times New Roman'

    rng = _np.random.default_rng(seed)

    def _get_image(global_idx: int):
        ds_idx = int(_np.searchsorted(dataset_boundaries[1:], global_idx, side='right'))
        local_idx = global_idx - dataset_boundaries[ds_idx]
        ds, img_col = img_datasets[ds_idx]
        if ds is None or img_col is None:
            return None
        return ds[int(local_idx)][img_col]

    def _sample(indices, n: int) -> list:
        idx_list = list(indices)
        chosen = rng.choice(len(idx_list), size=min(n, len(idx_list)), replace=False)
        return [_get_image(idx_list[int(c)]) for c in chosen]

    imgs_a = _sample(cluster_a_indices, n_samples)
    imgs_b = _sample(cluster_b_indices, n_samples)

    n_cols = n_samples
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 1.4, 3.2))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for row, imgs in enumerate([imgs_a, imgs_b]):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis('off')
            if col < len(imgs) and imgs[col] is not None:
                arr = _np.array(imgs[col])
                ax.imshow(arr, cmap='gray' if arr.ndim == 2 else None)

    axes[0, 0].set_ylabel(model_a or 'A', fontsize=7)
    axes[1, 0].set_ylabel(model_b or 'B', fontsize=7)

    fig.suptitle(
        f'Cluster A{cluster_a} ↔ B{cluster_b}  (J={jaccard:.3f})', fontsize=9, y=1.01
    )
    fig.tight_layout(pad=0.3)
    return fig


def plot_cluster_grid_images(
    groups,
    indices_by_cluster: dict,
    img_datasets: list,
    dataset_boundaries: list[int],
    side: str = 'a',
    n_samples: int = 8,
    model_name: str = '',
    seed: int = 42,
):
    """Grid of images: rows = matched groups (same order for A and B), cols = samples.

    Pass the same ``groups`` list and the same ``seed`` for both model grids to
    guarantee row-level correspondence between the two figures.

    Parameters
    ----------
    groups : list of MatchGroup
        Matched groups; determines row ordering.
    indices_by_cluster : dict[int, np.ndarray]
        Cluster index → global sample indices for one model side.
    img_datasets : list of (dataset, img_col_name | None)
    dataset_boundaries : list of int
    side : {'a', 'b'}
        Which clusters to read from each group ('a' → group.a_clusters).
    n_samples : int
        Number of images per row.
    model_name : str
        Figure suptitle.
    seed : int
        Random seed for reproducible sampling.
    """
    import matplotlib as _mpl
    import matplotlib.pyplot as plt
    import numpy as _np

    _mpl.rcParams['font.family'] = 'Times New Roman'

    rng = _np.random.default_rng(seed)

    def _get_image(global_idx: int):
        ds_idx = int(_np.searchsorted(dataset_boundaries[1:], global_idx, side='right'))
        local_idx = global_idx - dataset_boundaries[ds_idx]
        ds, img_col = img_datasets[ds_idx]
        if ds is None or img_col is None:
            return None
        return ds[int(local_idx)][img_col]

    def _sample(indices, n: int) -> list:
        idx_list = list(indices)
        chosen = rng.choice(len(idx_list), size=min(n, len(idx_list)), replace=False)
        return [_get_image(idx_list[int(c)]) for c in chosen]

    n_groups = len(groups)
    n_cols = n_samples
    fig, axes = plt.subplots(n_groups, n_cols, figsize=(n_cols * 1.4, n_groups * 1.6))
    axes = _np.array(axes).reshape(n_groups, n_cols)

    for row, group in enumerate(groups):
        cluster_ids = group.a_clusters if side == 'a' else group.b_clusters
        combined = _np.concatenate(
            [indices_by_cluster[i] for i in cluster_ids if i in indices_by_cluster]
        )
        imgs = _sample(combined, n_cols)
        label = '+'.join(str(i) for i in cluster_ids)
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis('off')
            if col < len(imgs) and imgs[col] is not None:
                arr = _np.array(imgs[col])
                ax.imshow(arr, cmap='gray' if arr.ndim == 2 else None)
        axes[row, 0].set_ylabel(
            f'C{label}', fontsize=7, rotation=0, labelpad=28, va='center'
        )

    fig.suptitle(model_name, fontsize=10)
    fig.tight_layout(pad=0.3)
    return fig


__all__ = [
    'plot_prototype_heatmap',
    'plot_similarity_profile',
    'plot_dual_profile',
    'plot_bipartite_merge',
    'plot_force_graph',
    'plot_cluster_pair_images',
    'plot_cluster_grid_images',
]
