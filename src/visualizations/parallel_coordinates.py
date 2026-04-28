import io

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def make_parallel_coordinates(
    df,
    pc_method,
    n_dims,
    figsize=None,
    label_col='dataset',
    selected_labels=None,
    label_names=None,
):
    df = df.to_pandas()
    pc_cols = [f'PC{i}' for i in range(n_dims, 0, -1)]
    n = len(pc_cols)

    all_labels = sorted(df[label_col].unique())
    n_labels = len(all_labels)
    palette = list(cm.tab10.colors)
    label_color = {lbl: palette[i % len(palette)] for i, lbl in enumerate(all_labels)}

    if label_names is None:
        label_names = {lbl: str(lbl) for lbl in all_labels}

    norms = {}
    for col in pc_cols:
        mn, mx = df[col].min(), df[col].max()
        norms[col] = {'min': mn, 'max': mx, 'norm': (df[col] - mn) / (mx - mn + 1e-8)}

    if figsize is None:
        figsize = (max(10, n * 1.8 + 2), 9)

    fig, ax = plt.subplots(figsize=figsize)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    lx = -1.0
    ax.set_xlim(lx - 0.2, n - 0.5)
    ax.set_ylim(-0.12, 1.15)

    lbl_ypos = {
        lbl: i / (n_labels - 1) if n_labels > 1 else 0.5
        for i, lbl in enumerate(all_labels)
    }

    draw_set = set(all_labels) if selected_labels is None else set(selected_labels)

    def bezier_path(xs, ys, n_pts=30):
        t = np.linspace(0, 1, n_pts)
        px, py = [], []
        for j in range(len(xs) - 1):
            x0, y0, x1, y1 = xs[j], ys[j], xs[j + 1], ys[j + 1]
            cx0 = x0 + (x1 - x0) * 0.2
            cx1 = x0 + (x1 - x0) * 0.8
            bx = (
                (1 - t) ** 3 * x0
                + 3 * (1 - t) ** 2 * t * cx0
                + 3 * (1 - t) * t**2 * cx1
                + t**3 * x1
            )
            by = (
                (1 - t) ** 3 * y0
                + 3 * (1 - t) ** 2 * t * y0
                + 3 * (1 - t) * t**2 * y1
                + t**3 * y1
            )
            px.extend(bx[1:] if j > 0 else bx)
            py.extend(by[1:] if j > 0 else by)
        return px, py

    for idx in range(len(df)):
        lbl = df[label_col].iloc[idx]
        if lbl not in draw_set:
            continue
        ys = [lbl_ypos[lbl]] + [norms[col]['norm'].iloc[idx] for col in pc_cols]
        xs = [lx] + list(range(n))
        px, py = bezier_path(xs, ys)
        ax.plot(px, py, c=label_color[lbl], lw=0.7, alpha=0.1)

    # label axis — always shows all labels, dims non-selected ones
    ax.plot([lx, lx], [0.0, 1.0], color='#999', lw=0.8, ls='--')
    ax.text(lx, 1.09, 'Label', ha='center', va='bottom', fontsize=8, color='#666')
    for lbl in all_labels:
        yp = lbl_ypos[lbl]
        c = label_color[lbl]
        name = label_names.get(lbl, str(lbl))
        is_active = selected_labels is None or lbl in draw_set
        ax.plot([lx - 0.03, lx + 0.03], [yp, yp], color='#999', lw=0.8)
        ax.text(
            lx - 0.07,
            yp,
            name,
            ha='right',
            va='center',
            fontsize=8,
            color=c,
            fontweight='bold',
            alpha=1.0 if is_active else 0.25,
        )

    for i, col in enumerate(pc_cols):
        mn = norms[col]['min']
        mx = norms[col]['max']
        ax.plot([i, i], [0.0, 1.0], 'k-', lw=1.0)
        ax.text(i, 1.09, col, ha='center', va='bottom', fontsize=9)
        for tick in np.linspace(0, 1, 5):
            val = mn + tick * (mx - mn)
            ax.plot([i - 0.04, i + 0.04], [tick, tick], 'k-', lw=0.7)
            ax.text(i - 0.07, tick, f'{val:.2f}', ha='right', va='center', fontsize=6.5)

    return fig


def to_pdf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight')
    buf.seek(0)
    return buf.read()
