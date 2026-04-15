"""TDA feature extraction from persistence diagrams (scikit-tda / ripser format)."""

import numpy as np
import torch


def _finite(dgm: np.ndarray) -> np.ndarray:
    """Return only the rows with finite death value."""
    return dgm[np.isfinite(dgm[:, 1])]


# ---------------------------------------------------------------------------
# Diagram entropy
# ---------------------------------------------------------------------------

def compute_diagram_entropy(dgms: list[np.ndarray]) -> list[float]:
    """Compute persistent entropy for each homological dimension.

    Persistent entropy is defined as the Shannon entropy of the normalised
    lifetime distribution:

        H_d = -Σ_i (l_i / L) · log(l_i / L)

    where l_i = death_i - birth_i and L = Σ_i l_i, summed over finite
    diagram points.

    Parameters
    ----------
    dgms:
        List of persistence diagrams as returned by
        :func:`compute_persistence_diagram`.  ``dgms[d]`` has shape
        ``(n_pts_d, 2)`` with columns ``[birth, death]``.

    Returns
    -------
    list[float]
        One entropy value per homological dimension (same length as *dgms*).
    """
    entropies: list[float] = []
    for dgm in dgms:
        fd = _finite(dgm)
        if len(fd) == 0:
            entropies.append(0.0)
            continue
        lifetimes = fd[:, 1] - fd[:, 0]
        total = lifetimes.sum()
        if total == 0.0:
            entropies.append(0.0)
            continue
        p = lifetimes / total
        entropies.append(float(-np.sum(p * np.log(p))))
    return entropies


# ---------------------------------------------------------------------------
# Persistence image
# ---------------------------------------------------------------------------

def compute_persistence_image(
    dgms: list[np.ndarray],
    n_bins: int = 100,
    sigma: float = 0.1,
) -> torch.Tensor:
    """Compute persistence images and return them as a tensor.

    Each diagram point (b, d) with finite death is mapped to the
    (birth, persistence) half-plane and smeared with a Gaussian of width
    *sigma*, weighted by its persistence p = d - b.  The resulting density
    is sampled on a shared ``n_bins × n_bins`` grid whose range is determined
    by the union of all finite points across every homological dimension.

    Parameters
    ----------
    dgms:
        List of persistence diagrams (ripser format).
    n_bins:
        Grid resolution along each axis of the image.
    sigma:
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    torch.Tensor of shape ``(n_dims, n_bins, n_bins)``
        One persistence image per homological dimension.
        Axes are ``(birth_axis, persistence_axis)``.
    """
    # Collect all finite (birth, persistence) pairs to fix a shared grid range
    all_b: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    for dgm in dgms:
        fd = _finite(dgm)
        if len(fd):
            all_b.append(fd[:, 0])
            all_p.append(fd[:, 1] - fd[:, 0])

    if all_b:
        b_min = float(np.concatenate(all_b).min()) - sigma
        b_max = float(np.concatenate(all_b).max()) + sigma
        p_min = 0.0
        p_max = float(np.concatenate(all_p).max()) + sigma
    else:
        b_min, b_max, p_min, p_max = 0.0, 1.0, 0.0, 1.0

    b_grid = np.linspace(b_min, b_max, n_bins)  # (n_bins,)
    p_grid = np.linspace(p_min, p_max, n_bins)  # (n_bins,)

    images: list[torch.Tensor] = []
    for dgm in dgms:
        fd = _finite(dgm)
        if len(fd) == 0:
            images.append(torch.zeros(n_bins, n_bins, dtype=torch.float32))
            continue

        b = fd[:, 0]              # (n_pts,)
        p = fd[:, 1] - fd[:, 0]  # (n_pts,)

        # Vectorised Gaussian kernels:
        #   gb[i, j] = exp(-(b_grid[j] - b[i])^2 / 2σ²)  shape (n_pts, n_bins)
        #   gp[i, j] = exp(-(p_grid[j] - p[i])^2 / 2σ²)  shape (n_pts, n_bins)
        # Image = Σ_i  p[i] · gb[i, :] ⊗ gp[i, :]  →  einsum 'i,ij,ik->jk'
        inv2s2 = 1.0 / (2.0 * sigma**2)
        gb = np.exp(-((b_grid[np.newaxis, :] - b[:, np.newaxis]) ** 2) * inv2s2)
        gp = np.exp(-((p_grid[np.newaxis, :] - p[:, np.newaxis]) ** 2) * inv2s2)
        img = np.einsum('i,ij,ik->jk', p, gb, gp).astype(np.float32)

        images.append(torch.from_numpy(img))

    return torch.stack(images)  # (n_dims, n_bins, n_bins)


# ---------------------------------------------------------------------------
# Betti curve
# ---------------------------------------------------------------------------

def compute_betti_curve(
    dgms: list[np.ndarray],
    n_bins: int = 100,
) -> np.ndarray:
    """Compute Betti number curves for each homological dimension.

    At each filtration value t, the Betti number B_d(t) counts how many
    diagram points satisfy birth ≤ t < death (infinite death counts as
    always alive).

    Parameters
    ----------
    dgms:
        List of persistence diagrams (ripser format).
    n_bins:
        Number of filtration-parameter samples along the curve.

    Returns
    -------
    np.ndarray of shape ``(n_dims, n_bins)``
        One Betti curve per homological dimension.
    """
    # Build t-grid from the range of all births and finite deaths
    all_b = np.concatenate([d[:, 0] for d in dgms if len(d)]) if any(len(d) for d in dgms) else np.array([0.0])
    finite_deaths = [_finite(d)[:, 1] for d in dgms if len(_finite(d))]
    all_d = np.concatenate(finite_deaths) if finite_deaths else np.array([1.0])

    t_min = float(all_b.min())
    t_max = float(all_d.max())
    if t_min >= t_max:
        t_max = t_min + 1.0

    t_grid = np.linspace(t_min, t_max, n_bins)  # (n_bins,)

    curves: list[np.ndarray] = []
    for dgm in dgms:
        if len(dgm) == 0:
            curves.append(np.zeros(n_bins, dtype=np.float64))
            continue
        births = dgm[:, 0]  # (n_pts,)
        deaths = dgm[:, 1]  # (n_pts,) — may be inf
        # alive[i, j] = True if point i is alive at t_grid[j]
        alive = (
            births[:, np.newaxis] <= t_grid[np.newaxis, :]
        ) & (
            deaths[:, np.newaxis] > t_grid[np.newaxis, :]
        )
        curves.append(alive.sum(axis=0).astype(np.float64))

    return np.array(curves)  # (n_dims, n_bins)
