import numpy as np

from src.tda.features import (
    compute_betti_curve,
    compute_diagram_entropy,
    compute_persistence_image,
)
from src.tda.persistence import compute_persistence_diagram


def compute_tda_features(
    latent: np.ndarray,
    max_dim: int = 2,
    simplicial_filter: str = 'VietorisRips',
    n_bins: int = 100,
    sigma: float = 0.1,
    **filtration_kwargs,
) -> dict[str, list]:
    """Compute TDA features from a pre-processed latent point cloud.

    Parameters
    ----------
    latent : np.ndarray, shape (n_points, n_features)
        Pre-processed point cloud.
    max_dim : int
        Maximum homological dimension.
    simplicial_filter : str
        Filtration method.
    n_bins : int
        Grid resolution for persistence images and Betti curves.
    sigma : float
        Gaussian kernel width for persistence images.
    **filtration_kwargs
        Extra kwargs for ripser (e.g., metric, thresh).

    Returns
    -------
    dict with keys matching TDA_KEYS:
        - persistence_diagram: flat list of [birth, death, dim]
        - diagram_entropy: list of entropy per dimension
        - persistence_image: nested list of images per dimension
        - betti_curve: nested list of curves per dimension
    """
    dgms = compute_persistence_diagram(
        latent,
        max_dim=max_dim,
        simplicial_filter=simplicial_filter,
        **filtration_kwargs,
    )

    entropy = compute_diagram_entropy(dgms)
    image = compute_persistence_image(dgms, n_bins=n_bins, sigma=sigma)
    betti = compute_betti_curve(dgms, n_bins=n_bins)

    parts = [
        np.column_stack([dgm, np.full(len(dgm), dim, dtype=np.float32)])
        for dim, dgm in enumerate(dgms)
        if len(dgm)
    ]
    flat_diagram: list = np.vstack(parts).tolist() if parts else []

    return {
        'persistence_diagram': [flat_diagram],
        'diagram_entropy': [entropy],
        'persistence_image': [image.tolist()],
        'betti_curve': [betti.tolist()],
    }
