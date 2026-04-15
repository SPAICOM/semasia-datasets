import numpy as np
import torch

from src.tda.features import (
    compute_betti_curve,
    compute_diagram_entropy,
    compute_persistence_image,
)
from src.tda.persistence import (
    DimReductionMethod,
    NormalizeMethod,
    SimplicialFilter,
    compute_persistence_diagram,
)


def compute_tda_signature(
    latent: torch.Tensor | np.ndarray,
    max_dim: int = 4,
    simplicial_filter: SimplicialFilter = 'VietorisRips',
    n_bins: int = 100,
    sigma: float = 0.1,
    max_points: int = 1000,
    seed: int = 42,
    normalize: NormalizeMethod | None = None,
    dim_reduction: DimReductionMethod | None = None,
    dim_reduction_components: int = 50,
    **filtration_kwargs,
) -> dict[str, list]:
    """Compute TDA signatures from a latent-space point cloud.

    All n_points rows of *latent* are treated as a single point cloud living
    in the embedding space.  The function runs a simplicial filtration, then
    extracts four topological descriptors and returns them in a flat dict
    whose keys match :data:`TDA_KEYS`.

    Parameters
    ----------
    latent:
        Tensor / array of shape ``(n_points, n_features)`` – typically the
        full set of embeddings for one model × dataset × split triplet.
    max_dim:
        Maximum homological dimension (inclusive). Dimensions 0, 1, …,
        ``max_dim`` are computed.
    simplicial_filter:
        Filtration to use when building the simplicial complex.
        Currently ``'VietorisRips'`` (via ripser).
    n_bins:
        Grid resolution for the persistence image (both axes) and the
        Betti number curve.
    sigma:
        Gaussian kernel width for the persistence image.
    normalize:
        Optional point-cloud normalisation applied before dim reduction /
        filtration.  Choices: ``'standard'``, ``'minmax'``, ``'l2'``,
        ``None`` (default — no normalisation).
    dim_reduction:
        Optional dimensionality reduction applied after normalisation.
        Choices: ``'pca'``, ``'umap'``, ``'tsne'``, ``None`` (default).
    dim_reduction_components:
        Target number of dimensions when *dim_reduction* is not ``None``.
    **filtration_kwargs:
        Extra keyword arguments forwarded to the backend
        (e.g. ``metric='cosine'``, ``thresh=2.0``, ``coeff=3``).

    Returns
    -------
    dict with keys matching :data:`TDA_KEYS`:

    ``'persistence_diagram'``
        ``list[list[float]]`` of shape ``(n_diagram_points, 3)`` –
        each inner list is ``[birth, death, homology_dimension]``.
    ``'diagram_entropy'``
        ``list[float]`` of length ``max_dim + 1`` –
        persistent entropy per homological dimension.
    ``'persistence_image'``
        ``list[list[list[float]]]`` of shape
        ``(max_dim + 1, n_bins, n_bins)`` – one image per dimension,
        reconstructable via ``torch.tensor(row['persistence_image'])``.
    ``'betti_curve'``
        ``list[list[float]]`` of shape ``(max_dim + 1, n_bins)`` –
        Betti number curve per homological dimension.
    """
    # dgms[d] shape: (n_pts_d, 2) with [birth, death], death may be inf
    dgms = compute_persistence_diagram(
        latent,
        max_dim=max_dim,
        simplicial_filter=simplicial_filter,
        max_points=max_points,
        seed=seed,
        normalize=normalize,
        dim_reduction=dim_reduction,
        dim_reduction_components=dim_reduction_components,
        **filtration_kwargs,
    )

    entropy = compute_diagram_entropy(dgms)
    image = compute_persistence_image(dgms, n_bins=n_bins, sigma=sigma)
    betti = compute_betti_curve(dgms, n_bins=n_bins)

    # Stack diagrams from all dimensions into a single (n_pts, 3) array
    # with columns [birth, death, homology_dim] — a self-contained format
    # that preserves dimension labels and round-trips cleanly through parquet.
    parts = [
        np.column_stack([dgm, np.full(len(dgm), dim, dtype=np.float32)])
        for dim, dgm in enumerate(dgms)
        if len(dgm)
    ]
    flat_diagram: list = np.vstack(parts).tolist() if parts else []

    # Wrap every value in […] so polars treats each as a single list-valued
    # cell rather than interpreting the list length as the row count.
    return {
        'persistence_diagram': [flat_diagram],
        'diagram_entropy': [entropy],
        'persistence_image': [image.tolist()],
        'betti_curve': [betti.tolist()],
    }
