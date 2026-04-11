""""""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy as np

from .backends import backend_dispatch

__all__ = ['prototypical']


@backend_dispatch
def prototypical(
    X: np.ndarray | torch.Tensor,
    n_samples: int,
    *,
    clusters: np.ndarray | torch.Tensor | None = None,
    clusterer: object | None = None,
    clusterer_cls: callable[..., object] | None = None,
    n_clusters: int | None = None,
    clusterer_kwargs: dict | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Compute cluster prototypes by averaging random subsets of samples.

    This function supports three mutually exclusive clustering modes:

    1. **Precomputed clusters**: provide ``clusters``
    2. **External clusterer**: provide ``clusterer``
    3. **Internal clustering**: provide ``clusterer_cls``

    Internal clustering supports both:
    - clusterers that require ``n_clusters`` (e.g. KMeans)
    - clusterers that do not (e.g. DBSCAN)

    Parameters
    ----------
    X : numpy.ndarray or torch.Tensor, shape (N, D)
        Input data matrix.
    n_samples : int
        Number of samples randomly selected from each cluster to compute
        the prototype.
    clusters : numpy.ndarray or torch.Tensor, shape (N,), optional
        Precomputed cluster assignments.
    clusterer : object, optional
        External clustering object implementing ``fit_predict(X)``.
    clusterer_cls : callable, optional
        Clustering class or factory used for internal clustering.
        Must return an object implementing ``fit_predict(X)``.
    n_clusters : int, optional
        Number of clusters to create when required by the clustering algorithm.
        Ignored by algorithms that do not accept it.
    clusterer_kwargs : dict, optional
        Additional keyword arguments passed to ``clusterer_cls``.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    prototypes : numpy.ndarray or torch.Tensor, shape (K, D)
        Prototype vectors for each cluster.
    clusters : numpy.ndarray or torch.Tensor, shape (N,)
        Cluster assignments used to compute the prototypes.

    Raises
    ------
    ValueError
        If an invalid or ambiguous clustering configuration is provided.
    ValueError
        If a cluster contains fewer than ``n_samples`` samples.

    Notes
    -----
    - The output backend matches the input backend.
    - Noise labels (e.g. ``-1`` from DBSCAN) are treated as a valid cluster.
    - Sampling within clusters is performed without replacement.

    Examples
    --------
    Internal clustering with KMeans::

        protos, clusters = prototypical(
            X,
            n_samples=10,
            clusterer_cls=KMeans,
            n_clusters=5,
            seed=42,
        )

    Internal clustering with DBSCAN::

        protos, clusters = prototypical(
            X,
            n_samples=10,
            clusterer_cls=DBSCAN,
            clusterer_kwargs={'eps': 0.5},
        )
    """
    if seed is not None:
        torch.manual_seed(seed)

    clusterer_kwargs = clusterer_kwargs or {}

    # Validate clustering configuration
    modes = (
        clusters is not None,
        clusterer is not None,
        clusterer_cls is not None,
    )
    if sum(modes) != 1:
        raise ValueError(
            'Provide exactly one of `clusters`, `clusterer`, or `clusterer_cls`.'
        )

    # Resolve clusters
    if clusters is not None:
        clusters = torch.as_tensor(clusters, device=X.device)

    elif clusterer is not None:
        clusters = torch.as_tensor(
            clusterer.fit_predict(X),
            device=X.device,
        )

    else:
        # Internal clustering
        kwargs = dict(clusterer_kwargs)

        if n_clusters is not None:
            kwargs.setdefault('n_clusters', n_clusters)

        if seed is not None:
            kwargs.setdefault('random_state', seed)

        clusterer = clusterer_cls(**kwargs)
        clusters = torch.as_tensor(
            clusterer.fit_predict(X),
            device=X.device,
        )

    unique_clusters = torch.unique(clusters)
    prototypes = torch.empty(
        (unique_clusters.numel(), X.size(1)),
        device=X.device,
        dtype=X.dtype,
    )

    for i, c in enumerate(unique_clusters):
        inps = X[clusters == c]

        if inps.size(0) < n_samples:
            raise ValueError(
                f'Cluster {int(c)} has {inps.size(0)} samples, '
                f'but n_samples={n_samples}.'
            )

        idx = torch.randperm(inps.size(0), device=X.device)[:n_samples]
        prototypes[i] = inps[idx].mean(dim=0)

    return prototypes, clusters


@backend_dispatch
def parseval_frame(
    X: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Compute the analysis and synthesis operators of a Parseval frame.

    Given an input matrix ``X``, this function computes a Parseval frame
    via the singular value decomposition (SVD). The resulting operators
    satisfy the Parseval tight frame condition.

    Parameters
    ----------
    X : numpy.ndarray or torch.Tensor, shape (N, D)
        Input data matrix.

    Returns
    -------
    F : numpy.ndarray or torch.Tensor, shape (N, N)
        Analysis operator of the Parseval frame.
    G : numpy.ndarray or torch.Tensor, shape (N, N)
        Synthesis operator of the Parseval frame. For a Parseval frame,
        ``G`` is the Hermitian transpose of ``F``.

    Notes
    -----
    - The computation is backend-agnostic: NumPy inputs are dispatched
      to Torch internally and converted back.
    - The frame operators are computed as ``U @ Vᴴ``, where ``U`` and
      ``V`` come from the SVD of ``X``.
    - The resulting frame is tight and satisfies ``Fᴴ F = I``.

    Examples
    --------
    >>> F, G = parseval_frame(X)
    """
    U, _, Vh = torch.linalg.svd(X, full_matrices=False)

    # Analysis operator
    F = U @ Vh

    # Synthesis operator (Hermitian transpose)
    G = F.H

    return F, G


if __name__ == '__main__':
    pass
