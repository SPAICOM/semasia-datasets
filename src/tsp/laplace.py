"""Graph Laplacian matrix and spectral decomposition."""

from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

LaplacianType = Literal['unnormalized', 'symmetric', 'random_walk']


def compute_laplacian(
    adjacency: np.ndarray | sp.spmatrix,
    normalization: LaplacianType = 'symmetric',
) -> np.ndarray | sp.spmatrix:
    """Compute the graph Laplacian of an undirected weighted graph.

    Parameters
    ----------
    adjacency : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric adjacency matrix of the graph.
    normalization : {'unnormalized', 'symmetric', 'random_walk'}
        Laplacian variant to compute:

        - ``'unnormalized'``: L = D − A
        - ``'symmetric'``:    L_sym = I − D^{−1/2} A D^{−1/2}
        - ``'random_walk'``:  L_rw  = I − D^{−1} A

    Returns
    -------
    np.ndarray or sp.spmatrix, shape (n, n)
        Laplacian matrix in the same sparse/dense format as *adjacency*.
        ``'symmetric'`` and ``'unnormalized'`` variants are symmetric PSD;
        ``'random_walk'`` is generally non-symmetric.
    """
    is_sparse = sp.issparse(adjacency)

    if is_sparse:
        A = adjacency.astype(np.float64).tocsr()
        degrees = np.asarray(A.sum(axis=1)).ravel()

        match normalization:
            case 'unnormalized':
                return sp.diags(degrees, format='csr') - A
            case 'symmetric':
                d_inv_sqrt = np.where(degrees > 0, degrees ** -0.5, 0.0)
                D_inv_sqrt = sp.diags(d_inv_sqrt)
                return sp.eye(A.shape[0], format='csr') - D_inv_sqrt @ A @ D_inv_sqrt
            case 'random_walk':
                d_inv = np.where(degrees > 0, 1.0 / degrees, 0.0)
                return sp.eye(A.shape[0], format='csr') - sp.diags(d_inv) @ A
            case _:
                raise ValueError(
                    f'Unknown normalization {normalization!r}. '
                    "Choices: 'unnormalized', 'symmetric', 'random_walk'."
                )

    A = np.asarray(adjacency, dtype=np.float64)
    degrees = A.sum(axis=1)

    match normalization:
        case 'unnormalized':
            return np.diag(degrees) - A
        case 'symmetric':
            d_inv_sqrt = np.where(degrees > 0, degrees ** -0.5, 0.0)
            D_inv_sqrt = np.diag(d_inv_sqrt)
            return np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
        case 'random_walk':
            d_inv = np.where(degrees > 0, 1.0 / degrees, 0.0)
            return np.eye(A.shape[0]) - np.diag(d_inv) @ A
        case _:
            raise ValueError(
                f'Unknown normalization {normalization!r}. '
                "Choices: 'unnormalized', 'symmetric', 'random_walk'."
            )


def compute_eigenvectors(
    laplacian: np.ndarray | sp.spmatrix,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the *k* smallest eigenpairs of a symmetric Laplacian.

    Designed for use with ``'unnormalized'`` and ``'symmetric'`` Laplacians,
    both of which are positive semi-definite.  For ``'random_walk'``
    Laplacians (non-symmetric) convert to the symmetric form first, or solve
    the generalised eigenvalue problem ``L v = λ D v`` externally.

    Uses a shift-invert strategy (``sigma=0``) for numerical stability when
    extracting eigenvectors near λ = 0.

    Parameters
    ----------
    laplacian : np.ndarray or sp.spmatrix, shape (n, n)
        Symmetric positive semi-definite Laplacian matrix.
    k : int
        Number of eigenpairs to return.  Clamped to ``[1, n]``.

    Returns
    -------
    eigenvalues : np.ndarray, shape (k,)
        The *k* smallest eigenvalues sorted in ascending order.
    eigenvectors : np.ndarray, shape (n, k)
        Corresponding unit-norm eigenvectors stored as columns.
    """
    n = laplacian.shape[0]

    if sp.issparse(laplacian):
        k_eff = min(k, n - 1)
        L = laplacian.tocsr()
        vals, vecs = eigsh(L, k=k_eff, sigma=0.0, which='LM')
    else:
        k_eff = min(k, n)
        L = np.asarray(laplacian, dtype=np.float64)
        vals, vecs = eigh(L, subset_by_index=[0, k_eff - 1])

    order = np.argsort(vals)
    return vals[order].astype(np.float64), vecs[:, order].astype(np.float64)


__all__ = ['compute_laplacian', 'compute_eigenvectors', 'LaplacianType']
