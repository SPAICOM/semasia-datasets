""""""

import timm
import torch
from torch.nn import Module


@torch.no_grad()
def call_model(
    batch: dict[str, torch.Tensor],
    model: Module,
    device: str,
) -> dict[str, torch.Tensor]:
    """Forward pass to get encodings."""
    sample_encodings = model(batch['encoding'].to(device))
    return {'hidden': sample_encodings}


def check_embedding(
    latents: torch.Tensor,
    model_name: str,
    expected_dim: int | None,
) -> None:
    """Raise ``ValueError`` if the embeddings cannot be written to a dataset.

    The embedding must be 2-D, non-empty (length > 0), finite, and its length must
    equal the registry ``latent_dim`` when that value is known.
    """
    if latents.ndim != 2 or latents.shape[0] == 0:
        raise ValueError(
            f'{model_name}: unexpected embedding shape {tuple(latents.shape)}'
        )
    if latents.shape[1] == 0:
        raise ValueError(f'{model_name}: empty embedding (length 0)')
    if expected_dim is not None and latents.shape[1] != expected_dim:
        raise ValueError(
            f'{model_name}: embedding length {latents.shape[1]} != registry latent_dim '
            f'{expected_dim}'
        )
    if not torch.isfinite(latents).all():
        raise ValueError(f'{model_name}: non-finite values in the embedding')


def load_model(
    model_name: str,
) -> Module:
    """Load a timm model as an embedding extractor (no classifier)."""
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    # InceptionNeXt: num_classes=0 leaves head.fc2 = Linear(hidden, 0), so the
    # forward returns an empty (B, 0) embedding. Drop it to keep the hidden pre-logits.
    fc2 = getattr(getattr(model, 'head', None), 'fc2', None)
    if isinstance(fc2, torch.nn.Linear) and fc2.out_features == 0:
        model.head.fc2 = torch.nn.Identity()
    return model.requires_grad_(False).eval()


def encoder_params_below_threshold(model_name: str, threshold: int) -> bool:
    """
    Check whether a timm model encoder has fewer parameters than a given threshold.

    The encoder is defined as the model instantiated with ``num_classes=0``,
    which removes the classification head and keeps only the backbone.

    Parameters
    ----------
    model_name : str
        Name of the timm model (e.g., ``"resnet50"``,
        ``"vit_base_patch16_224"``).
    threshold : int
        Maximum allowed number of parameters.

    Returns
    -------
    bool
        ``True`` if the encoder parameter count is less than or equal to
        ``threshold``, ``False`` otherwise.

    Notes
    -----
    - No pretrained weights are downloaded.
    - The parameter count depends on the exact model variant.
    - Model instantiation is required to compute parameter counts.

    Examples
    --------
    >>> encoder_params_below_threshold('resnet50', 25_000_000)
    True

    >>> encoder_params_below_threshold('vit_large_patch16_224', 25_000_000)
    False
    """
    model = timm.create_model(model_name, pretrained=False, num_classes=0)

    num_params = sum(p.numel() for p in model.parameters())
    return num_params <= threshold


def main() -> None:
    """"""
    return None


if __name__ == '__main__':
    main()
