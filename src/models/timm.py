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


def load_model(
    model_name: str,
    device: str,
) -> Module:
    """Load a timm model as an embedding extractor (no classifier)."""
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.to(device)
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
