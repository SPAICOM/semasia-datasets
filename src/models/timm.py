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


def main() -> None:
    """"""
    return None


if __name__ == '__main__':
    main()
