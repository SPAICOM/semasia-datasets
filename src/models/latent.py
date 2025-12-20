""""""

import pytorch_lightning as pl


class LatentExtractor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # inference-only
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Inference step: returns hidden latents.
        Lightning handles no_grad + device placement.
        """
        x = self.model(batch)
        return x.detach().cpu()


def main() -> None:
    """"""
    return None


if __name__ == '__main__':
    main()
