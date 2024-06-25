from hydra.utils import instantiate
import torch

def build_optim(cfg, model):
    optimizer, scheduler, criterion = None, None, None
    if hasattr(cfg, "optimizer"):
        optimizer = instantiate(cfg.optimizer, model.parameters())
    if hasattr(cfg, "scheduler"):
        scheduler = instantiate(cfg.scheduler.scheduler, optimizer)
    if hasattr(cfg, "loss"):
        criterion = instantiate(cfg.loss)  # , pos_weight=torch.ones([1], device='cuda')*2)

    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


class L1Aggregated(torch.nn.modules.loss._Loss):
    def __init__(
        self,
    ):
        super().__init__()
        self.loss = torch.nn.L1Loss()

    def forward(
        self,
        preds,
        labels,
    ):
        other_loss = self.loss(preds['other_preds'], labels[:, 0, :])
        tin_loss = self.loss(preds['tin_preds'], labels[:, 1, :])
        thatch_loss = self.loss(preds['thatch_preds'], labels[:, 2, :])

        return (other_loss + tin_loss + thatch_loss) / 3