import torch


class CosineDecayWarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float,
        max_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            lr = self.max_lr * (step + 1) / self.warmup_steps
        elif step < self.max_steps:
            decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            coeff = 0.5 * (1.0 + torch.cos(torch.pi * torch.tensor(decay_ratio)))
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
        else:
            lr = self.min_lr

        return [lr for _ in self.base_lrs]
