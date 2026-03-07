"""EMA weight averaging callback, ported from neurips-wl-challenge."""
from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn


class EMAWeightAveraging(WeightAveraging):
    def __init__(self, decay: float = 0.99, start_step: int = 100):
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay))
        self._start_step = start_step

    def should_update(self, step_idx=None, epoch_idx=None):
        return (step_idx is not None) and (step_idx >= self._start_step)
