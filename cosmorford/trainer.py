# cosmorford/trainer.py
import torch
from lightning.pytorch.cli import ArgsType, LightningCLI
from lightning import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision("medium")


class CustomSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        for logger in trainer.loggers:
            if issubclass(type(logger), WandbLogger):
                logger.experiment.config.update(self.config.as_dict())
        return super().save_config(trainer, pl_module, stage)


def trainer_cli(args: ArgsType = None, run: bool = True):
    return LightningCLI(
        args=args,
        run=run,
        save_config_kwargs={"overwrite": True},
        save_config_callback=CustomSaveConfigCallback,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    trainer_cli(run=True)
