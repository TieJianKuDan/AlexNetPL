import os
import sys

sys.path.append(os.getcwd())

import data
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, loggers, callbacks, seed_everything

from core import net


def main(flag, checkpoint=None):
    torch.set_float32_matmul_precision("medium")
    config = OmegaConf.load(
        open("scripts/config.yaml", "r")
    )

    seed_everything(config.seed, True)

    dm = data.MinistLDM(
        config.dataset.val_ratio,
        config.optim.batch_size,
        6
    )
    dm.setup(None)

    model = net.AlexNetPL(config, dm.sample_num)

    trainer = Trainer(
        max_epochs=config.optim.max_epochs,
        accelerator="gpu",
        logger=[
            loggers.CSVLogger(
                "./logs/csv",
                name="mnist",
                flush_logs_every_n_steps=500
            ),
            loggers.TensorBoardLogger(
                "./logs/tb",
                name="mnist",
            )
        ],
        precision="32",
        enable_checkpointing=True,
        callbacks=[
            callbacks.ModelCheckpoint(
                dirpath="checkpoints",
                monitor="val_loss"
            ),
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
            )
        ],
        deterministic="warn"
    )
    
    if flag == "train":
        trainer.fit(
            model, dm,
            # ckpt_path="./checkpoints/latest.ckpt"
        )
    elif flag == "test" \
        and checkpoint is not None:
        trainer.test(
            model, dm, 
            ckpt_path=checkpoint
        )
if __name__ == "__main__":
    main("train")