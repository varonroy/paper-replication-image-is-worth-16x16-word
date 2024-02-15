import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


import torchmetrics

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import LightningModule, Trainer

from model import Model, ModelConfig
from data import FashionMnistDataModule

import os


class ModelModule(LightningModule):
    def __init__(self, model_config, learning_rate):
        super().__init__()
        self.model_config = model_config

        self.save_hyperparameters()

        self.model = Model(model_config)

        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.acc_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=model_config.num_classes
        )

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, log_mode=None):
        x, label = batch

        x = x.to(self.device)
        label = label.to(self.device)

        out = self.forward(x)

        loss = self.loss_fn(
            out.view(-1, self.model_config.num_classes),
            label.view(-1),
        )
        acc = self.acc_metric(out, label).float().mean()

        if log_mode is not None:
            self.log_dict(
                {
                    f"{log_mode}-loss": loss.item(),
                    f"{log_mode}-acc": acc.item(),
                },
                sync_dist=True,
            )

        return loss

    def training_step(self, batch):
        return self.common_step(batch, log_mode="train")

    def validation_step(self, batch):
        return self.common_step(batch, log_mode="validation")

    def test_step(self, batch):
        return self.common_step(batch, log_mode="test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return [
            {
                "optimizer": optimizer,
                "scheduler": ReduceLROnPlateau(optimizer),
            }
        ]


if __name__ == "__main__":
    IMAGE_SIZE = 224
    PATCH_SIZE = 16

    if not os.path.isdir("./out"):
        print("`./out` dir doesn't exist")
        exit(-1)

    val_fraction = 0.1

    embed_dim = 512
    num_heads = 8
    num_layers = 8

    learning_rate = 5e-4

    epochs = 15
    batch_size = 256
    num_workers = 24

    # properly utilize tensor cores
    torch.set_float32_matmul_precision("medium")

    data = FashionMnistDataModule(
        resize=(224, 224),
        val_fraction=val_fraction,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model_config = ModelConfig(
        num_classes=len(data.classes()),
        in_channels=data.num_channels(),
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    model = ModelModule(
        model_config,
        learning_rate=learning_rate,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        min_epochs=1,
        max_epochs=epochs,
        logger=[CSVLogger("logs")],
        callbacks=[lr_monitor],
    )
    trainer.fit(model, data)
    trainer.test(model, data)
    torch.save(model.model, "./out/model")
