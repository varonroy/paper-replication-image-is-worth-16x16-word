from pytorch_lightning.loggers import CSVLogger
import torch
import torch.nn as nn
import torch.optim as optim

import torchmetrics

from pytorch_lightning import LightningModule, Trainer

from model import Model
from data import FashionMnistDataModule


class ModelModule(LightningModule):
    def __init__(self, model, num_classes, learning_rate):
        super().__init__()
        # self.save_hyperparameters() # TODO: fix this

        self.model = model.to("cuda")

        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.acc_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, log_mode=None):
        x, label = batch

        x = x.to(self.device)
        label = label.to(self.device)

        out = self.forward(x)

        loss = self.loss_fn(
            out.view(-1, self.num_classes),
            label.view(-1),
        )
        acc = self.acc_metric(out, label).float().mean()

        if log_mode is not None:
            self.log_dict(
                {
                    f"{log_mode}-loss": loss.item(),
                    f"{log_mode}-acc": acc.item(),
                }
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
            }
        ]


if __name__ == "__main__":
    input_size = (224, 224)
    patch_size = 32

    val_fraction = 0.1

    embed_dim = 512
    num_heads = 8
    num_layers = 8

    learning_rate = 4e-4

    epochs = 5
    batch_size = 128
    num_workers = 18

    # properly utilize tensor cores
    torch.set_float32_matmul_precision("medium")

    data = FashionMnistDataModule(
        resize=input_size,
        val_fraction=val_fraction,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = Model(
        num_classes=len(data.classes()),
        in_channels=data.num_channels(),
        patch_size=patch_size,
        image_size=input_size[0],
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        device="cuda",
    )

    model = ModelModule(
        model,
        num_classes=len(data.classes()),
        learning_rate=learning_rate,
    )

    trainer = Trainer(
        min_epochs=1,
        max_epochs=epochs,
        logger=[CSVLogger("logs")],
    )
    trainer.fit(model, data)
    trainer.test(model, data)
    torch.save(model.model, "./out/model")
