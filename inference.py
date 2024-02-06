import torch

from data import FashionMnistDataModule

import matplotlib.pyplot as plt

import os


def get_latest_checkpoint():
    latest_checkpoint = None
    latest_timestamp = 0

    for root, dirs, _ in os.walk("./logs"):
        if "checkpoints" in dirs:
            checkpoint_dir = os.path.join(root, "checkpoints")
            checkpoint_files = os.listdir(checkpoint_dir)
            for file in checkpoint_files:
                file_path = os.path.join(checkpoint_dir, file)
                if os.path.isfile(file_path):
                    timestamp = os.path.getmtime(file_path)
                    if timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_checkpoint = file_path

    return latest_checkpoint


if __name__ == "__main__":
    with torch.no_grad():
        input_size = (224, 224)
        patch_size = 32

        val_fraction = 0.1

        embed_dim = 512
        num_heads = 8
        num_layers = 8

        learning_rate = 4e-4

        epochs = 5
        batch_size = 4
        num_workers = 18

        # properly utilize tensor cores
        torch.set_float32_matmul_precision("medium")

        data = FashionMnistDataModule(
            resize=input_size,
            val_fraction=val_fraction,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        file = get_latest_checkpoint()
        assert file is not None, "Could not find checkpoints"
        model = torch.load("./out/model").to("cuda")
        model.eval()

        loader = data.test_dataloader()
        images, labels = next(iter(loader))
        images = images.to("cuda")

        out = model(images)  # [batch, classes]

        out_classes = torch.argmax(out)

        fig, axs = plt.subplots(2, batch_size // 2)

        def plot_image(i, j, image, label):
            image = image.permute((1, 2, 0))
            image = image.cpu().numpy()
            label = label.cpu().numpy().item()
            label = data.classes()[label]
            ax = axs[i][j]
            ax.imshow(image, cmap="gray")
            ax.set_title(label)

        plot_image(0, 0, images[0], labels[0])
        plot_image(0, 1, images[1], labels[1])
        plot_image(1, 0, images[2], labels[2])
        plot_image(1, 1, images[3], labels[3])

        plt.show()
