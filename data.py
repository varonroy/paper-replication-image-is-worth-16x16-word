from abc import ABC, abstractmethod

import torch.utils.data
import torchvision.transforms

from pytorch_lightning import LightningDataModule

from datasets import DatasetDict, load_dataset


class DatasetInfo(ABC):
    @abstractmethod
    def classes(self):
        raise Exception

    @abstractmethod
    def num_channels(self):
        raise Exception


class FashionMnistDataset(torch.utils.data.Dataset, DatasetInfo):
    def __init__(self, dataset, resize) -> None:
        super().__init__()
        self.dataset = dataset
        self.resize = torchvision.transforms.Resize(resize)

    def classes(self):
        return [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

    def num_channels(self):
        return 1

    @staticmethod
    def load(resize):
        dataset = load_dataset("fashion_mnist").with_format("torch")
        assert type(dataset) == DatasetDict
        train, test = dataset["train"], dataset["test"]
        return FashionMnistDataset(train, resize), FashionMnistDataset(test, resize)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset.__getitem__(idx)
        image, label = item["image"], item["label"]
        image = (image.to(dtype=torch.float32) - 120) / 120
        image = image.unsqueeze(0)
        image = self.resize(image)
        return image, label


class FashionMnistDataModule(LightningDataModule, DatasetInfo):
    def __init__(self, resize, val_fraction, batch_size, num_workers) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        train, test = FashionMnistDataset.load(resize)

        val_size = int(len(train) * val_fraction)
        train, val = torch.utils.data.random_split(
            train,
            [len(train) - val_size, val_size],
        )

        self.train_dataset = train
        self.validation_dataset = val
        self.test_dataset = test

    def classes(self):
        return self.test_dataset.classes()

    def num_channels(self):
        return self.test_dataset.num_channels()

    def create_dataloader(self, dataset, shuffle):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self.create_dataloader(self.train_dataset, True)

    def val_dataloader(self):
        return self.create_dataloader(self.validation_dataset, False)

    def test_dataloader(self):
        return self.create_dataloader(self.test_dataset, False)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = load_dataset("fashion_mnist").with_format("torch")
    assert type(dataset) == DatasetDict
    train, test = dataset["train"], dataset["test"]

    data_module = FashionMnistDataModule((224, 224), 0.1, 1, 1)
    classes = data_module.classes()

    fig, axs = plt.subplots(2, 2)

    ax = axs[0][0]
    item = train.__getitem__(0)
    image, label = item["image"], item["label"]
    ax.imshow(image, cmap="gray")
    ax.set_title(classes[label])

    ax = axs[0][1]
    item = test.__getitem__(500)
    image, label = item["image"], item["label"]
    ax.imshow(image, cmap="gray")
    ax.set_title(classes[label])

    ax = axs[1][0]
    item = next(iter(data_module.train_dataloader()))
    image, label = item[0].squeeze(0), item[1].squeeze(0)
    image = image.permute((1, 2, 0))
    ax.imshow(image, cmap="gray")
    ax.set_title(classes[label])

    ax = axs[1][1]
    item = next(iter(data_module.test_dataloader()))
    image, label = item[0].squeeze(0), item[1].squeeze(0)
    image = image.permute((1, 2, 0))
    ax.imshow(image, cmap="gray")
    ax.set_title(classes[label])

    plt.show()
