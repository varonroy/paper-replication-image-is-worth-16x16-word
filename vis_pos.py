import torch

import matplotlib.pyplot as plt


if __name__ == "__main__":
    input_size = (224, 224)
    patch_size = 32
    embed_dim = 512

    nrows = 10
    ncols = 10

    model = torch.load("./out/model")
    proj = model.proj

    kernels = proj.weight.data  # [embed_dim, 1, patch_size, patch_size]
    kernels = kernels.squeeze(1)  # [embed_dim, patch_size, patch_size]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

    def plot_kernel(idx):
        i = idx // nrows
        j = idx - i * nrows

        kernel = kernels[idx]
        axs[i][j].imshow(kernel)

    for i in range(nrows * ncols):
        plot_kernel(i)

    plt.show()
