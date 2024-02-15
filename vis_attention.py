import torch

import matplotlib.pyplot as plt

from data import FashionMnistDataModule


@torch.no_grad
def main():
    batch_size = 8

    if torch.cuda.is_available():
        model = torch.load("./out/model")
    else:
        model = torch.load("./out/model", map_location=torch.device("cpu"))

    image_size = model.config.image_size
    patch_size = model.config.patch_size

    data = FashionMnistDataModule(
        resize=image_size,
        val_fraction=0.1,
        batch_size=batch_size,
        num_workers=2,
    )

    # exmaple shapes:
    #  img:   torch.Size([4, 1, 224, 224])
    #  label: torch.Size([4])
    img, label = next(iter(data.test_dataloader()))

    # note:
    # in the following example this case the image size was 224 and the
    # patch size was 32 therefore 224 / 32 = 7, meaning we hasve 7^2=49 image
    # patches plus one patch for the class token
    # example shapes:
    #   out:     torch.Size([4, 10])
    #   attn:    list of torch.Size([4, 50, 512])
    #   weight:  list of torch.Size([4, 50, 50])
    out, attn, weight = model.forward_collect_attn(img)

    # remove the class token
    attn = [x[:, 1:, :] for x in attn]
    weight = [x[:, :, 1:] for x in weight]

    num_layers = len(attn)
    nrows = len(attn) + 2
    ncols = attn[0].size()[0]  # batch size

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

    # plot image
    for image_idx in range(ncols):
        layer = 0
        axs[layer][image_idx].imshow(img[image_idx].permute(1, 2, 0))

    # plot attention maps
    for layer in range(num_layers):
        for image_idx in range(ncols):
            axs[layer + 1][image_idx].imshow(
                weight[layer][image_idx][0][:]
                .reshape(image_size // patch_size, image_size // patch_size)
                .unsqueeze(-1)
            )

    upsampel = torch.nn.Upsample(scale_factor=patch_size, mode="bilinear")

    # plot attention maps on top of the images
    for image_idx in range(ncols):
        layer = num_layers - 1

        img_permuted = img[image_idx].permute(1, 2, 0)  # [224, 224, 1]

        attn_map = weight[layer - 1][image_idx][0][:].reshape(
            1,
            1,
            image_size // patch_size,
            image_size // patch_size,
        )
        attn_map = upsampel(attn_map)
        attn_map = attn_map.reshape(image_size, image_size, 1)

        axs[nrows - 1][image_idx].imshow(img_permuted * attn_map)

    plt.show()


if __name__ == "__main__":
    main()
