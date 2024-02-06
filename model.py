import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, out_dim) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        z = x
        x = self.norm1(x)
        attn_output, _ = self.mha(x, x, x)
        x = attn_output
        x = x + z

        z = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + z

        return x


class Model(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        patch_size,
        image_size,
        embed_dim,
        num_heads,
        num_layers,
        device="cpu",
    ):
        super().__init__()

        assert (
            image_size % patch_size == 0
        ), "image size must be a multiple of the patch size"

        num_patches = (image_size // patch_size) ** 2

        self.pos_one_hot = (
            nn.functional.one_hot(torch.arange(0, num_patches + 1).to(dtype=torch.long))
            .unsqueeze(0)
            .float()
            .to(device)
        )
        self.pos_linear = nn.Linear(num_patches + 1, embed_dim)

        self.class_token = torch.ones((1, 1, embed_dim), requires_grad=True).to(device)

        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        self.encoder = nn.Sequential(
            *[Encoder(embed_dim, num_heads, embed_dim) for _ in range(num_layers)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        # embedding layer
        x = self.proj(x)

        # reshape
        x = x.reshape((x.size()[0], self.embed_dim, -1))  # batch, embed_dim, patch
        x = x.permute((0, 2, 1))  # batch, patch, embed_dim

        # expand the class token to the whole batch
        class_token = torch.cat(
            [self.class_token] * x.size()[0], dim=0
        )  # batch, 1, embed_dim

        # prepend the class token
        x = torch.cat((class_token, x), dim=1)

        # add the positional embeddings
        x += self.pos_linear(self.pos_one_hot)

        # pass through the encoder
        x = self.encoder(x)

        # take the first token of the last layer and pass it through the MLP head
        x = x[:, 0, :].squeeze(1)  # batch_size, embed_dim
        x = self.head(x)
        return x


if __name__ == "__main__":
    image_size = 256
    patch_size = 32
    channels = 3
    batch_size = 10

    num_classes = 10

    input_size = (batch_size, channels, image_size, image_size)

    model = Model(num_classes, channels, patch_size, image_size, 512, 8, 8)

    x = torch.randn(input_size)
    print("input: ", x.size())

    out = model(x)
    print("output:", out.size())

    from torchinfo import summary

    print("model")
    summary(model, input_size=input_size, device="cpu")
