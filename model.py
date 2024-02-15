import torch
import torch.nn as nn
from dataclasses import dataclass


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, out_dim) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
            dropout=0.5,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Linear(embed_dim, out_dim)

    def forward_collect_attn(self, x):
        z = x
        x = self.norm1(x)
        attn_output, weights = self.mha(x, x, x)
        x = attn_output
        x = x + z

        z = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + z

        return x, attn_output, weights

    def forward(self, x):
        x, _, _ = self.forward_collect_attn(x)
        return x


@dataclass
class ModelConfig:
    num_classes: int
    in_channels: int
    patch_size: int
    image_size: int
    embed_dim: int
    num_heads: int
    num_layers: int


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        assert (
            config.image_size % config.patch_size == 0
        ), "image size must be a multiple of the patch size"

        self.config = config

        num_patches = (config.image_size // config.patch_size) ** 2

        self.pos_one_hot = (
            nn.functional.one_hot(torch.arange(0, num_patches + 1).to(dtype=torch.long))
            .unsqueeze(0)
            .float()
        )
        self.pos_linear = nn.Linear(num_patches + 1, config.embed_dim)

        self.class_token = torch.ones((1, 1, config.embed_dim), requires_grad=True)

        self.proj = nn.Conv2d(
            config.in_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
        )

        self.encoder = nn.Sequential(
            *[
                Encoder(config.embed_dim, config.num_heads, config.embed_dim)
                for _ in range(config.num_layers)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.num_classes),
        )

    def forward_pre_encoder(self, x):
        # move extra tensors to the corect device
        self.class_token = self.class_token.to(x)
        self.pos_one_hot = self.pos_one_hot.to(x)

        # embedding layer
        x = self.proj(x)

        # reshape
        x = x.reshape(
            (x.size()[0], self.config.embed_dim, -1)
        )  # batch, embed_dim, patch
        x = x.permute((0, 2, 1))  # batch, patch, embed_dim

        # expand the class token to the whole batch
        class_token = torch.cat(
            [self.class_token] * x.size()[0], dim=0
        )  # batch, 1, embed_dim

        # prepend the class token
        x = torch.cat((class_token, x), dim=1)

        # add the positional embeddings
        x += self.pos_linear(self.pos_one_hot)
        return x

    def forward_post_encoder(self, x):
        # take the first token of the last layer and pass it through the MLP head
        x = x[:, 0, :].squeeze(1)  # batch_size, embed_dim
        x = self.head(x)
        return x

    def forward_collect_attn(self, x):
        x = self.forward_pre_encoder(x)
        attns = []
        attns_weights = []
        for layer in self.encoder:
            x, attn, weights = layer.forward_collect_attn(x)
            attns.append(attn)
            attns_weights.append(weights)
        x = self.forward_post_encoder(x)
        return x, attns, attns_weights

    def forward(self, x):
        x = self.forward_pre_encoder(x)
        x = self.encoder(x)
        x = self.forward_post_encoder(x)
        return x


if __name__ == "__main__":
    image_size = 256
    patch_size = 32
    channels = 3
    batch_size = 10

    num_classes = 10

    input_size = (batch_size, channels, image_size, image_size)

    config = ModelConfig(
        num_classes=num_classes,
        in_channels=channels,
        patch_size=patch_size,
        image_size=image_size,
        embed_dim=512,
        num_heads=8,
        num_layers=8,
    )
    model = Model(config)

    x = torch.randn(input_size)
    print("input: ", x.size())

    out = model(x)
    print("output:", out.size())

    from torchinfo import summary

    print("model")
    summary(model, input_size=input_size, device="cpu")
