"""This notebooks trains a cycle gan network from CelebA and DiscoElysium character portraits
"""
__author__ = "https://github.com/vahvero"

# %% Imports

import logging
import os
import random

import PIL
import torch
from torchvision.transforms import Resize
from matplotlib import pyplot as plt
from torch import Tensor, nn, FloatTensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image, to_tensor, resize
from torch.utils.data import DataLoader
from torch.optim import Adam

# %% Set constants
verbose = False

log_folder = "logs"
intermidiate_results_folder = os.path.join(log_folder, "epochs_results")
intermidiate_models_folder = os.path.join(log_folder, "models")
os.makedirs(intermidiate_results_folder, exist_ok=True)
os.makedirs(intermidiate_models_folder, exist_ok=True)

# These should already exits
disco_root = "assets/disco"
celeb_root = "assets/celeba"
test_root = "assets/my_images"
image_resize_size = (256, 256)
batch_size = 2
learning_rate = 2e-4
max_epochs = int(1000)
device = torch.device("cuda:0")

# %% Set up logging
if verbose:
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                filename=os.path.join(log_folder, "train.log"),
                mode="w+",
            ),
            logging.StreamHandler(),
        ],
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                filename=os.path.join(log_folder, "train.log"),
                mode="w+",
            )
        ],
    )

logger = logging.getLogger(__name__)

ImageTensor = FloatTensor


# %% Initiliaze datasets
logger.info("Initializing datasets")


def read_image(filepath: str) -> Tensor:

    with PIL.Image.open(filepath) as fobj:
        return to_tensor(fobj.convert("RGB"))


class FolderDataset(Dataset):
    def __init__(self, root: str):
        self.root: str = root
        self.images: list[str] = [
            os.path.join(self.root, path) for path in os.listdir(self.root)
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> ImageTensor:
        logger.debug("Reading %s", self.images[index])
        return read_image(self.images[index])

    def shuffle(self) -> None:
        """Shuffle underlying data order"""
        random.shuffle(self.images)


disco_dataset = FolderDataset(
    root=disco_root,
)

celeba_dataset = FolderDataset(
    root=celeb_root,
)


# %% Visualize

if verbose:

    fig, axes = plt.subplots(nrows=3, ncols=2)

    for index, (ax0, ax1) in enumerate(axes):

        ax0.imshow(to_pil_image(disco_dataset[index]))
        ax0.set_title(f"Disco[{index}]")
        ax1.imshow(to_pil_image(celeba_dataset[index]))
        ax1.set_title(f"CelebA[{index}]")

    fig.suptitle("Datasets' images")
    fig.tight_layout()

# %% Training dataset


class DualDataset(Dataset):
    def __init__(self, base: FolderDataset, style: FolderDataset, size=(256, 256)):
        self.base = base
        self.style = style

        self.transform = Resize(size)

    def __len__(self) -> int:
        return min(
            len(self.base),
            len(self.style),
        )

    def __getitem__(self, idx: int) -> tuple[ImageTensor, ImageTensor]:
        return (
            self.transform(self.base[idx]),
            self.transform(self.style[idx]),
        )


train_dataset = DualDataset(celeba_dataset, disco_dataset, size=image_resize_size)

assert train_dataset[0][0].shape == (3, *image_resize_size)
assert train_dataset[0][1].shape == (3, *image_resize_size)


if verbose:
    fig, axes = plt.subplots(nrows=3, ncols=2)
    for index, ((ax0, ax1), (base_image, style_image)) in enumerate(
        zip(axes, train_dataset)
    ):
        ax0.imshow(to_pil_image(base_image))
        ax0.set_title(f"Base {index}")

        ax1.imshow(to_pil_image(style_image))
        ax1.set_title(f"Style {index}")
    fig.suptitle("Dual dataset showcase")
    fig.tight_layout()

# %% Define model


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        dropout=True,
    ):
        super(Upsample, self).__init__()
        self.dropout = dropout
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=nn.InstanceNorm2d,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout_layer = nn.Dropout2d(0.5)

    def forward(self, x, shortcut=None):
        x = self.block(x)
        if self.dropout:
            x = self.dropout_layer(x)

        if shortcut is not None:
            x = torch.cat([x, shortcut], dim=1)

        return x


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        apply_instancenorm=True,
    ):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=nn.InstanceNorm2d,
        )
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.apply_norm = apply_instancenorm

    def forward(self, x):
        x = self.conv(x)
        if self.apply_norm:
            x = self.norm(x)
        x = self.relu(x)

        return x


class UNETGenerator(nn.Module):
    def __init__(self, filter_size=64):
        super(UNETGenerator, self).__init__()
        self.downsamples = nn.ModuleList(
            [
                Downsample(
                    3, filter_size, kernel_size=4, apply_instancenorm=False
                ),  # (b, filter, 128, 128)
                Downsample(filter_size, filter_size * 2),  # (b, filter * 2, 64, 64)
                Downsample(filter_size * 2, filter_size * 4),  # (b, filter * 4, 32, 32)
                Downsample(filter_size * 4, filter_size * 8),  # (b, filter * 8, 16, 16)
                Downsample(filter_size * 8, filter_size * 8),  # (b, filter * 8, 8, 8)
                Downsample(filter_size * 8, filter_size * 8),  # (b, filter * 8, 4, 4)
                Downsample(filter_size * 8, filter_size * 8),  # (b, filter * 8, 2, 2)
            ]
        )

        self.upsamples = nn.ModuleList(
            [
                Upsample(filter_size * 8, filter_size * 8),
                Upsample(filter_size * 16, filter_size * 8),
                Upsample(filter_size * 16, filter_size * 8),
                Upsample(filter_size * 16, filter_size * 4, dropout=False),
                Upsample(filter_size * 8, filter_size * 2, dropout=False),
                Upsample(filter_size * 4, filter_size, dropout=False),
            ]
        )

        self.last = nn.Sequential(
            nn.ConvTranspose2d(filter_size * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        skips = []
        for l in self.downsamples:
            x = l(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        for l, s in zip(self.upsamples, skips):
            x = l(x, s)

        out = self.last(x)

        return out


class UNETDiscriminator(nn.Module):
    def __init__(self, filter_size=64):
        super(UNETDiscriminator, self).__init__()

        self.block = nn.Sequential(
            Downsample(
                3, filter_size, kernel_size=4, stride=2, apply_instancenorm=False
            ),
            Downsample(filter_size, filter_size * 2, kernel_size=4, stride=2),
            Downsample(filter_size * 2, filter_size * 4, kernel_size=4, stride=2),
            Downsample(filter_size * 4, filter_size * 8, kernel_size=4, stride=1),
        )

        self.last = nn.Conv2d(filter_size * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.block(x)
        x = self.last(x)

        return x


# %% Create dataloader

dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=min(torch.multiprocessing.cpu_count(), batch_size),
)
# Test dataloader
for base, style in dataloader:
    break

if verbose:
    fig, axes = plt.subplots(nrows=len(base), ncols=2)

    for idx, (base_image, style_image) in enumerate(zip(base, style, strict=True)):
        axes[idx, 0].imshow(to_pil_image(disco_dataset[index]))
        axes[idx, 0].set_title(f"Disco[{index}]")
        axes[idx, 1].imshow(to_pil_image(celeba_dataset[index]))
        axes[idx, 1].set_title(f"CelebA[{index}]")

    fig.suptitle("Dataloader images")
    fig.tight_layout()

# %% Test constructors
net = UNETGenerator()
out = net(base)

net = UNETDiscriminator()
out = net(base)


# %% Setup optimizers

gen_base = UNETGenerator()
gen_style = UNETGenerator()
dis_base = UNETDiscriminator()
dis_style = UNETDiscriminator()


gen_base_optimizer = Adam(
    gen_base.parameters(),
    lr=learning_rate,
    betas=(0.5, 0.999),
)
gen_style_optimizer = Adam(
    gen_style.parameters(),
    lr=learning_rate,
    betas=(0.5, 0.999),
)
dis_base_optimizer = Adam(
    dis_base.parameters(),
    lr=learning_rate,
    betas=(0.5, 0.999),
)
dis_style_optimizer = Adam(
    dis_style.parameters(),
    lr=learning_rate,
    betas=(0.5, 0.999),
)

rec_loss_fn = nn.L1Loss()
gen_loss_fn = nn.MSELoss()
dis_loss_fn = nn.MSELoss()


# %% Start training


gen_base.to(device)
gen_style.to(device)
dis_base.to(device)
dis_style.to(device)

gan_train_loss = []
dis_valid_train_loss = []
dis_fake_train_loss = []

for epoch in range(1, max_epochs + 1):

    gan_total_epoch_loss = []
    dis_valid_total_epoch_loss = []
    dis_fake_total_epoch_loss = []
    for base, style in dataloader:
        gen_base_optimizer.zero_grad()
        gen_style_optimizer.zero_grad()
        dis_base_optimizer.zero_grad()
        dis_style_optimizer.zero_grad()

        base = base.to(device)
        style = style.to(device)

        # Generators
        valid = torch.ones(batch_size, 1, 30, 30).to(device)

        # Val loss
        vbase_loss = gen_loss_fn(dis_base(gen_base(style)), valid)
        vstyle_loss = gen_loss_fn(dis_style(gen_style(base)), valid)
        val_loss = (vbase_loss + vstyle_loss) / 2

        # Rec loss
        rbase_loss = rec_loss_fn(
            gen_base(gen_style(base)),
            base,
        )
        rstyle_loss = rec_loss_fn(
            gen_style(gen_base(style)),
            style,
        )

        rec_loss = (rbase_loss + rstyle_loss) / 2

        # Identity
        idbase_loss = rec_loss_fn(
            gen_style(base),
            base,
        )
        idstyle_loss = rec_loss_fn(
            gen_base(style),
            style,
        )

        id_loss = (idbase_loss + idstyle_loss) / 2

        # Run backwards on GAN losses

        total_gan_loss = rec_loss + 5 * val_loss + 10 * id_loss
        total_gan_loss.backward()

        # Discriminators
        fake = torch.zeros(batch_size, 1, 30, 30).to(device)

        # Use generated fake images to calculate loss
        dis_fake_base_loss = dis_loss_fn(dis_style(gen_style(base.detach())), fake)
        dis_fake_style_loss = dis_loss_fn(dis_base(gen_base(style.detach())), fake)

        dis_fake_loss = (dis_fake_base_loss + dis_fake_style_loss) / 2

        # Use true images to calculate loss
        dis_valid_style_loss = dis_loss_fn(dis_style(style.detach()), valid)
        dis_valid_base_loss = dis_loss_fn(dis_base(base.detach()), valid)

        dis_valid_loss = (dis_valid_style_loss + dis_valid_base_loss) / 2

        # Run backward on discriminator loss
        dis_valid_loss.backward()
        dis_fake_loss.backward()

        gan_total_epoch_loss.append(total_gan_loss.item())
        dis_valid_total_epoch_loss.append(dis_valid_loss.item())
        dis_fake_total_epoch_loss.append(dis_fake_loss.item())

        gen_base_optimizer.step()
        gen_style_optimizer.step()
        dis_base_optimizer.step()
        dis_style_optimizer.step()

    gan_train_loss.append(sum(gan_total_epoch_loss))
    dis_valid_train_loss.append(sum(dis_valid_total_epoch_loss))
    dis_fake_train_loss.append(sum(dis_fake_total_epoch_loss))

    logger.info("GAN loss %4f", gan_train_loss[-1])
    logger.info("Dicriminator valid loss %4f", dis_valid_train_loss[-1])
    logger.info("Discriminator fake loss %4f", dis_fake_train_loss[-1])

    logger.info("%d / %d epoch done", epoch, max_epochs)

    torch.save(
        gen_style.state_dict(),
        os.path.join(intermidiate_models_folder, f"{epoch}_gen_style.pth"),
    )
    torch.save(
        gen_base.state_dict(),
        os.path.join(intermidiate_models_folder, f"{epoch}_gen_base.pth"),
    )
    torch.save(
        dis_base.state_dict(),
        os.path.join(intermidiate_models_folder, f"{epoch}_dis_base.pth"),
    )
    torch.save(
        dis_style.state_dict(),
        os.path.join(intermidiate_models_folder, f"{epoch}_dis_style.pth"),
    )

    gen_style.eval()
    gen_base.eval()
    with torch.no_grad():

        gen_base_img = gen_style(base)[0]
        gen_style_img = gen_base(style)[0]
        fig, axes = plt.subplots(nrows=2, ncols=2)

        axes[0, 0].imshow(to_pil_image(gen_base_img))
        axes[0, 1].imshow(to_pil_image(base[0]))
        axes[0, 1].set_title("Base original")
        axes[0, 0].set_title("Base -> Style")
        axes[1, 0].imshow(to_pil_image(gen_style_img))
        axes[1, 1].imshow(to_pil_image(style[0]))
        axes[1, 1].set_title("Style original")
        axes[1, 0].set_title("Style -> Base")
        fig.tight_layout()
        fig.savefig(
            os.path.join(intermidiate_results_folder, f"{epoch}_test_gallery.png")
        )
        plt.close(fig)

    gen_style.train()
    gen_base.train()

# %% Show losss

plt.figure()
plt.plot(gan_train_loss, label="GAN loss")
plt.plot(dis_valid_train_loss, label="Discriminator valid")
plt.plot(dis_fake_train_loss, label="Disciminator fake")
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Loss by epoch")
plt.legend()
plt.savefig(os.path.join(log_folder, "train_loss_graph.png"))
if verbose:
    plt.show()


# %% Test result

gen_style.eval()
gen_base.eval()
with torch.no_grad():

    gen_base_img = gen_style(base)[0]
    gen_style_img = gen_base(style)[0]
    fig, axes = plt.subplots(nrows=2, ncols=2)

    axes[0, 0].imshow(to_pil_image(gen_base_img))
    axes[0, 1].imshow(to_pil_image(base[0]))
    axes[0, 1].set_title("Base original")
    axes[0, 0].set_title("Base -> Style")
    axes[1, 0].imshow(to_pil_image(gen_style_img))
    axes[1, 1].imshow(to_pil_image(style[0]))
    axes[1, 1].set_title("Style original")
    axes[1, 0].set_title("Style -> Base")
    fig.tight_layout()
    fig.savefig(os.path.join(log_folder, "test_gallery.png"))
    if verbose:
        plt.show()


# %% Test with real images

img_filenames = os.listdir(
    test_root,
)

if len(img_filenames) > 0:
    with torch.no_grad():
        gen_style.eval()

        fig, axes = plt.subplots(nrows=len(img_filenames), ncols=2, figsize=(16, 16))
        for idx, filename in enumerate(img_filenames):

            with PIL.Image.open(os.path.join(test_root, filename)) as pil:
                img_tensor = to_tensor(pil.convert("RGB")).to(device)
                img_tensor = resize(img_tensor, image_resize_size)
                generated = gen_style(img_tensor.unsqueeze(0))

                axes[idx, 0].imshow(to_pil_image(generated[0]))
                axes[idx, 1].imshow(to_pil_image(img_tensor))
        fig.suptitle("Your images gallery")
        fig.savefig("generated_images_gallery.png")
    if verbose:
        fig.show()

# %% Script ended
