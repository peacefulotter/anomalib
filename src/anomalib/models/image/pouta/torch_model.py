"""PyTorch model for the Pouta model implementation."""

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F  # noqa: N812


def ssim(image1, image2, K=[0.01, 0.03], window_size=11):
    """
    from: https://github.com/andyj1/ssim_index/blob/master/SSIMIndex.py
    """
    _, channel1, _, _ = image1.size()
    _, channel2, _, _ = image2.size()
    channel = min(channel1, channel2)

    # gaussian window generation
    sigma = 1.5  # default
    gauss = torch.Tensor(
        [
            torch.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )

    # define constants
    # * L = 255 for constants doesn't produce meaningful results; thus L = 1
    # C1 = (K[0]*L)**2;
    # C2 = (K[1]*L)**2;
    C1 = K[0] ** 2
    C2 = K[1] ** 2

    mu1 = F.conv2d(image1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(image2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(image1 * image1, window, padding=window_size // 2, groups=channel)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(image2 * image2, window, padding=window_size // 2, groups=channel)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(image1 * image2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()


class ReconstructiveNetwork(nn.Module):
    def __init__(self):
        super(ReconstructiveNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        Fe, Fd = [], []
        for layer in self.encoder:
            x = layer(x)
            Fe.append(x)
        for layer in self.decoder:
            x = layer(x)
            Fd.append(x)
        return x, Fe, Fd


class FeatureCorrelationModule(nn.Module):
    def __init__(self, in_channels):
        super(FeatureCorrelationModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, Fe, Fd):
        Fe = self.conv1(Fe)
        Fed = torch.cat([Fe, Fd], dim=1)
        Fed = self.conv2(Fed)
        return Fed


class HierarchicalSupervisionGroup(nn.Module):
    def __init__(self, in_channels):
        super(HierarchicalSupervisionGroup, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_ws = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_cn = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, Fc, Fh):
        Fh = self.upsample(Fh)
        Fh = self.conv1(Fh)

        Wsp = self.conv_ws(Fh)  # W_sp(i)

        Wcn = self.global_avg_pool(Fh)
        Wcn = self.conv_cn(Wcn)  # W_cn(i)

        Fh = Fc * Wsp * Wcn  # F'_Ci
        return Fh


class MultiScaleSupervision(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleSupervision, self).__init__()
        self.blocks = [
            nn.Sequential(
                nn.Upsample(scale_factor=2 * i, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, 1, kernel_size=1),
            )
            for i in range(4)
        ]

    def forward(self, Fs: list[torch.Tensor]):
        f_mss = [block(f) for block, f in zip(self.blocks, Fs)]
        return f_mss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class LSSLoss(nn.Module):
    def __init__(
        self,
        alpha=1,
        gamma=2,
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.l1_loss = nn.L1Loss()

    def forward(self, predicted_anomaly_map, synthetic_anomaly_map):
        focal_loss = self.focal_loss(predicted_anomaly_map, synthetic_anomaly_map)
        l1_loss = self.l1_loss(predicted_anomaly_map, synthetic_anomaly_map)
        lss_loss = focal_loss + l1_loss
        return lss_loss


class PoutaModel(nn.Module):
    """Pouta Module.

    Args:
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.reconstructive_network = ReconstructiveNetwork()
        self.fcm1 = FeatureCorrelationModule(64)
        self.fcm2 = FeatureCorrelationModule(128)
        self.fcm3 = FeatureCorrelationModule(256)
        self.fcm4 = FeatureCorrelationModule(512)
        self.hsg1 = HierarchicalSupervisionGroup(64)
        self.hsg2 = HierarchicalSupervisionGroup(128)
        self.hsg3 = HierarchicalSupervisionGroup(256)
        self.mss = MultiScaleSupervision(64)  # Assuming final upsampling output
        self.lss_loss = LSSLoss()
        self.lambdas = torch.tensor([0.4, 0.3, 0.2, 0.1])

    def forward(
        self, input_tensor: torch.Tensor
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.

        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode.

        Args:
            input_tensor (torch.Tensor): Input tensor

        Returns:
            Tensor | dict[str, torch.Tensor]: Embedding for training, anomaly map and anomaly score for testing.
        """
        _, Fe, Fd = self.reconstructive_network(input_tensor)

        Fe1, Fe2, Fe3, Fe4 = Fe
        Fd4, Fd3, Fd2, Fd1 = Fd

        # Utilize image-level reconstruction errors
        Ir = Fd1

        # Reuse reconstructive representations
        Fc1 = self.fcm1(Fe1, Fd1)
        Fc2 = self.fcm2(Fe2, Fd2)
        Fc3 = self.fcm3(Fe3, Fd3)
        Fc4 = self.fcm4(Fe4, Fd4)

        Fh3 = self.hsg3(Fc3, Fc4)
        Fh2 = self.hsg2(Fc2, Fh3)
        Fh1 = self.hsg1(Fc1, Fh2)

        return Ir, Fh1, Fh2, Fh3, Fc4

    def shared_step(self, stage: str, batch: dict[torch.Tensor]):
        image, synthetic_anomaly_map = batch["image"], batch["anomaly_map"]
        Ir, Fh1, Fh2, Fh3, Fc4 = self(image)

        if self.training:
            refined_anomaly_maps = self.mss([Fh1, Fh2, Fh3, Fc4])

            l_rec = F.mse_loss(Ir, image) + ssim(Ir, image)
            l_pre = self.lss_loss()

            loss_lss = self.lss_loss(refined_anomaly_maps, synthetic_anomaly_map)
            l_mss = self.lambdas @ loss_lss

            loss = l_rec + l_pre + l_mss
            self.log(f"{stage}_loss", loss)

        else:
            l_mss = 0

        return loss

    def training_step(self, batch: dict[torch.Tensor], batch_idx):
        return self.shared_step("train", batch)

    def validation_step(self, batch: dict[torch.Tensor], batch_idx):
        return self.shared_step("val", batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0002)
