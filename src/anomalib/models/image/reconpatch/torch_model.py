"""PyTorch model for the Reconpatch model implementation."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn
from enum import Enum, auto
from ema_pytorch import EMA
from torch.nn import functional as F  # noqa: N812

from anomalib.models.image.patchcore.torch_model import PatchcoreModel


class ReconpatchMode(Enum):
    TRAINING = auto()
    COLLECTING = auto()


class ReconPatchcoreModel(PatchcoreModel):
    def get_embedding(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {
            layer: self.feature_pooler(feature) for layer, feature in features.items()
        }
        embedding = self.generate_embedding(features)

        if self.tiler:
            embedding = self.tiler.untile(embedding)

        emb_shape = embedding.shape
        embedding = self.reshape_embedding(embedding)

        return embedding, emb_shape

    def compute_patch_scores(
        self,
        embedding: torch.Tensor,
        emb_shape: torch.Size,
        output_size: torch.Size,
    ) -> dict[torch.Tensor]:
        batch_size, _, width, height = emb_shape
        # apply nearest neighbor search
        patch_scores, locations = self.nearest_neighbors(
            embedding=embedding, n_neighbors=1
        )
        # reshape to batch dimension
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))
        # compute anomaly score
        pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
        # reshape to w, h
        patch_scores = patch_scores.reshape((batch_size, 1, width, height))
        # get anomaly map
        anomaly_map = self.anomaly_map_generator(patch_scores, output_size)

        output = {"anomaly_map": anomaly_map, "pred_score": pred_score}
        return output

    def forward(
        self, input_tensor: torch.Tensor
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        raise


class ContextualSimilarity(nn.Module):
    def __init__(self, k=5):
        self.k = k

    def forward(self, z: torch.Tensor):

        distances = torch.cdist(z, z)
        print(distances.shape)
        kth_nearst = -torch.topk(-distances, k=self.k, sorted=True).values[:, -1]
        print(kth_nearst.shape)
        mask = (distances <= kth_nearst[:None]).float()
        print(mask.shape)

        similarity = (mask @ mask.T) / torch.sum(mask, dim=-1, keepdims=True)
        print(similarity.shape)
        R = mask * mask.T
        print(R.shape)
        similarity = (similarity @ R.T) / torch.sum(R, dim=-1, keepdims=True)
        print(similarity.shape)
        return 0.5 * (similarity + similarity.T)


class PairwiseSimilarity(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, z):
        return torch.exp(-torch.cdist(z, z) / self.sigma)


class ContrastiveLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.pairwise_similarity = PairwiseSimilarity()
        self.contextual_similarity = ContextualSimilarity()
        self.alpha = alpha

    def forward(self, z: torch.Tensor, z_ema: torch.Tensor):
        p_sim = self.pairwise_similarity(z_ema)
        c_sim = self.contextual_similarity(z_ema)
        w = self.alpha * p_sim + (1 - self.alpha) * c_sim

        distances = torch.sqrt(torch.cdist(z, z) + 1e-9)
        delta = distances / torch.mean(distances, dim=-1, keepdims=True)

        rc_loss = w * (delta**2) + (1 - w) * (F.relu(self.margin - delta) ** 2)
        rc_loss = torch.mean(rc_loss, dim=-1).sum()
        return rc_loss


class ReconpatchModel(nn.Module):
    """ReConPatch Module.

    Args:
        layers (list[str]): Layers used for feature extraction
        backbone (str, optional): Pre-trained model backbone.
            Defaults to ``resnet18``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
        num_neighbors (int, optional): Number of nearest neighbors.
            Defaults to ``9``.
    """

    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
        coreset_sampling_ratio: float = 0.1,
        representation_dim: int = 2048,
        projection_dim: int = 128,
        beta=0.995,
    ) -> None:
        super().__init__()
        self.patchcore = ReconPatchcoreModel(
            layers=layers,
            backbone=backbone,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
        )

        in_dim = self.patchcore.feature_extractor.out_dims[-1]

        self.representation = nn.Linear(in_dim, representation_dim)
        self.projection = nn.Linear(representation_dim, projection_dim)

        self.ema_representation = EMA(
            self.representation,
            beta=beta,
            update_after_step=0,
            update_every=1,
        )
        self.ema_projection = EMA(
            self.representation,
            beta=beta,
            update_after_step=0,
            update_every=1,
        )

        self.contrastive_loss = ContrastiveLoss()

        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.embeddings = []

    def forward(
        self, input_tensor: torch.Tensor, mode: ReconpatchMode
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        embedding, emb_shape = self.patchcore.get_embedding(input_tensor)

        r = self.representation(embedding)

        if self.training and mode == ReconpatchMode.TRAINING:
            z = self.projection(r)

            with torch.no_grad():
                r_ema = self.ema_representation(embedding)
                z_ema = self.ema_projection(r_ema)

            self.ema_representation.update()
            self.ema_projection.update()

            loss = self.contrastive_loss(z, z_ema)
            return loss

        elif self.training and mode == ReconpatchMode.COLLECTING:
            self.embeddings.append(r)

        else:
            output_size = input_tensor.shape[-2:]
            output = self.patchcore.compute_patch_scores(
                embedding, emb_shape, output_size
            )
            return output

    def fit(self) -> None:
        """Apply subsampling to the embedding collected from the training set."""
        # logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        # logger.info("Applying core-set subsampling to get the embedding.")
        self.patchcore.subsample_embedding(embeddings, self.coreset_sampling_ratio)
