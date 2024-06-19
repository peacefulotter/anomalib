"""PyTorch model for the PatchCore model implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from torch_ema import ExponentialMovingAverage


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
    def __init__(
        self,
    ):
        pass

    def forward(self, x):
        pass


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
    ) -> None:
        super().__init__()
        self.patchcore = ReconPatchcoreModel(
            layers=layers,
            backbone=backbone,
            pre_trained=pre_trained,
            num_neighbors=num_neighbors,
        )
        # TODO: backbone.in_dim
        # TODO: projection dim?
        self.representation = nn.Linear()
        self.projection = nn.Linear()

        self.ema_representation = nn.Linear()
        self.ema_projection = nn.Linear()

        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.995)

        # TODO: freeze ema

    def forward(
        self, input_tensor: torch.Tensor
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        embedding, emb_shape = self.patchcore.get_embedding(input_tensor)
        embedding = self.representation(embedding)
        # ema_embeddi

        if self.training:
            embedding = self.projection(embedding)
            # self.ema.update()

        if not self.training:
            output_size = input_tensor.shape[-2:]
            output = self.patchcore.compute_patch_scores(
                embedding, emb_shape, output_size
            )

        return output
