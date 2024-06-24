"""Towards Total Recall in Industrial Anomaly Detection.

Paper https://arxiv.org/abs/2106.08265.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize, Transform

from anomalib import LearningType
from anomalib.models.components import AnomalyModule

from .torch_model import PoutaModel

logger = logging.getLogger(__name__)


class Pouta(AnomalyModule):
    """PoutaLightning Module to train PatchCore algorithm.

    Args:

    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def configure_optimizers(self) -> None:
        return None

    def training_step(
        self, batch: dict[str, str | torch.Tensor], *args, **kwargs
    ) -> None:
        """
        Args:
            batch (dict[str, str | torch.Tensor]): Batch containing image filename, image, label and mask
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            dict[str, np.ndarray]: Embedding Vector
        """
        del args, kwargs  # These variables are not used.

        pass

    def validation_step(
        self, batch: dict[str, str | torch.Tensor], *args, **kwargs
    ) -> STEP_OUTPUT:
        """Get batch of anomaly maps from input image batch.

        Args:
            batch (dict[str, str | torch.Tensor]): Batch containing image filename, image, label and mask
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: Image filenames, test images, GT and predicted label/masks
        """
        # These variables are not used.
        del args, kwargs

        # Get anomaly maps and predicted scores from the model.
        output = self.model(batch["image"])

        # Add anomaly maps and predicted scores to the batch.
        batch["anomaly_maps"] = output["anomaly_map"]
        batch["pred_scores"] = output["pred_score"]

        return batch

    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        """Return Pouta trainer arguments."""
        return None

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    def configure_transforms(
        self, image_size: tuple[int, int] | None = None
    ) -> Transform:
        """Default transform for Padim."""
        image_size = image_size or (256, 256)
        # scale center crop size proportional to image size
        height, width = image_size
        center_crop_size = (int(height * (224 / 256)), int(width * (224 / 256)))
        return Compose(
            [
                Resize(image_size, antialias=True),
                CenterCrop(center_crop_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )

    # TODO: REQUIRES CENTER CROP?
    # def configure_transforms(
    #     self, image_size: tuple[int, int] | None = None
    # ) -> Transform:
    #     """Default transform for Padim."""
    #     image_size = image_size or (256, 256)
    #     return Compose(
    #         [
    #             Resize(image_size, antialias=True),
    #             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ],
    #     )
