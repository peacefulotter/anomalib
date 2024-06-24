"""ReConPatch : Contrastive Patch Representation Learning for Industrial Anomaly Detection.

Paper https://arxiv.org/abs/2305.16713.
"""

import torch
import logging

from adamp import AdamP
from collections.abc import Sequence
from torch.optim.optimizer import Optimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize, Transform

from anomalib import LearningType
from anomalib.models.components import AnomalyModule

from .torch_model import ReconpatchMode, ReconpatchModel


logger = logging.getLogger(__name__)


class Reconpatch(AnomalyModule):
    """ReconpatchLightning Module to train ReCOnPatch algorithm.

    >>> model = Reconpatch()
    >>> trainer.fit(model, datamodule) # Training for N epochs
    >>> model.model.mode = ReconpatchMode.COLLECTING
    >>> trainer.fit(model, datamodule) # Collecting embeddings on 1 epoch
    >>> model.fit() # Creating memory bank

    Args:
        backbone (str): Backbone CNN network
            Defaults to ``wide_resnet50_2``.
        layers (list[str]): Layers to extract features from the backbone CNN
            Defaults to ``["layer2", "layer3"]``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
        num_neighbors (int, optional): Number of nearest neighbors.
            Defaults to ``9``.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
        num_neighbors: int = 9,
        lr=0.05,
    ) -> None:
        super().__init__()

        self.model: ReconpatchModel = ReconpatchModel(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layers,
            num_neighbors=num_neighbors,
        )
        self.lr = lr
        self.mode: ReconpatchMode = ReconpatchMode.TRAINING

    def configure_optimizers(self) -> Optimizer:
        optimizer = AdamP(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(
        self, batch: dict[str, str | torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        """Generate feature embedding of the batch.

        Args:
            batch (dict[str, str | torch.Tensor]): Batch containing image filename, image, label and mask
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            dict[str, np.ndarray]: Embedding Vector
        """
        del args, kwargs  # These variables are not used.

        loss = self.model(batch["image"], self.mode)
        return loss

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
