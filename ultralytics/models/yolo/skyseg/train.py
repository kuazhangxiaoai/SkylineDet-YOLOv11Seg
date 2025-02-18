# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy
import os

from ultralytics.models import yolo
from ultralytics.nn.tasks import SkySegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK, SKYSEG_CFG
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.data import build_sky_dataset

class SkySegmentationTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationTrainer

        args = dict(model="yolov8n-seg.pt", data="coco8-seg.yaml", epochs=3)
        trainer = SegmentationTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "skyseg"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return SegmentationModel initialized with specified config and weights."""
        model = SkySegmentationModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """
            Build YOLO Dataset.

            Args:
                img_path (str): Path to the folder containing images.
                mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
                batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_sky_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return yolo.skyseg.SkySegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with labels and box coordinates."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png

def train(cfg=DEFAULT_CFG, use_python=False):
    """Train a YOLO segmentation model based on passed arguments."""
    model = cfg.model or 'yolov11n-seg.pt'
    data = cfg.data or 'coco128-seg.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''
    cfg.name = os.path.join(cfg.name, 'train')

    args = dict(model=model, data=data, device=device, task="skyseg")
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = SkySegmentationTrainer(cfg=cfg, overrides=args)
        trainer.train()


if __name__ == '__main__':
    train(cfg=SKYSEG_CFG)