# Ultralytics YOLO 🚀, AGPL-3.0 license
import os
import cv2
import re
import numpy as np
import torch
from pathlib import Path
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops, SKYSEG_CFG


class SkySegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model="yolov8n-seg.pt", source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "skyseg"

    def preprocess(self, im):
        assert len(im) == 1, "The sky segmentation predictor just can process one image for one call."
        im = im[0]
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.args.imgsz
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)),int(round(dh - 0.1))
        left, right = int(round(dw - 0.1)),int(round(dw - 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

        self.borders = [top, left, bottom, right]
        im = super().preprocess([im])
        return im

    def postprocess(self, preds, img, orig_imgs):
        """Applies non-max suppression and processes detections for each image in an input batch."""
        p = ops.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        for i, (pred, orig_img, img_path) in enumerate(zip(p, orig_imgs, self.batch[0])):
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results

    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str:
        """Write inference results to a file or directory.

        Args:
            i (int): Index of the current image in the batch.
            p (Path): Path to the current image.
            im (torch.Tensor): Preprocessed image tensor.
            s (List[str]): List of result strings.

        Returns:
            (str): String with result information.
        """
        string = ""  # print string
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 0 if frame undetermined

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # used in other locations
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # Add predictions to image
        imagename = result.path.split(os.sep)[-1]
        if self.args.save or self.args.show:
            image_dir, mask_dir = os.path.join(self.save_dir, "image"), os.path.join(self.save_dir, "mask")
            os.mkdir(image_dir) if not os.path.exists(image_dir) else None
            os.mkdir(mask_dir) if not os.path.exists(mask_dir) else None
            if result.masks is None:
                print(imagename + ":error")
                return string
            self.plot_predict_samples(
                result.orig_img,
                result.masks[0].data[0],
                nc=1,
                colors=(255,255,255),
                fname=self.save_dir / "image" / imagename,
                mname=self.save_dir / "mask" / imagename,
                one_hot=False,
                overlap=False,
            )

        # Save results
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))

        return string


    def plot_predict_samples(self, image, masks, nc, colors, fname, mname, one_hot=True, overlap=False):
        """Plot and visualize the predicting results.

        Args:
            image(torch.Tensor| numpy.ndarray): input image
            masks: (torch.Tensor| numpy.ndarray): predict mask
            nc(int): number of categories
            colors(List): colors for each categories
            fname(str): saved image path
            mname(str): save mask path
            one_hot(bool): is the format of mask one-hot
            overlap(bool): plot mask on image.

        Returns:
            None
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().float().numpy()

        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        if np.max(image) <= 1:
            image *= 255  # de-normalise (optional)

        if np.max(masks) <= 1:
            masks = (masks * 255).astype(np.uint8)

        h, w, _ = image.shape  # batch size, _, height, width
        hm, wm = masks.shape
        mask_bgr = np.ones((hm, wm, 3), dtype=np.uint8) * 0
        if one_hot:
            mask = masks.argmax(axis=0).astype(np.uint8)
        else:
            mask = masks


        mask_bgr[mask >= 125] = (255,255,255)
        msk = cv2.resize(mask_bgr, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

        if overlap:
            img = cv2.addWeighted(image, 0.7, msk, 0.3, 0)
            cv2.imwrite(fname, img)
        else:
            cv2.imwrite(fname, image)
            cv2.imwrite(mname, msk)


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Train a YOLO segmentation model based on passed arguments."""
    model = cfg.model or 'yolov11n-seg.pt'
    source = cfg.source   # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''
    cfg.name = os.path.join(cfg.name, 'predict')
    args = dict(model=model, source=source, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).predict(**args)
    else:
        predictor = SkySegmentationPredictor(cfg=cfg, overrides=args)
        predictor(source=source, model=model)


if __name__ == '__main__':
    predict(cfg=SKYSEG_CFG)
