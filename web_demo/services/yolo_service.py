from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO

from .settings import YOLO_MODEL_PATH, DRIVE_CLASS_ID, OBS_CLASS_ID


class YOLOService:
    def __init__(self) -> None:
        self._model = None

    def _load_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(str(YOLO_MODEL_PATH))
        return self._model

    @staticmethod
    def _save_mask(mask: np.ndarray, save_path: Path) -> None:
        cv2.imwrite(str(save_path), (mask * 255).astype(np.uint8))

    def run(self, image_path: Path, run_dir: Path) -> Dict[str, Any]:
        model = self._load_model()
        results = model.predict(
            source=str(image_path),
            task="segment",
            save=False,
            verbose=False,
            conf=0.25,
            retina_masks=True,
        )

        r = results[0]
        orig_h, orig_w = r.orig_shape

        drive_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        obs_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        if r.masks is not None and r.boxes is not None:
            masks = r.masks.data.detach().cpu().numpy()
            classes = r.boxes.cls.detach().cpu().numpy().astype(int)
            for mask_i, cls_id in zip(masks, classes):
                binary_mask = (mask_i > 0.5).astype(np.uint8)
                if cls_id == DRIVE_CLASS_ID:
                    drive_mask = np.maximum(drive_mask, binary_mask)
                elif cls_id == OBS_CLASS_ID:
                    obs_mask = np.maximum(obs_mask, binary_mask)

        drive_mask_path = run_dir / "drive_mask.png"
        obs_mask_path = run_dir / "obstacle_mask.png"
        mask_overlay_path = run_dir / "mask_overlay.png"
        detect_overlay_path = run_dir / "detection_overlay.png"

        self._save_mask(drive_mask, drive_mask_path)
        self._save_mask(obs_mask, obs_mask_path)

        mask_overlay = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        mask_overlay[drive_mask == 1] = (0, 255, 0)
        mask_overlay[obs_mask == 1] = (0, 0, 255)
        cv2.imwrite(str(mask_overlay_path), mask_overlay)

        detect_overlay = r.plot()
        cv2.imwrite(str(detect_overlay_path), detect_overlay)

        objects = []
        if r.boxes is not None and len(r.boxes) > 0:
            names = r.names
            for cls_tensor, conf_tensor in zip(r.boxes.cls, r.boxes.conf):
                cls_id = int(cls_tensor.item())
                obj_name = names.get(cls_id, str(cls_id))
                objects.append(
                    {
                        "class_id": cls_id,
                        "class_name": obj_name,
                        "confidence": round(float(conf_tensor.item()), 3),
                    }
                )

        counts = {}
        for obj in objects:
            counts[obj["class_name"]] = counts.get(obj["class_name"], 0) + 1

        return {
            "drive_mask": drive_mask,
            "obs_mask": obs_mask,
            "objects": objects,
            "object_counts": counts,
            "files": {
                "drive_mask": drive_mask_path,
                "obstacle_mask": obs_mask_path,
                "mask_overlay": mask_overlay_path,
                "detection_overlay": detect_overlay_path,
            },
        }
