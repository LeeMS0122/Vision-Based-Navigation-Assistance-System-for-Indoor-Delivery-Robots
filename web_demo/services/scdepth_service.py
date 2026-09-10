from pathlib import Path
from typing import Dict, Any
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from .settings import SCDEPTH_PROJECT_ROOT, SCDEPTH_CKPT_PATH, SCDEPTH_INFER_SIZE

if str(SCDEPTH_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(SCDEPTH_PROJECT_ROOT))


class SCDepthService:
    def __init__(self) -> None:
        self._model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _build_hparams(self):
        from config import get_opts  # type: ignore

        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0]]
            hparams = get_opts()
        finally:
            sys.argv = original_argv
        hparams.ckpt_path = str(SCDEPTH_CKPT_PATH)
        return hparams

    def _load_model(self):
        if self._model is None:
            from SC_DepthV3 import SC_DepthV3  # type: ignore

            hparams = self._build_hparams()
            model = SC_DepthV3.load_from_checkpoint(str(SCDEPTH_CKPT_PATH), hparams=hparams)
            model = model.to(self.device)
            model.eval()
            self._model = model
        return self._model

    @staticmethod
    def _load_image_tensor(image_path: Path, resize=None):
        pil_img = Image.open(image_path).convert("RGB")
        if resize is not None:
            pil_img = pil_img.resize(resize)
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        return pil_img, img_tensor

    @staticmethod
    def _normalize_depth_for_vis(depth: np.ndarray) -> np.ndarray:
        p2 = np.percentile(depth, 2)
        p98 = np.percentile(depth, 98)
        if p98 - p2 < 1e-8:
            return np.zeros_like(depth, dtype=np.float32)
        norm = np.clip((depth - p2) / (p98 - p2), 0, 1)
        return norm.astype(np.float32)

    def run(self, image_path: Path, run_dir: Path) -> Dict[str, Any]:
        model = self._load_model()
        _, img_tensor = self._load_image_tensor(image_path, resize=SCDEPTH_INFER_SIZE)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            outputs = model.depth_net(img_tensor)
            if isinstance(outputs, dict):
                if ("disp", 0) in outputs:
                    pred = outputs[("disp", 0)]
                elif 0 in outputs:
                    pred = outputs[0]
                else:
                    pred = list(outputs.values())[0]
            else:
                pred = outputs

        depth_map = pred.squeeze().detach().cpu().numpy().astype(np.float32)
        depth_norm = self._normalize_depth_for_vis(depth_map)

        depth_gray_path = run_dir / "depth_norm.png"
        depth_vis_path = run_dir / "depth_visualization.png"

        gray_img = np.uint8(depth_norm * 255)
        cv2.imwrite(str(depth_gray_path), gray_img)

        depth_color = cv2.applyColorMap(gray_img, cv2.COLORMAP_PLASMA)
        cv2.imwrite(str(depth_vis_path), depth_color)

        return {
            "depth_map": depth_map,
            "depth_norm": depth_norm,
            "files": {
                "depth_norm": depth_gray_path,
                "depth_visualization": depth_vis_path,
            },
        }
