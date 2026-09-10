import json
from pathlib import Path
from typing import Any, Dict, List

import cv2

from .pipeline_service import pipeline_service
from .status_service import status_service
from .settings import VIDEO_RUNS_BASE_DIR
from web_demo.utils.video_io import extract_sampled_frames, compose_video_from_frames


class VideoService:
    @staticmethod
    def _to_outputs_web_path(path: Path) -> str:
        rel = path.relative_to(VIDEO_RUNS_BASE_DIR.parent)
        return f"/demo-outputs/{rel.as_posix()}"

    @staticmethod
    def _annotate_frame(base_image_path: Path, status: Dict[str, Any], frame_index: int, save_path: Path) -> None:
        img = cv2.imread(str(base_image_path))
        if img is None:
            raise RuntimeError(f"Failed to read frame image: {base_image_path}")

        lines = [
            f"Frame: {frame_index}",
            f"Risk: {status['current_risk']}",
            f"Direction: {status['recommended_direction']}",
            f"Replanning: {'YES' if status['path_recalculation_required'] else 'NO'}",
        ]

        y = 30
        for ln in lines:
            cv2.putText(img, ln, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 3, cv2.LINE_AA)
            cv2.putText(img, ln, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
            y += 30

        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)

    @staticmethod
    def _build_summary(frame_results: List[Dict[str, Any]], video_info: Dict[str, Any]) -> Dict[str, Any]:
        risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        replans = 0
        path_found = 0

        for fr in frame_results:
            risk = fr["status"]["current_risk"]
            if risk in risk_counts:
                risk_counts[risk] += 1

            if fr["status"]["path_recalculation_required"]:
                replans += 1
            if fr["status"]["path_found"]:
                path_found += 1

        total = len(frame_results)

        return {
            "total_processed_frames": total,
            "path_found_frames": path_found,
            "path_found_ratio": round(path_found / total, 4) if total > 0 else 0.0,
            "risk_counts": risk_counts,
            "replanning_required_frames": replans,
            "video_info": video_info,
        }

    def run(self, input_video_path: Path, run_dir: Path, sample_fps: float = 5.0) -> Dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)

        sampled_frames_dir = run_dir / "sampled_frames"
        frame_runs_dir = run_dir / "frame_runs"
        overlay_frames_dir = run_dir / "overlay_frames"

        sampled_frames, video_info = extract_sampled_frames(
            video_path=input_video_path,
            frames_dir=sampled_frames_dir,
            sample_fps=sample_fps,
        )

        if not sampled_frames:
            raise RuntimeError("No sampled frames extracted from the uploaded video")

        frame_results = []
        overlay_paths = []
        prev_status_for_replan = None

        for frame_meta in sampled_frames:
            idx = int(frame_meta["sampled_index"])
            frame_path: Path = frame_meta["frame_path"]
            frame_run_dir = frame_runs_dir / f"frame_{idx:06d}"
            frame_run_dir.mkdir(parents=True, exist_ok=True)

            frame_pipeline = pipeline_service.run_for_video_frame(frame_path=frame_path, frame_dir=frame_run_dir)
            summary = frame_pipeline["summary"]
            internals = frame_pipeline["internals"]

            status = status_service.build_frame_status(
                path=internals["path"],
                path_found=internals["path_found"],
                start=internals["start"],
                obs_mask=internals["obs_mask"],
                object_counts=summary["detected_objects"],
                previous_status=prev_status_for_replan,
            )
            prev_status_for_replan = {
                "path_found": status["path_found"],
                "risk": status["current_risk"],
                "obstacle_ratio": status["obstacle_ratio"],
            }

            overlay_path = overlay_frames_dir / f"overlay_{idx:06d}.jpg"
            self._annotate_frame(
                base_image_path=frame_pipeline["files"]["astar_on_rgb"],
                status=status,
                frame_index=idx,
                save_path=overlay_path,
            )
            overlay_paths.append(overlay_path)

            frame_record = {
                "frame_index": idx,
                "source_frame_index": int(frame_meta["source_frame_index"]),
                "timestamp_sec": float(frame_meta["timestamp_sec"]),
                "status": status,
                "summary": {
                    "detected_objects": summary["detected_objects"],
                    "path_length": summary["path_length"],
                    "start": summary["start"],
                    "goal": summary["goal"],
                },
                "artifacts": {
                    "overlay_frame": self._to_outputs_web_path(overlay_path),
                    "astar_on_rgb": frame_pipeline["files_web"]["astar_on_rgb"],
                    "cost_map": frame_pipeline["files_web"]["cost_map"],
                    "depth_visualization": frame_pipeline["files_web"]["depth_visualization"],
                },
                "images": {
                    "original_image": frame_pipeline["files_web"]["original_image"],
                    "detection_overlay": frame_pipeline["files_web"]["detection_overlay"],
                    "risk_overlay": frame_pipeline["files_web"]["risk_overlay"],
                    "astar_on_rgb": frame_pipeline["files_web"]["astar_on_rgb"],
                    "drive_mask": frame_pipeline["files_web"]["drive_mask"],
                    "obstacle_mask": frame_pipeline["files_web"]["obstacle_mask"],
                    "depth_visualization": frame_pipeline["files_web"]["depth_visualization"],
                    "cost_map": frame_pipeline["files_web"]["cost_map"],
                },
            }
            frame_results.append(frame_record)

        result_video_path = run_dir / "result_video.mp4"
        compose_video_from_frames(
            frame_paths=overlay_paths,
            output_path=result_video_path,
            fps=video_info["sample_fps_actual"],
        )

        frames_metadata_path = run_dir / "frames_metadata.json"
        summary_path = run_dir / "summary.json"

        summary = self._build_summary(frame_results=frame_results, video_info=video_info)

        payload = {
            "video_info": video_info,
            "frames": frame_results,
        }
        with frames_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return {
            "status": "ok",
            "result_video_path": self._to_outputs_web_path(result_video_path),
            "frames_metadata_json_path": self._to_outputs_web_path(frames_metadata_path),
            "summary_json_path": self._to_outputs_web_path(summary_path),
            "summary": summary,
            "frames_metadata": payload,
        }


video_service = VideoService()
