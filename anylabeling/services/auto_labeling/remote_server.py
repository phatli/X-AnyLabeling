import base64
import cv2
import json
import math
import numpy as np
import os
import requests
import time
from pathlib import Path

from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QApplication

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult


class RemoteServer(Model):

    class Meta:
        required_config_names = [
            "type",
            "display_name",
        ]
        widgets = [
            "remote_server_select_combobox",
            "output_label",
            "output_select_combobox",
            "button_remote_server_cleanup",
        ]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
            "rotation": QCoreApplication.translate("Model", "OBB (Rotated)"),
            "rectangle": QCoreApplication.translate("Model", "HBB (Horizontal)"),
        }
        default_output_mode = "rotation"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        self.server_url = self.config.get(
            "server_url",
            os.getenv("XANYLABELING_SERVER_URL", "http://localhost:8000"),
        )
        self.predict_url = f"{self.server_url}/v1/predict"

        print(f'{self.config.get("api_key", "")}')
        self.headers = {
            "Content-Type": "application/json",
            "Token": self.config.get("api_key", ""),
        }

        self.current_model_id = None
        self.timeout = self.config.get("timeout", 30)
        self.models_info = {}
        self.models_info_cache_at = 0.0
        self.models_info_cache_ttl = self._safe_float(
            self.config.get("models_cache_ttl", 3.0), 3.0
        )

        self.marks = []
        self.conf_threshold = 0.0
        self.iou_threshold = 0.0
        self.epsilon_factor = 0.001
        self.replace = True
        self.reset_tracker_flag = False

        # Segment Anything 3
        self.label = None
        self.group_id = None
        self.video_session_id = None
        self.video_initialized = False
        self.video_prompt_frame = None
        self.video_session_image_list = None
        self.video_prompt_type = None
        self.video_has_object = False
        self.video_last_shape = None
        self.video_overlap_iou = self._safe_float(
            self.config.get("video_overlap_iou", 0.5), 0.5
        )
        self.video_max_jump_ratio = self._safe_float(
            self.config.get("video_max_jump_ratio", 3.0), 3.0
        )
        self.video_negative_points_limit = self._safe_int(
            self.config.get("video_negative_points_limit", 5), 5
        )

    def _normalize_output_mode(self) -> str:
        mode = (self.output_mode or "").strip().lower()
        if mode in {"obb", "rotation"}:
            return "rotation"
        if mode in {"hbb", "rectangle", "box"}:
            return "rectangle"
        if mode in {"polygon", "mask", "seg"}:
            return "polygon"
        return self.Meta.default_output_mode

    @staticmethod
    def _safe_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _load_annotation_info(self, image_path):
        annotation_path = Path(image_path).with_suffix(".json")
        if not annotation_path.exists():
            return [], None, None
        try:
            with annotation_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return [], None, None

        shapes = []
        for shape in data.get("shapes", []):
            points = shape.get("points", [])
            if not points:
                continue
            shapes.append(
                {
                    "shape_type": shape.get("shape_type", "polygon"),
                    "points": points,
                }
            )
        return shapes, data.get("imageWidth"), data.get("imageHeight")

    def _shape_bbox(self, shape):
        points = shape.get("points", [])
        if not points:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    def _shape_center_and_size(self, shape):
        bbox = self._shape_bbox(shape)
        if bbox is None:
            return None
        x_min, y_min, x_max, y_max = bbox
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        return cx, cy, max(x_max - x_min, 1.0), max(y_max - y_min, 1.0)

    def _shape_to_mask(self, shape, width, height):
        points = shape.get("points", [])
        if not points:
            return None
        pts = np.array(points, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return None
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        return mask

    def _shape_iou(self, shape_a, shape_b, width, height):
        mask_a = self._shape_to_mask(shape_a, width, height)
        mask_b = self._shape_to_mask(shape_b, width, height)
        if mask_a is None or mask_b is None:
            return 0.0
        inter = np.logical_and(mask_a, mask_b).sum()
        if inter == 0:
            return 0.0
        union = np.logical_or(mask_a, mask_b).sum()
        if union == 0:
            return 0.0
        return float(inter) / float(union)

    def _overlaps_existing(self, shape, existing_shapes, width, height):
        if not existing_shapes or width is None or height is None:
            return False
        threshold = max(self.video_overlap_iou, 0.0)
        if threshold <= 0:
            return False
        for existing in existing_shapes:
            if (
                self._shape_iou(shape, existing, width, height)
                >= threshold
            ):
                return True
        return False

    def _jump_too_large(self, shape, last_shape):
        if last_shape is None:
            return False
        curr = self._shape_center_and_size(shape)
        prev = self._shape_center_and_size(last_shape)
        if curr is None or prev is None:
            return False
        cx, cy, _, _ = curr
        px, py, pw, ph = prev
        max_jump_ratio = max(self.video_max_jump_ratio, 0.0)
        if max_jump_ratio <= 0:
            return False
        dist = math.hypot(cx - px, cy - py)
        limit = max(pw, ph) * max_jump_ratio
        return dist > limit

    def _build_negative_points(self, existing_shapes, pos_points):
        if not existing_shapes:
            return [], []
        negative_points = []
        negative_labels = []
        min_dist = 5.0
        for shape in existing_shapes:
            center = self._shape_center_and_size(shape)
            if center is None:
                continue
            cx, cy, _, _ = center
            too_close = False
            for px, py in pos_points:
                if math.hypot(cx - px, cy - py) <= min_dist:
                    too_close = True
                    break
            if too_close:
                continue
            negative_points.append([cx, cy])
            negative_labels.append(0)
            if (
                self.video_negative_points_limit > 0
                and len(negative_points)
                >= self.video_negative_points_limit
            ):
                break
        return negative_points, negative_labels

    def _extract_error_message(self, result):
        error = result.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return str(message)
        return None

    def set_cache_auto_label(self, text, gid):
        """Set cache auto label"""
        self.label = text
        self.group_id = gid

    def set_model_id(self, model_id):
        self.current_model_id = model_id

    def get_available_models(self):
        """Fetch available models from remote server"""
        now = time.time()
        if (
            self.models_info
            and self.models_info_cache_ttl > 0
            and now - self.models_info_cache_at
            < self.models_info_cache_ttl
        ):
            return self.models_info
        try:
            models_url = f"{self.server_url}/v1/models"
            response = requests.get(
                url=models_url, headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            self.models_info = result.get("data", {})
            self.models_info_cache_at = now
            return self.models_info
        except Exception as e:
            logger.error(f"Failed to fetch available models: {e}")
            if self.models_info:
                return self.models_info
            return {}

    def get_batch_processing_mode(self):
        """Get batch processing mode for current model"""
        if self.current_model_id is None:
            return "default"
        model_info = self.models_info.get(self.current_model_id, {})
        return model_info.get("batch_processing_mode", "default")

    def set_auto_labeling_marks(self, marks):
        """Set auto labeling marks"""
        self.marks = marks

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle the preservation of existing annotations based on the checkbox state."""
        self.replace = not state

    def set_auto_labeling_conf(self, conf_thresh):
        self.conf_threshold = conf_thresh

    def set_auto_labeling_iou(self, iou_thresh):
        self.iou_threshold = iou_thresh

    def set_mask_fineness(self, epsilon):
        self.epsilon_factor = epsilon

    def set_auto_labeling_reset_tracker(self):
        """Reset tracker state for tracking models."""
        self.reset_tracker_flag = True

    def clear_server_cache(self):
        """Release server memory/GPU cache"""
        if not self.current_model_id:
            self.on_message(self.tr("No server model selected; cleanup skipped."))
            return False

        cleanup_url = (
            f"{self.server_url}/v1/models/{self.current_model_id}/cleanup"
        )
        try:
            response = requests.post(
                url=cleanup_url, headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json() if response.content else {}
            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            sessions = data.get("sessions_cleared")
            tasks = data.get("tasks_cancelled")
            if sessions is not None or tasks is not None:
                message = self.tr(
                    "Released server cache. Sessions cleared: {sessions}, tasks cancelled: {tasks}."
                ).format(
                    sessions=sessions if sessions is not None else 0,
                    tasks=tasks if tasks is not None else 0,
                )
            else:
                message = self.tr("Cleanup request sent to server.")
            self.on_message(message)
            self._reset_video_session()
            return True
        except Exception as e:
            logger.error(f"Server cleanup error: {e}")
            self.on_message(
                self.tr("Failed to clean server cache: {error}").format(error=str(e))
            )
            return False

    def predict_shapes(
        self, image, image_path=None, text_prompt=None, run_tracker=False
    ):
        if image is None:
            return AutoLabelingResult([], replace=self.replace)

        if self.current_model_id is None:
            logger.warning("No model selected")
            return AutoLabelingResult([], replace=self.replace)

        batch_mode = self.get_batch_processing_mode()
        if batch_mode == "video" and text_prompt:
            return self._handle_video_prompt(image_path, text_prompt)
        elif batch_mode == "video" and self.marks:
            result = self._handle_video_point_prompt(image_path)
            if run_tracker:
                logger.info("Starting video propagation after point prompt...")
                return self._handle_video_propagation()
            return result
        elif batch_mode == "video" and run_tracker:
            logger.info("Starting video propagation...")
            return self._handle_video_propagation()

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".bmp": "image/bmp",
                ".webp": "image/webp",
            }.get(ext, "image/jpeg")
            img_data_uri = f"data:{mime_type};base64,{img_base64}"
        else:
            try:
                cv_image = qt_img_to_rgb_cv_img(image, image_path)
            except Exception as e:
                logger.warning(f"Could not process image: {e}")
                return AutoLabelingResult([], replace=self.replace)

            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            is_success, buffer = cv2.imencode(".png", cv_image_bgr)
            if not is_success:
                raise ValueError("Failed to encode image.")
            img_base64 = base64.b64encode(buffer).decode("utf-8")
            img_data_uri = f"data:image/png;base64,{img_base64}"

        params = {}
        params["conf_threshold"] = self.conf_threshold
        params["iou_threshold"] = self.iou_threshold
        params["epsilon_factor"] = self.epsilon_factor
        params["output_mode"] = self._normalize_output_mode()
        if text_prompt:
            logger.debug(f"Received text prompt: {text_prompt}")
            params["text_prompt"] = text_prompt.rstrip(".")
        if self.marks:
            params["marks"] = self.marks
        if self.reset_tracker_flag:
            params["reset_tracker"] = True
            self.reset_tracker_flag = False

        payload = {
            "model": self.current_model_id,
            "image": img_data_uri,
            "params": params,
        }
        logger.debug(
            f"Sending request to {self.predict_url} with payload: "
            f"model: {self.set_model_id}, "
            f"paramters: {params}"
        )

        try:
            response = requests.post(
                url=self.predict_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Remote server prediction result: {result}")

            data = result.get("data", {})
            shapes = []
            for shape_data in data.get("shapes", []):
                shape = Shape(
                    label=shape_data["label"],
                    shape_type=shape_data["shape_type"],
                    attributes=shape_data.get("attributes", {}),
                    description=shape_data.get("description", None),
                    difficult=shape_data.get("difficult", False),
                    direction=shape_data.get("direction", 0),
                    flags=shape_data.get("flags", None),
                    group_id=shape_data.get("group_id", None),
                    kie_linking=shape_data.get("kie_linking", []),
                    score=shape_data.get("score", None),
                )

                for point in shape_data["points"]:
                    shape.add_point(QtCore.QPointF(point[0], point[1]))

                shapes.append(shape)

            description = data.get("description", "")

            replace = data.get("replace")
            if replace is None:
                replace = self.replace

            return AutoLabelingResult(
                shapes, replace=replace, description=description
            )

        except Exception as e:
            logger.error(f"Remote server error: {e}")
            self.on_message(f"Remote server error: {e}")
            return AutoLabelingResult([], replace=self.replace)

    def _handle_video_prompt(self, image_path, text_prompt):
        """Handle video prompt: initialize or reuse session and add prompt.

        Args:
            image_path: Path to current image file.
            text_prompt: Text prompt string.

        Returns:
            AutoLabelingResult with shapes from prompt frame.
        """
        if not image_path or not os.path.exists(image_path):
            return AutoLabelingResult([], replace=self.replace)

        try:
            image_list = []
            widget = getattr(self, "_widget", None)
            if widget:
                image_list = getattr(widget, "image_list", [])

            if not image_list:
                dir_path = Path(image_path).parent
                valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                all_images = [
                    str(file_path)
                    for file_path in dir_path.iterdir()
                    if file_path.is_file()
                    and file_path.suffix.lower() in valid_extensions
                ]

                try:
                    all_images.sort(
                        key=lambda p: int(
                            "".join(filter(str.isdigit, Path(p).name))
                        )
                    )
                except ValueError:
                    all_images.sort()

                image_list = all_images

            current_index = 0
            try:
                current_index = image_list.index(image_path)
            except ValueError:
                logger.warning(
                    f"Current image {image_path} not found in image list, using index 0"
                )

            logger.info(
                f"Video prompt: current_frame_index={current_index}, "
                f"image_path={image_path}, total_frames={len(image_list)}"
            )

            if not image_list:
                return AutoLabelingResult([], replace=self.replace)

            if (
                self.video_session_id
                and self.video_session_image_list != image_list
            ):
                logger.info("Image list changed, resetting video session")
                self._reset_video_session()

            if self.video_session_id and self.video_prompt_type != "text":
                logger.info(
                    "Switching from point prompt to text prompt, resetting video session"
                )
                self._reset_video_session()

            if not self.video_session_id:
                frames_data = []
                for frame_path in image_list:
                    if os.path.exists(frame_path):
                        with open(frame_path, "rb") as f:
                            frame_base64 = base64.b64encode(f.read()).decode(
                                "utf-8"
                            )
                        ext = os.path.splitext(frame_path)[1].lower()
                        mime_type = {
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".png": "image/png",
                            ".bmp": "image/bmp",
                            ".webp": "image/webp",
                        }.get(ext, "image/jpeg")
                        frames_data.append(
                            f"data:{mime_type};base64,{frame_base64}"
                        )

                message = self.tr(
                    "Packing completed, initializing video session... "
                    "(This may take some time, please wait patiently)"
                )
                logger.info(message)
                self.on_message(message)

                init_url = f"{self.server_url}/v1/video/init"
                init_response = requests.post(
                    url=init_url,
                    json={
                        "model": self.current_model_id,
                        "frames": frames_data,
                        "start_frame_index": 0,
                    },
                    headers=self.headers,
                    timeout=self.timeout * 2,
                )
                init_response.raise_for_status()
                init_result = init_response.json()
                if init_result.get("success") is False:
                    message = self._extract_error_message(init_result)
                    if message:
                        self.on_message(message)
                    self._reset_video_session()
                    return AutoLabelingResult([], replace=self.replace)
                self.video_session_id = init_result.get("data", {}).get(
                    "session_id"
                )
                if not self.video_session_id:
                    message = self._extract_error_message(init_result)
                    if message:
                        self.on_message(message)
                    self._reset_video_session()
                    return AutoLabelingResult([], replace=self.replace)
                self.video_initialized = True
                self.video_session_image_list = image_list.copy()
                logger.info(
                    f"Video session initialized: {self.video_session_id}"
                )

            prompt_url = f"{self.server_url}/v1/video/prompt"
            prompt_response = requests.post(
                url=prompt_url,
                json={
                    "session_id": self.video_session_id,
                    "model": self.current_model_id,
                    "text_prompt": text_prompt.rstrip("."),
                    "frame_index": current_index,
                    "params": {
                        "conf_threshold": self.conf_threshold,
                        "epsilon_factor": self.epsilon_factor,
                        "output_mode": self._normalize_output_mode(),
                    },
                },
                headers=self.headers,
                timeout=self.timeout,
            )
            prompt_response.raise_for_status()
            prompt_result = prompt_response.json()
            if prompt_result.get("success") is False:
                message = self._extract_error_message(prompt_result)
                if message:
                    self.on_message(message)
                self._reset_video_session()
                return AutoLabelingResult([], replace=self.replace)
            data = prompt_result.get("data", {})

            self.video_prompt_frame = current_index
            self.video_prompt_type = "text"

            shapes = []
            for shape_data in data.get("masks", []):
                shape = Shape(
                    label=shape_data.get("label", "AUTOLABEL_OBJECT"),
                    shape_type=shape_data.get("shape_type", "rectangle"),
                    score=shape_data.get("score", None),
                    group_id=shape_data.get("group_id", None),
                )
                for point in shape_data.get("points", []):
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shapes.append(shape)

            return AutoLabelingResult(
                shapes, replace=self.replace, description=""
            )

        except Exception as e:
            logger.error(f"Video prompt error: {e}")
            self.on_message(f"Video prompt error: {e}")
            self._reset_video_session()
            return AutoLabelingResult([], replace=self.replace)

    def _handle_video_point_prompt(self, image_path):
        """Handle video point prompt: initialize or reuse session and add point prompt.

        Args:
            image_path: Path to current image file.

        Returns:
            AutoLabelingResult with shapes from prompt frame.
        """
        if not image_path or not os.path.exists(image_path):
            return AutoLabelingResult([], replace=False)

        try:
            image_list = []
            widget = getattr(self, "_widget", None)
            if widget:
                image_list = getattr(widget, "image_list", [])

            if not image_list:
                dir_path = Path(image_path).parent
                valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                all_images = [
                    str(file_path)
                    for file_path in dir_path.iterdir()
                    if file_path.is_file()
                    and file_path.suffix.lower() in valid_extensions
                ]

                try:
                    all_images.sort(
                        key=lambda p: int(
                            "".join(filter(str.isdigit, Path(p).name))
                        )
                    )
                except ValueError:
                    all_images.sort()

                image_list = all_images

            current_index = 0
            try:
                current_index = image_list.index(image_path)
            except ValueError:
                logger.warning(
                    f"Current image {image_path} not found in image list, using index 0"
                )

            logger.info(
                f"Video point prompt: current_frame_index={current_index}, "
                f"image_path={image_path}, total_frames={len(image_list)}"
            )

            if not image_list:
                return AutoLabelingResult([], replace=False)

            if (
                self.video_session_id
                and self.video_session_image_list != image_list
            ):
                logger.info("Image list changed, resetting video session")
                self._reset_video_session()

            if self.video_session_id and self.video_prompt_type != "point":
                logger.info(
                    "Switching from text prompt to point prompt, resetting video session"
                )
                self._reset_video_session()

            if not self.video_session_id:
                frames_data = []
                for frame_path in image_list:
                    if os.path.exists(frame_path):
                        with open(frame_path, "rb") as f:
                            frame_base64 = base64.b64encode(f.read()).decode(
                                "utf-8"
                            )
                        ext = os.path.splitext(frame_path)[1].lower()
                        mime_type = {
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".png": "image/png",
                            ".bmp": "image/bmp",
                            ".webp": "image/webp",
                        }.get(ext, "image/jpeg")
                        frames_data.append(
                            f"data:{mime_type};base64,{frame_base64}"
                        )

                message = self.tr(
                    "Packing completed, initializing video session... "
                    "(This may take some time, please wait patiently)"
                )
                logger.info(message)
                self.on_message(message)

                init_url = f"{self.server_url}/v1/video/init"
                init_response = requests.post(
                    url=init_url,
                    json={
                        "model": self.current_model_id,
                        "frames": frames_data,
                        "start_frame_index": 0,
                    },
                    headers=self.headers,
                    timeout=self.timeout * 2,
                )
                init_response.raise_for_status()
                init_result = init_response.json()
                if init_result.get("success") is False:
                    message = self._extract_error_message(init_result)
                    if message:
                        self.on_message(message)
                    self._reset_video_session()
                    return AutoLabelingResult([], replace=False)
                self.video_session_id = init_result.get("data", {}).get(
                    "session_id"
                )
                if not self.video_session_id:
                    message = self._extract_error_message(init_result)
                    if message:
                        self.on_message(message)
                    self._reset_video_session()
                    return AutoLabelingResult([], replace=False)
                self.video_initialized = True
                self.video_session_image_list = image_list.copy()
                logger.info(
                    f"Video session initialized: {self.video_session_id}"
                )

            frame_img = cv2.imread(image_path)
            if frame_img is None:
                return AutoLabelingResult([], replace=False)
            img_height, img_width = frame_img.shape[:2]

            point_coords = []
            point_labels = []

            for mark in self.marks:
                mark_type = mark.get("type")
                if mark_type == "point":
                    data = mark.get("data", [])
                    if len(data) == 2:
                        point_coords.append(data)
                        label = mark.get("label", 1)
                        point_labels.append(label)

            if not point_coords:
                logger.warning("No valid point prompts provided")
                return AutoLabelingResult([], replace=False)

            points_abs = np.array(point_coords, dtype=np.float32)
            labels = np.array(point_labels, dtype=np.int32)

            points_rel = points_abs / np.array(
                [img_width, img_height], dtype=np.float32
            )
            points_tensor = points_rel.tolist()
            point_labels_tensor = labels.tolist()

            existing_shapes, image_width, image_height = (
                self._load_annotation_info(image_path)
            )
            if existing_shapes:
                extra_points, extra_labels = self._build_negative_points(
                    existing_shapes, points_abs.tolist()
                )
                if extra_points:
                    extra_points_rel = (
                        np.array(extra_points, dtype=np.float32)
                        / np.array(
                            [img_width, img_height], dtype=np.float32
                        )
                    ).tolist()
                    points_tensor.extend(extra_points_rel)
                    point_labels_tensor.extend(extra_labels)

            prompt_url = f"{self.server_url}/v1/video/prompt"
            prompt_params = {
                "conf_threshold": self.conf_threshold,
                "epsilon_factor": self.epsilon_factor,
                "output_mode": self._normalize_output_mode(),
            }
            if self.reset_tracker_flag or self.video_has_object:
                prompt_params["reset_tracker"] = True
                self.reset_tracker_flag = False
            prompt_response = requests.post(
                url=prompt_url,
                json={
                    "session_id": self.video_session_id,
                    "model": self.current_model_id,
                    "frame_index": current_index,
                    "points": points_tensor,
                    "point_labels": point_labels_tensor,
                    "obj_id": 10000,
                    "params": prompt_params,
                },
                headers=self.headers,
                timeout=self.timeout,
            )
            prompt_response.raise_for_status()
            prompt_result = prompt_response.json()
            if prompt_result.get("success") is False:
                message = self._extract_error_message(prompt_result)
                if message:
                    self.on_message(message)
                self._reset_video_session()
                return AutoLabelingResult([], replace=False)
            data = prompt_result.get("data", {})

            self.video_prompt_frame = current_index
            self.video_prompt_type = "point"

            shapes = []
            for shape_data in data.get("masks", []):
                shape = Shape(
                    label="AUTOLABEL_OBJECT",
                    shape_type=shape_data.get("shape_type", "rectangle"),
                    score=shape_data.get("score", None),
                    group_id=shape_data.get("group_id", None),
                )
                for point in shape_data.get("points", []):
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shapes.append(shape)

            return AutoLabelingResult(shapes, replace=False, description="")

        except Exception as e:
            logger.error(f"Video point prompt error: {e}")
            self.on_message(f"Video point prompt error: {e}")
            self._reset_video_session()
            return AutoLabelingResult([], replace=False)

    def _handle_video_propagation(self):
        """Handle video propagation using SSE streaming."""
        if not self.video_session_id:
            logger.warning("No video session initialized")
            return AutoLabelingResult([], replace=False)

        widget = getattr(self, "_widget", None)
        image_list = getattr(widget, "image_list", []) if widget else []
        progress_dialog = (
            getattr(widget, "_progress_dialog", None) if widget else None
        )

        try:
            stream_url = f"{self.server_url}/v1/video/propagate/stream"
            logger.info(
                f"Starting streaming propagation: session_id={self.video_session_id}"
            )

            request_json = {
                "session_id": self.video_session_id,
                "model": self.current_model_id,
            }
            if self.video_prompt_frame is not None:
                request_json["start_frame"] = self.video_prompt_frame

            response = requests.post(
                url=stream_url,
                json=request_json,
                headers=self.headers,
                stream=True,
                timeout=None,
            )
            response.raise_for_status()

            results = {}
            total_frames = 0
            start_idx = None
            end_idx = None
            frames_to_process = 0
            is_first_progress = True

            cancelled = False
            for line in response.iter_lines(decode_unicode=True):
                if widget and getattr(widget, "cancel_processing", False):
                    logger.info("Propagation cancelled by user")
                    cancelled = True
                    response.close()
                    break

                if not line:
                    continue

                if not line.startswith("data: "):
                    continue

                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                if event_type == "started":
                    total_frames = event.get("total_frames", 0)
                    start_frame_index = event.get("start_frame_index", 0)
                    start_idx = self.video_prompt_frame or start_frame_index
                    if total_frames > 0:
                        end_idx = total_frames - 1
                    else:
                        end_idx = start_idx
                    if end_idx < start_idx:
                        end_idx = start_idx
                    frames_to_process = (
                        end_idx - start_idx + 1
                        if total_frames > 0
                        else 0
                    )
                    if progress_dialog and frames_to_process > 0:
                        progress_dialog.setMaximum(frames_to_process)
                        progress_dialog.setValue(0)
                        progress_dialog.setLabelText(
                            self.tr("Model warming up, please wait...")
                        )
                        QApplication.processEvents()
                    logger.info(
                        f"Propagation started: total_frames={total_frames}, "
                        f"start_frame_index={start_frame_index}"
                    )

                elif event_type == "progress":
                    current_frame = event.get("current_frame", 0)
                    if start_idx is None:
                        start_idx = self.video_prompt_frame or 0
                    if end_idx is None:
                        end_idx = total_frames - 1
                    if end_idx < start_idx:
                        end_idx = start_idx
                    if frames_to_process <= 0 and total_frames > 0:
                        frames_to_process = end_idx - start_idx + 1
                    relative_frame = current_frame - start_idx
                    if relative_frame < 0:
                        relative_frame = 0
                    display_frame = relative_frame + 1
                    if frames_to_process > 0:
                        display_frame = min(display_frame, frames_to_process)

                    if progress_dialog:
                        try:
                            if is_first_progress:
                                is_first_progress = False
                                QApplication.processEvents()

                            if display_frame <= frames_to_process:
                                progress_dialog.setValue(display_frame)
                            template = self.tr("Processing frame %s/%s")
                            message_text = template % (
                                display_frame,
                                frames_to_process,
                            )
                            progress_dialog.setLabelText(message_text)
                            QApplication.processEvents()
                        except Exception as e:
                            logger.warning(f"Error updating progress: {e}")

                elif event_type == "completed":
                    results = event.get("results", {})
                    logger.info(
                        f"Propagation completed: {len(results)} frame results"
                    )

                elif event_type == "error":
                    error_msg = event.get("message", "Unknown error")
                    logger.error(f"Propagation error: {error_msg}")
                    self.on_message(f"Propagation failed: {error_msg}")
                    self._reset_video_session()
                    return AutoLabelingResult([], replace=False)

            if cancelled:
                self._reset_video_session()
                return AutoLabelingResult([], replace=False)

            if not results:
                logger.warning("No results received from propagation")
                return AutoLabelingResult([], replace=False)

            if not widget or not image_list:
                logger.warning("Widget or image list not available")
                return AutoLabelingResult([], replace=False)

            try:
                frame_indices = sorted(
                    int(key) for key in results.keys()
                )
            except Exception:
                frame_indices = []
            if frame_indices:
                logger.info(
                    "前向推理结果统计: 结果帧数=%d, 帧范围=%d-%d, image_list=%d, prompt_frame=%s",
                    len(frame_indices),
                    frame_indices[0],
                    frame_indices[-1],
                    len(image_list),
                    str(self.video_prompt_frame),
                )
            else:
                logger.info(
                    "前向推理结果统计: 结果帧数=0, image_list=%d, prompt_frame=%s",
                    len(image_list),
                    str(self.video_prompt_frame),
                )

            stats = {
                "saved_frames": 0,
                "skipped_out_of_range": 0,
                "skipped_no_masks": 0,
                "skipped_no_points": 0,
                "skipped_overlap": 0,
                "skipped_jump": 0,
                "skipped_all_filtered": 0,
            }
            samples = {
                "out_of_range": [],
                "no_masks": [],
                "no_points": [],
                "overlap": [],
                "jump": [],
                "all_filtered": [],
            }

            self.video_last_shape = None
            saved_count = 0
            for frame_idx_str, frame_result in results.items():
                try:
                    frame_idx = int(frame_idx_str)
                    if 0 <= frame_idx < len(image_list):
                        frame_file = image_list[frame_idx]
                        masks = frame_result.get("masks", [])
                        if not masks:
                            stats["skipped_no_masks"] += 1
                            if len(samples["no_masks"]) < 10:
                                samples["no_masks"].append(frame_idx)
                            continue

                        existing_shapes, img_width, img_height = (
                            self._load_annotation_info(frame_file)
                        )
                        shapes = []
                        frame_has_points = False
                        frame_has_overlap = False
                        frame_has_jump = False
                        for shape_data in masks:
                            points = shape_data.get("points", [])
                            if not points:
                                frame_has_points = True
                                stats["skipped_no_points"] += 1
                                if len(samples["no_points"]) < 10:
                                    samples["no_points"].append(frame_idx)
                                continue
                            frame_has_points = True
                            if self._overlaps_existing(
                                shape_data,
                                existing_shapes,
                                img_width,
                                img_height,
                            ):
                                frame_has_overlap = True
                                continue
                            if self._jump_too_large(
                                shape_data, self.video_last_shape
                            ):
                                frame_has_jump = True
                                continue
                            label = shape_data.get("label", "AUTOLABEL_OBJECT")
                            if self.label is not None:
                                label = self.label
                            group_id = shape_data.get("group_id", None)
                            if self.group_id is not None:
                                group_id = self.group_id
                            shape = Shape(
                                label=label,
                                shape_type=shape_data.get(
                                    "shape_type", "rectangle"
                                ),
                                score=shape_data.get("score", None),
                                group_id=group_id,
                            )
                            for point in points:
                                shape.add_point(
                                    QtCore.QPointF(point[0], point[1])
                                )
                            if shape.points:
                                shapes.append(shape)
                                self.video_last_shape = shape_data

                        if shapes:
                            from anylabeling.views.labeling.utils.batch import (
                                save_auto_labeling_result,
                            )

                            replace_value = (
                                False
                                if self.video_prompt_type == "point"
                                else self.replace
                            )
                            save_auto_labeling_result(
                                widget,
                                frame_file,
                                AutoLabelingResult(
                                    shapes, replace=replace_value
                                ),
                            )
                            saved_count += 1
                            stats["saved_frames"] += 1
                        else:
                            stats["skipped_all_filtered"] += 1
                            if len(samples["all_filtered"]) < 10:
                                samples["all_filtered"].append(frame_idx)
                            if frame_has_overlap:
                                stats["skipped_overlap"] += 1
                                if len(samples["overlap"]) < 10:
                                    samples["overlap"].append(frame_idx)
                            if frame_has_jump:
                                stats["skipped_jump"] += 1
                                if len(samples["jump"]) < 10:
                                    samples["jump"].append(frame_idx)
                    else:
                        stats["skipped_out_of_range"] += 1
                        if len(samples["out_of_range"]) < 10:
                            samples["out_of_range"].append(frame_idx)
                        continue
                except (ValueError, KeyError) as e:
                    logger.warning(
                        f"Error processing frame {frame_idx_str}: {e}"
                    )

            self.label, self.group_id = None, None
            if saved_count > 0:
                self.video_has_object = True
            logger.info(
                "前向推理保存统计: 保存=%d, 越界=%d, 无mask=%d, 无points=%d, 重叠过滤=%d, 跳跃过滤=%d, 全部被过滤=%d",
                stats["saved_frames"],
                stats["skipped_out_of_range"],
                stats["skipped_no_masks"],
                stats["skipped_no_points"],
                stats["skipped_overlap"],
                stats["skipped_jump"],
                stats["skipped_all_filtered"],
            )
            logger.info(
                "前向推理样本帧: 越界=%s, 无mask=%s, 无points=%s, 重叠=%s, 跳跃=%s, 全部过滤=%s",
                samples["out_of_range"],
                samples["no_masks"],
                samples["no_points"],
                samples["overlap"],
                samples["jump"],
                samples["all_filtered"],
            )
            logger.info(f"Saved results for {saved_count} frames")
            return AutoLabelingResult([], replace=False)

        except requests.exceptions.RequestException as e:
            logger.error(f"Video propagation request error: {e}")
            self.on_message(f"Video propagation error: {e}")
            self._reset_video_session()
            self.label, self.group_id = None, None
            return AutoLabelingResult([], replace=False)
        except Exception as e:
            logger.error(f"Video propagation error: {e}", exc_info=True)
            self.on_message(f"Video propagation error: {e}")
            self._reset_video_session()
            self.label, self.group_id = None, None
            return AutoLabelingResult([], replace=False)

    def _reset_video_session(self):
        """Reset video session state and cleanup server session."""
        if self.video_session_id:
            try:
                cleanup_url = f"{self.server_url}/v1/video/cleanup/{self.video_session_id}"
                requests.post(
                    url=cleanup_url,
                    params={"model": self.current_model_id},
                    headers=self.headers,
                    timeout=self.timeout,
                )
                logger.debug(f"Cleaned up session {self.video_session_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup session: {e}")

        self.label = None
        self.group_id = None
        self.video_has_object = False
        self.video_last_shape = None
        self.video_session_id = None
        self.video_initialized = False
        self.video_prompt_frame = None
        self.video_session_image_list = None
        self.video_prompt_type = None

    def unload(self):
        """Unload the model"""
        pass
