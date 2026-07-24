from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import math
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


@dataclass
class TrackedObject:
    """
    Structured object representation for tracked entities in Sentinel.

    Encapsulates:
    - Track ID (persistent object identifier)
    - Class (class name & COCO ID)
    - Bottom-Center Coordinates ((cx, y2) ground contact position)
    - Velocity / Direction ((vx, vy) vector, speed, angle)
    - Active Zone (current containing ROI name)
    - Dwell Time (duration in seconds tracked in scene/zone)
    - Crop Thumbnail (numpy image crop patch)
    """

    track_id: int
    class_name: str
    class_id: int = 0
    bottom_center: Tuple[float, float] = (0.0, 0.0)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)
    active_zone: Optional[str] = None
    dwell_time: float = 0.0
    confidence: float = 0.0
    crop_thumbnail: Optional[np.ndarray] = field(default=None, repr=False)

    def __getitem__(self, key: str) -> Union[int, str, float, Tuple[float, float], Optional[str], Optional[np.ndarray]]:
        """Allows dictionary-style key access (e.g. obj['track_id'])."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    @classmethod
    def create(
        cls,
        track_id: int,
        class_name: str,
        bbox: Tuple[float, float, float, float],
        frame: Optional[np.ndarray] = None,
        class_id: int = 0,
        velocity: Tuple[float, float] = (0.0, 0.0),
        active_zone: Optional[str] = None,
        dwell_time: float = 0.0,
        confidence: float = 0.0,
    ) -> "TrackedObject":
        """
        Factory method to instantiate a TrackedObject and automatically slice the thumbnail crop from frame.
        """
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cx = (x1 + x2) / 2.0
        cy_bottom = float(y2)

        crop = None
        if frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            crop_x1, crop_y1 = max(0, x1), max(0, y1)
            crop_x2, crop_y2 = min(w, x2), min(h, y2)
            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        return cls(
            track_id=track_id,
            class_name=class_name,
            class_id=class_id,
            bottom_center=(cx, cy_bottom),
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            velocity=velocity,
            active_zone=active_zone,
            dwell_time=dwell_time,
            confidence=confidence,
            crop_thumbnail=crop,
        )

    @property
    def speed(self) -> float:
        """Calculates scalar speed magnitude (pixels per second or frame)."""
        vx, vy = self.velocity
        return math.hypot(vx, vy)

    @property
    def direction_angle(self) -> float:
        """Calculates movement heading angle in degrees (0 to 360)."""
        vx, vy = self.velocity
        if vx == 0 and vy == 0:
            return 0.0
        angle = math.degrees(math.atan2(vy, vx))
        return (angle + 360) % 360

    def save_thumbnail(self, file_path: str) -> bool:
        """Saves crop thumbnail image patch to disk if available."""
        if self.crop_thumbnail is None or self.crop_thumbnail.size == 0:
            return False
        return cv2.imwrite(file_path, self.crop_thumbnail)

    def to_dict(self, include_crop: bool = False) -> Dict[str, Union[int, str, float, List[float], Optional[str], Optional[np.ndarray]]]:
        """Converts TrackedObject representation into a clean dictionary."""
        d = {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "bottom_center": [round(self.bottom_center[0], 2), round(self.bottom_center[1], 2)],
            "bbox": [round(v, 2) for v in self.bbox],
            "velocity": [round(self.velocity[0], 2), round(self.velocity[1], 2)],
            "speed": round(self.speed, 2),
            "direction_angle": round(self.direction_angle, 1),
            "active_zone": self.active_zone,
            "dwell_time": round(self.dwell_time, 2),
            "confidence": round(self.confidence, 3),
        }
        if include_crop:
            d["crop_thumbnail"] = self.crop_thumbnail
        return d


def load_model(model_name: str = "yolo11n.pt") -> YOLO:
    """
    Loads a YOLO model from Ultralytics.

    Args:
        model_name: Name or path of the YOLO model weights (e.g. 'yolo11n.pt', 'yolov8n.pt')

    Returns:
        Loaded YOLO model instance.
    """
    return YOLO(model_name)


def track_frame(
    model: YOLO,
    frame: np.ndarray,
    persist: bool = True,
    tracker: str = "bytetrack.yaml",
    conf: float = 0.25,
    classes: Optional[List[int]] = None,
    verbose: bool = False,
) -> Results:
    """
    Wrapper around YOLO model.track() to execute persistent tracking on a single frame.

    Args:
        model: Loaded YOLO model.
        frame: BGR image frame (numpy array from OpenCV / VideoReader).
        persist: Retain track IDs across consecutive calls (crucial for video stream tracking).
        tracker: Tracker configuration file ('bytetrack.yaml' or 'botsort.yaml').
        conf: Confidence threshold for detection filtering.
        classes: List of class IDs to filter (e.g. [0] for person).
        verbose: Print tracking output logs.

    Returns:
        Ultralytics Results object for the frame.
    """
    results = model.track(
        source=frame,
        persist=persist,
        tracker=tracker,
        conf=conf,
        classes=classes,
        verbose=verbose,
    )
    return results[0]


def extract_tracks(result: Results, frame: Optional[np.ndarray] = None) -> List[TrackedObject]:
    """
    Extracts structured TrackedObject representations from a single YOLO result object.

    Args:
        result: Ultralytics Results object from track_frame().
        frame: Optional original BGR frame image array to slice thumbnails.

    Returns:
        List of TrackedObject instances.
    """
    objects = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return objects

    xyxys = boxes.xyxy
    confs = boxes.conf
    classes = boxes.cls
    names = result.names

    if isinstance(xyxys, torch.Tensor):
        xyxys_np = xyxys.cpu().numpy()
    else:
        xyxys_np = np.asarray(xyxys)

    if isinstance(confs, torch.Tensor):
        confs_np = confs.cpu().numpy()
    else:
        confs_np = np.asarray(confs)

    if isinstance(classes, torch.Tensor):
        classes_np = classes.int().cpu().numpy()
    else:
        classes_np = np.asarray(classes, dtype=int)

    if boxes.id is not None:
        box_ids = boxes.id
        if isinstance(box_ids, torch.Tensor):
            track_ids = box_ids.int().cpu().numpy()
        else:
            track_ids = np.asarray(box_ids, dtype=int)
    else:
        track_ids = [None] * len(boxes)

    for bbox, conf, cls_id, tid in zip(xyxys_np, confs_np, classes_np, track_ids):
        if tid is None:
            continue
        obj = TrackedObject.create(
            track_id=int(tid),
            class_name=names.get(int(cls_id), str(cls_id)),
            class_id=int(cls_id),
            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            confidence=float(conf),
            frame=frame,
        )
        objects.append(obj)

    return objects
