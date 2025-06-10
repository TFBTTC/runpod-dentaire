import cv2
import torch
import numpy as np
from statistics import median
from dataclasses import dataclass
from typing import List, Dict, Optional
from ultralytics import YOLO


def load_model(arch_path: str, weights_path: str) -> YOLO:
    model = YOLO(arch_path)
    state = torch.load(weights_path, map_location="cpu")
    model.model.load_state_dict(state)
    return model


model_dents = load_model("models/dents_arch.yaml", "models/dents_weights_only.pt")
model_implants = load_model("models/implants_arch.yaml", "models/implants_weights_only.pt")
model_bridges = load_model("models/bridges_arch.yaml", "models/bridges_weights_only.pt")

# Classification models
model_classes_dent = load_model("models/classes_dent_arch.yaml", "models/classes_dent_weights_only.pt")
model_endo = load_model("models/endo_arch.yaml", "models/endo_weights_only.pt")
model_restauration = load_model("models/restauration_arch.yaml", "models/restauration_weights_only.pt")


@dataclass
class Detection:
    bbox: List[float]
    score: float
    class_name: str
    center: List[float]
    classifications: Optional[Dict[str, Dict[str, float]]] = None


def iou(box_a: List[float], box_b: List[float]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area


def nms(detections: List[Detection], threshold: float = 0.5) -> List[Detection]:
    detections = sorted(detections, key=lambda d: d.score, reverse=True)
    keep: List[Detection] = []
    while detections:
        current = detections.pop(0)
        keep.append(current)
        detections = [d for d in detections if iou(d.bbox, current.bbox) <= threshold]
    return keep


def detect_objects(model: YOLO, image, keep_only: Optional[List[str]] = None) -> List[Detection]:
    results = model(image)
    names = model.names
    detections: List[Detection] = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        class_id = int(box.cls[0])
        score = float(box.conf[0])
        class_name = names[class_id]
        if keep_only and class_name not in keep_only:
            continue
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        detections.append(Detection([x1, y1, x2, y2], score, class_name, center))
    return detections


def crop_and_resize(image, bbox, size: int = 224):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=image.dtype)
    h, w = crop.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=image.dtype)
    x_off = (size - new_w) // 2
    y_off = (size - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def classify_detections(image, detections: List[Detection]) -> None:
    for det in detections:
        tooth = crop_and_resize(image, det.bbox)
        results = {}
        for key, model in [
            ("type", model_classes_dent),
            ("endo", model_endo),
            ("restauration", model_restauration),
        ]:
            res = model(tooth, verbose=False)[0]
            if hasattr(res, "probs") and res.probs is not None:
                idx = int(res.probs.top1)
                score = float(res.probs.data[idx])
                results[key] = {"class_name": model.names[idx], "score": score}
            else:
                results[key] = {"class_name": None, "score": None}
        det.classifications = results


CLASSES_DENTS = ["canine", "central incisor", "lateral incisor", "molar", "premolar"]


def split_into_quadrants(detections: List[Detection], width: int, height: int) -> Dict[int, List[Detection]]:
    mid_x, mid_y = width / 2, height / 2
    quadrants = {1: [], 2: [], 3: [], 4: []}
    for det in detections:
        x, y = det.center
        if y < mid_y:
            quadrant = 1 if x < mid_x else 2
        else:
            quadrant = 4 if x < mid_x else 3
        quadrants[quadrant].append(det)
    return quadrants


def sort_quadrant(dents: List[Detection], quadrant: int, mid_x: float) -> List[Detection]:
    if quadrant in (1, 4):
        return sorted(dents, key=lambda d: d.center[0], reverse=True)
    return sorted(dents, key=lambda d: d.center[0])


def assign_tooth_numbers(dents: List[Detection], quadrant: int, mid_x: float, width: int) -> List[Dict]:
    numbers = []
    if not dents:
        for idx in range(1, 9):
            numbers.append({"FDI": quadrant * 10 + idx, "status": "absente", "center": None})
        return numbers

    dents_sorted = sort_quadrant(dents, quadrant, mid_x)
    xs = [d.center[0] for d in dents_sorted]
    gaps = [abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
    typical = median(gaps) if gaps else width / 16
    threshold = typical * 1.5
    pos = 1
    first_gap = abs(xs[0] - mid_x)
    miss_start = max(0, int(round(first_gap / typical)) - 1) if first_gap > threshold else 0
    for _ in range(miss_start):
        if pos <= 8:
            numbers.append({"FDI": quadrant * 10 + pos, "status": "absente", "center": None})
            pos += 1
    numbers.append({
        "FDI": quadrant * 10 + pos,
        "status": "présente",
        "class_name": dents_sorted[0].class_name,
        "center": dents_sorted[0].center,
        "classifications": dents_sorted[0].classifications,
    })
    pos += 1
    prev_x = xs[0]
    for cur_x, det in zip(xs[1:], dents_sorted[1:]):
        gap = abs(cur_x - prev_x)
        miss = max(0, int(round(gap / typical)) - 1) if gap > threshold else 0
        for _ in range(miss):
            if pos <= 8:
                numbers.append({"FDI": quadrant * 10 + pos, "status": "absente", "center": None})
                pos += 1
        if pos <= 8:
            numbers.append({
                "FDI": quadrant * 10 + pos,
                "status": "présente",
                "class_name": det.class_name,
                "center": det.center,
                "classifications": det.classifications,
            })
            pos += 1
        prev_x = cur_x
    while pos <= 8:
        numbers.append({"FDI": quadrant * 10 + pos, "status": "absente", "center": None})
        pos += 1
    return numbers


def assign_fdi_numbers(quadrants: Dict[int, List[Detection]], width: int) -> List[Dict]:
    all_numbers = []
    mid_x = width / 2
    for q in [1, 2, 3, 4]:
        all_numbers.extend(assign_tooth_numbers(quadrants[q], q, mid_x, width))
    return all_numbers


def process_cv2_image(img):
    h, w = img.shape[:2]
    det_dents = nms(detect_objects(model_dents, img, CLASSES_DENTS))
    classify_detections(img, det_dents)
    det_implants = detect_objects(model_implants, img)
    det_bridges = detect_objects(model_bridges, img)

    quadrants = split_into_quadrants(det_dents, w, h)
    numbers = assign_fdi_numbers(quadrants, w)

    return {
        "dents": numbers,
        "implants": [d.__dict__ for d in det_implants],
        "bridges": [d.__dict__ for d in det_bridges],
    }
