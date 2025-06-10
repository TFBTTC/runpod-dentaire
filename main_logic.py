import cv2
import json
import os





from ultralytics import YOLO
import torch

# Charger l'architecture (par défaut YOLOv8n, ou adapte si tu as utilisé yolov8m, etc.)
model_dents = YOLO("models/dents_arch.yaml")  # ou le fichier YAML correspondant à ton modèle
model_dents.model.load_state_dict(torch.load("models/dents_weights_only.pt", map_location="cpu"))

model_implants = YOLO("models/implants_arch.yaml")
model_implants.model.load_state_dict(torch.load("models/implants_weights_only.pt", map_location="cpu"))

model_bridges = YOLO("models/bridges_arch.yaml")
model_bridges.model.load_state_dict(torch.load("models/bridges_weights_only.pt", map_location="cpu"))

CLASSES_DENTS = ['canine', 'central incisor', 'lateral incisor', 'molar', 'premolar']
CLASSES_IMPLANT = ['Implant']
CLASSES_BRIDGE = ['bridge']

FDI_TEMPLATE = {
    1: list(range(18, 10, -1)),
    2: list(range(21, 29)),
    3: list(range(38, 30, -1)),
    4: list(range(31, 39))
}


# === UTILS ===
def midpoint(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def quadrant_from_y(y, y_split):
    return 1 if y < y_split else 3

def assign_to_quadrants(detections, y_split):
    quadrants = {1: [], 2: [], 3: [], 4: []}
    for d in detections:
        cls = d['class_name']
        x, y = d['center']
        if y < y_split:
            if x < 640:
                quadrants[1].append(d)
            else:
                quadrants[2].append(d)
        else:
            if x < 640:
                quadrants[4].append(d)
            else:
                quadrants[3].append(d)
    return quadrants

def export_summary(dents_fdi, implants_fdi, bridges_fdi):
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write("--- Schéma dentaire (FDI) ---\n")
        for dent in sorted(dents_fdi, key=lambda d: d['FDI']):
            f.write(f"Dent {dent['FDI']} : {dent['class_name']} — {dent['status']}\n")

        f.write("\n--- Implants ---\n")
        for imp in implants_fdi:
            f.write(f"Implant remplace la dent {imp['FDI']}\n")

        f.write("\n--- Bridges ---\n")
        for br in bridges_fdi:
            f.write(f"Bridge de {br['FDI'][0]} à {br['FDI'][-1]}\n")


from ultralytics import YOLO
import cv2

# Chargement des modèles
model_dents = YOLO("models/dents.pt")
model_implants = YOLO("models/implants.pt")
model_bridges = YOLO("models/bridges.pt")

# Définition des classes utiles
CLASSES_DENTS = ['canine', 'central incisor', 'lateral incisor', 'molar', 'premolar']
CLASSES_IMPLANT = ['Implant']
CLASSES_BRIDGE = ['bridge']

FDI_TEMPLATE = {
    1: list(range(18, 10, -1)),
    2: list(range(21, 29)),
    3: list(range(38, 30, -1)),
    4: list(range(31, 39))
}

# === UTILS ===
def midpoint(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def assign_to_quadrants(detections, y_split):
    quadrants = {1: [], 2: [], 3: [], 4: []}
    for d in detections:
        x, y = d['center']
        if y < y_split:
            if x < 640:
                quadrants[1].append(d)
            else:
                quadrants[2].append(d)
        else:
            if x < 640:
                quadrants[4].append(d)
            else:
                quadrants[3].append(d)
    return quadrants

def export_summary(dents_fdi, implants_fdi, bridges_fdi):
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write("--- Schéma dentaire (FDI) ---\n")
        for dent in sorted(dents_fdi, key=lambda d: d['FDI']):
            f.write(f"Dent {dent['FDI']} : {dent['class_name']} — {dent['status']}\n")

        f.write("\n--- Implants ---\n")
        for imp in implants_fdi:
            f.write(f"Implant remplace la dent {imp['FDI']}\n")

        f.write("\n--- Bridges ---\n")
        for br in bridges_fdi:
            f.write(f"Bridge de {br['FDI'][0]} à {br['FDI'][-1]}\n")


# === Fonction principale ===
import cv2
import numpy as np
from ultralytics import YOLO

FDI_ORDER = {
    1: list(range(18, 10, -1)),
    2: list(range(21, 29)),
    3: list(range(38, 30, -1)),
    4: list(range(41, 49))
}

FDI_CLASS_MAP = {
    "central incisor": [1],
    "lateral incisor": [2],
    "canine": [3],
    "premolar": [4, 5],
    "molar": [6, 7, 8]
}

CLASSES_DENTS = list(FDI_CLASS_MAP.keys())

model_dents = YOLO("models/dents.pt")
model_implants = YOLO("models/implants.pt")
model_bridges = YOLO("models/bridges.pt")


def detect_objects(model, image, keep_only=None):
    results = model(image)
    detections = []
    names = model.names
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = names[class_id]
        if keep_only and class_name not in keep_only:
            continue
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        detections.append({"bbox": [x1, y1, x2, y2], "class_name": class_name, "center": center})
    return detections


def split_into_quadrants(dents, width, height):
    mid_x, mid_y = width / 2, height / 2
    quadrants = {1: [], 2: [], 3: [], 4: []}
    for d in dents:
        x, y = d["center"]
        if y < mid_y:
            quadrant = 1 if x < mid_x else 2
        else:
            quadrant = 4 if x < mid_x else 3
        quadrants[quadrant].append(d)
    return quadrants


def find_missing_molar_case(molars, premolars, fdi_positions, reverse):
    if len(molars) != 2 or len(premolars) != 1 or len(fdi_positions) != 3:
        return None

    premolar = premolars[0]
    P = premolar["center"][0]
    sorted_molars = sorted(molars, key=lambda d: d["center"][0], reverse=reverse)
    M1 = sorted_molars[0]["center"][0]
    M2 = sorted_molars[1]["center"][0]

    D1 = abs(P - M1)
    D2 = abs(M1 - M2)

    if abs(D1 - D2) < 10:
        missing_index = 2
    elif D1 > D2:
        missing_index = 0
    else:
        missing_index = 1

    complete = []
    i_molar = 0
    for i in range(3):
        if i == missing_index:
            complete.append({"FDI": fdi_positions[i], "status": "absente", "class_name": "molar", "center": None})
        else:
            complete.append({
                "FDI": fdi_positions[i],
                "status": "présente",
                "class_name": "molar",
                "center": sorted_molars[i_molar]["center"]
            })
            i_molar += 1
    return complete


def assign_fdi_numbers(quadrants):
    assigned = []
    for q, dents in quadrants.items():
        reverse = q in [1, 4]
        for class_name, digits in FDI_CLASS_MAP.items():
            class_dents = [d for d in dents if d["class_name"] == class_name]
            fdi_positions = [int(f"{q}{digit}") for digit in digits]

            if class_name == "molar" and len(class_dents) == 2:
                premolars = [d for d in dents if d["class_name"] == "premolar"]
                custom = find_missing_molar_case(class_dents, premolars, fdi_positions, reverse)
                if custom:
                    assigned.extend(custom)
                    continue

            sorted_dents = sorted(class_dents, key=lambda d: d["center"][0], reverse=reverse)
            for i, fdi in enumerate(fdi_positions):
                if i < len(sorted_dents):
                    assigned.append({
                        "FDI": fdi,
                        "status": "présente",
                        "class_name": class_name,
                        "center": sorted_dents[i]["center"]
                    })
                else:
                    assigned.append({
                        "FDI": fdi,
                        "status": "absente",
                        "class_name": class_name,
                        "center": None
                    })
    return assigned


def process_cv2_image(img):
    h, w = img.shape[:2]

    # Détection filtrée
    detections_dents = detect_objects(model_dents, img, keep_only=CLASSES_DENTS)
    detections_implants = detect_objects(model_implants, img)
    detections_bridges = detect_objects(model_bridges, img)

    # Attribution FDI
    quadrants = split_into_quadrants(detections_dents, w, h)
    fdi_assigned = assign_fdi_numbers(quadrants)


    return {
        "dents": fdi_assigned,
        "implants": detections_implants,
        "bridges": detections_bridges
    }



if __name__ == "__main__":
    process_image("input.jpg")
