import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib
from collections import Counter

def get_dominant_color_lab(image_array, mask, n_colors=3):
    selected_pixels = image_array[mask]
    if len(selected_pixels) < n_colors:
        return None
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(selected_pixels)
    most_common_idx = Counter(kmeans.labels_).most_common(1)[0][0]
    dominant_color = kmeans.cluster_centers_[most_common_idx].astype(int)
    return tuple(dominant_color)

def closest_color_name(requested_color):
    min_dist = float("inf")
    closest_name = None
    requested_color = np.array(requested_color) / 255.0  # Normalize to [0, 1]

    for name, hex_value in matplotlib.colors.CSS4_COLORS.items():
        r, g, b = matplotlib.colors.hex2color(hex_value)
        dist = sum((comp1 - comp2) ** 2 for comp1, comp2 in zip(requested_color, (r, g, b)))
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    return closest_name

def process_image(image_path, n_colors=3):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Simula l'output di un modello di segmentazione
    # Qui devi integrare il modello di segmentazione che stai utilizzando

    # Supponiamo che 'pred_seg' sia un array numpy delle dimensioni dell'immagine
    # che rappresenta le maschere segmentate.
    pred_seg = np.random.randint(0, 3, size=image_np.shape[:2])  # Dummy segmentation map

    # Categorie fittizie per test
    id2label = {0: "Upper-clothes", 1: "Pants", 2: "Bag"}

    data = []
    for label_id, label_name in id2label.items():
        mask = pred_seg == label_id
        if np.any(mask):
            dominant_color = get_dominant_color_lab(image_np, mask, n_colors)
            if dominant_color:
                color_name = closest_color_name(dominant_color)
                data.append({
                    "category": label_name,
                    "color_rgb": dominant_color,
                    "color_name": color_name,
                    "image": Image.fromarray(np.uint8(mask) * 255)  # Maschera come immagine
                })

    df = pd.DataFrame(data)
    return df
