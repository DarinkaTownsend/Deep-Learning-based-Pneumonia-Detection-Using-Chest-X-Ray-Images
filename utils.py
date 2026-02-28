import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from collections import Counter


# -----------------------------
# Visualization
# -----------------------------

def show_images(folder_path, class_name, n=4):
    class_path = os.path.join(folder_path, class_name)
    images = os.listdir(class_path)[:n]

    plt.figure(figsize=(12, 4))
    for i, img_name in enumerate(images):
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path)

        plt.subplot(1, n, i+1)
        plt.imshow(img, cmap="gray")
        plt.title(class_name)
        plt.axis("off")
    plt.show()


# -----------------------------
# Dataset statistics
# -----------------------------

def count_images(folder):
    total = 0
    for cls in os.listdir(folder):
        cls_path = os.path.join(folder, cls)
        if os.path.isdir(cls_path):
            total += len(os.listdir(cls_path))
    return total


def class_counts(split_dir):
    counts = {}
    for cls in sorted(os.listdir(split_dir)):
        cls_path = os.path.join(split_dir, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len([f for f in os.listdir(cls_path) if not f.startswith(".")])
    return counts


def print_percentages(counts, name="SPLIT"):
    total = sum(counts.values())
    print(f"\n{name} total = {total}")
    for cls, c in counts.items():
        pct = (c / total) * 100
        print(f"{cls:10s}: {c:5d} ({pct:5.2f}%)")


def barplot_counts(counts, title):
    classes = list(counts.keys())
    values  = [counts[c] for c in classes]

    plt.figure(figsize=(5, 3))
    plt.bar(classes, values)
    plt.title(title)
    plt.ylabel("Num images")
    plt.show()

# -----------------------------
# Training
# -----------------------------

def compute_class_weights(dataset, device):
    labels = [label for _, label in dataset]
    counts = Counter(labels)
    num_classes = len(dataset.classes)
    total = len(dataset)
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print("Class distribution:")
    for cls_name, idx in dataset.class_to_idx.items():
        print(f"{cls_name}: count={counts[idx]}, weight={class_weights[idx].item():.4f}")
    return class_weights