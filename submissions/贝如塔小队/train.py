import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def features(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    lab_proxy = np.stack((rgb.mean(axis=2), rgb[:, :, 0] - rgb[:, :, 1],
                          rgb[:, :, 0] - rgb[:, :, 2]), axis=2)
    luminance = lab_proxy[:, :, 0]
    dy, dx = np.gradient(luminance)
    magnitude = np.hypot(dx, dy)
    angle = (np.arctan2(dy, dx) % np.pi) * (8 / np.pi)
    orientation = np.floor(angle).astype(int) % 8
    values = []
    for size in (1, 2, 4):
        step = 28 // size
        for row in range(size):
            for col in range(size):
                region = np.s_[row*step:(row+1)*step, col*step:(col+1)*step]
                pixels = lab_proxy[region].reshape(-1, 3)
                values.extend(pixels.mean(axis=0))
                values.extend(pixels.std(axis=0))
                histogram = np.bincount(orientation[region].ravel(),
                                        weights=magnitude[region].ravel(), minlength=8)
                values.extend(np.sqrt(histogram / (histogram.sum() + 1e-8)))
    for channel in range(3):
        limits = (0, 1) if channel == 0 else (-1, 1)
        hist, _ = np.histogram(lab_proxy[:, :, channel], bins=16, range=limits)
        values.extend(np.sqrt(hist / 784))
    return values


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def load_split(root, split):
    folder = root / split
    rows = read_rows(folder / "labels.csv")
    images = np.asarray([features(folder / row["filename"]) for row in rows])
    labels = np.asarray([int(row["label"]) for row in rows])
    return images, labels, [Path(row["filename"]).stem for row in rows]


def write_predictions(path, ids, labels):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["id", "label"])
        writer.writerows(zip(ids, labels, strict=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Extracted DermaMNIST directory")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--val-output", type=Path)
    args = parser.parse_args()

    x_train, y_train, _ = load_split(args.data, "train")
    x_val, y_val, val_ids = load_split(args.data, "val")
    print("Train class counts:", np.bincount(y_train), flush=True)
    scaler = StandardScaler()
    x_train = np.clip(scaler.fit_transform(x_train), -5, 5)
    x_val = np.clip(scaler.transform(x_val), -5, 5)
    frequencies = np.bincount(y_train)
    best_score = -1
    for strength in (0, 0.5, 1):
        weights = dict(enumerate((len(y_train) / (7 * frequencies)) ** strength))
        model = SVC(C=4, class_weight=weights, cache_size=256)
        model.fit(x_train, y_train)
        prediction = model.predict(x_val)
        macro_f1 = f1_score(y_val, prediction, average="macro")
        print(f"Weight exponent {strength}: accuracy={accuracy_score(y_val, prediction):.4f}, "
              f"balanced accuracy={balanced_accuracy_score(y_val, prediction):.4f}, "
              f"macro-F1={macro_f1:.4f}", flush=True)
        if macro_f1 > best_score:
            best_score, best_model, best_prediction = macro_f1, model, prediction
    if args.val_output:
        write_predictions(args.val_output, val_ids, best_prediction)
    test_ids = [row["id"] for row in read_rows(args.data / "sample_submission.csv")]
    x_test = np.asarray([features(args.data / "test" / f"{image_id}.png") for image_id in test_ids])
    write_predictions(args.output, test_ids, best_model.predict(np.clip(scaler.transform(x_test), -5, 5)))


if __name__ == "__main__":
    main()
