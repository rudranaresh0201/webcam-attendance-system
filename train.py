"""Train a recognizer on everything in dataset/.

    python train.py                  # backend from config.py
    python train.py --backend dnn
"""
import json
import sys

import cv2

import config
import utils
from engines import create_engine


def load_dataset(engine, dirs=None, max_images=None):
    """Run every stored image through the engine's detect + describe pipeline.

    dirs defaults to the registered students in dataset/; evaluate.py passes
    the person folders of a public dataset instead.

    Returns (descriptors, label_ids, labels) where labels maps
    id -> {roll, name, images, skipped}. Images in which the engine finds no
    face are skipped.
    """
    descriptors, label_ids, labels = [], [], {}
    for label_id, student_dir in enumerate(utils.student_dirs() if dirs is None else dirs):
        roll, name = utils.parse_student_dir(student_dir.name)
        used = skipped = 0
        for image_path in utils.image_files(student_dir)[:max_images]:
            image = cv2.imread(str(image_path))
            face = utils.largest_face(engine.detect(image)) if image is not None else None
            if face is None:
                skipped += 1
                continue
            descriptors.append(engine.describe(image, face))
            label_ids.append(label_id)
            used += 1
        labels[label_id] = {"roll": roll, "name": name, "images": used, "skipped": skipped}
    return descriptors, label_ids, labels


def check_dataset(labels):
    if not labels:
        sys.exit("Dataset is empty. Run register.py first.")
    too_few = [s for s in labels.values() if s["images"] < config.MIN_IMAGES_TO_TRAIN]
    if too_few:
        for s in too_few:
            print(f"{s['roll']} {s['name']}: only {s['images']} usable images")
        sys.exit(f"Each student needs at least {config.MIN_IMAGES_TO_TRAIN} usable images. Register them again.")


def main():
    args = utils.backend_parser("Train the face recognizer.").parse_args()
    try:
        engine = create_engine(args.backend)
    except RuntimeError as e:
        sys.exit(str(e))

    descriptors, label_ids, labels = load_dataset(engine)
    check_dataset(labels)
    engine.train(descriptors, label_ids)

    model_dir = config.model_dir(engine.name)
    model_dir.mkdir(parents=True, exist_ok=True)
    engine.save(model_dir)
    with open(model_dir / "labels.json", "w") as f:
        json.dump({i: {"roll": s["roll"], "name": s["name"]} for i, s in labels.items()}, f, indent=2)

    print(f"[{engine.name}] trained on {len(labels)} student(s), {len(descriptors)} images:")
    for s in labels.values():
        note = f", {s['skipped']} skipped (no face found)" if s["skipped"] else ""
        print(f"  {s['roll']}  {s['name']}  ({s['images']} images{note})")
    if len(labels) == 1:
        print("Warning: only one student registered. The recognizer always returns the nearest "
              "student, so with one student only the threshold separates them from strangers.")


if __name__ == "__main__":
    main()
