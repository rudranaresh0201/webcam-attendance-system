"""Measure recognition accuracy and suggest a threshold.

    python evaluate.py                           # on the registered students in dataset/
    python evaluate.py --backend dnn
    python evaluate.py --data path/to/lfw --min-images 20 --max-people 10 --max-images 30

--data takes any public dataset laid out as one folder per person.

Two experiments:
  identification  k-fold: train on most images of every person, test on the rest.
                  Is the nearest match the right person?
  impostors       leave one person out of training entirely, then show their
                  faces. Every one of them should come back as Unknown.
"""
import sys
from pathlib import Path

import numpy as np

import config
import utils
from engines import create_engine
from train import load_dataset

MAX_FALSE_ACCEPT = 0.01     # the suggested threshold lets at most this share of strangers through


def run_experiments(engine, descriptors, label_ids, folds):
    """Returns (genuine_scores, genuine_correct, impostor_scores) as arrays."""
    label_ids = np.array(label_ids)
    index_in_person = np.zeros(len(label_ids), dtype=int)
    for label in np.unique(label_ids):
        mine = np.flatnonzero(label_ids == label)
        index_in_person[mine] = np.arange(len(mine))

    def train_on(mask):
        engine.train([descriptors[i] for i in np.flatnonzero(mask)], label_ids[mask].tolist())

    genuine_scores, genuine_correct = [], []
    for fold in range(folds):
        test = index_in_person % folds == fold
        train_on(~test)
        for i in np.flatnonzero(test):
            match = engine.predict(descriptors[i])
            genuine_scores.append(match.score)
            genuine_correct.append(match.label == label_ids[i])

    impostor_scores = []
    for label in np.unique(label_ids):
        held_out = label_ids == label
        train_on(~held_out)
        impostor_scores.extend(engine.predict(descriptors[i]).score for i in np.flatnonzero(held_out))

    return np.array(genuine_scores), np.array(genuine_correct), np.array(impostor_scores)


def rates(engine, threshold, genuine, correct, impostor):
    """(false accept, missed, wrong student) rates at a threshold."""
    accept = (lambda s: s >= threshold) if engine.higher_is_better else (lambda s: s <= threshold)
    false_accept = accept(impostor).mean()              # stranger accepted as a student
    missed = 1 - (accept(genuine) & correct).mean()     # student not credited
    wrong = (accept(genuine) & ~correct).mean()         # student accepted as someone else
    return false_accept, missed, wrong


def evaluate(engine, dirs, max_images, folds):
    descriptors, label_ids, labels = load_dataset(engine, dirs, max_images)
    people = [s for s in labels.values() if s["images"] >= folds]
    if len(people) < 2 or len(people) < len(labels):
        sys.exit(f"Need at least 2 people, each with {folds}+ usable face images.")

    genuine, correct, impostor = run_experiments(engine, descriptors, label_ids, folds)

    if engine.higher_is_better:
        grid, name, fmt = np.arange(0.0, 1.0, 0.005), "SFACE_THRESHOLD", "{:.3f}"
    else:
        grid, name, fmt = np.arange(1.0, max(genuine.max(), impostor.max()) + 1), "LBPH_THRESHOLD", "{:.0f}"
    table = np.array([rates(engine, t, genuine, correct, impostor) for t in grid])
    equal_error = grid[np.argmin(np.abs(table[:, 0] - table[:, 1]))]
    safe = grid[table[:, 0] <= MAX_FALSE_ACCEPT]
    # most lenient threshold that still keeps strangers out
    suggested = (safe.min() if engine.higher_is_better else safe.max()) if len(safe) else None

    skipped = sum(s["skipped"] for s in labels.values())
    print(f"\n[{engine.name}] {len(labels)} people, {len(descriptors)} face images"
          + (f" ({skipped} skipped, no face found)" if skipped else ""))
    print(f"Identification accuracy (nearest match is the right person): {correct.mean():.1%}")
    print(f"{'threshold':<28}{'value':>8}{'false accept':>14}{'missed':>9}{'wrong student':>15}")
    rows = [("current (config.py)", engine.threshold), ("equal error rate", equal_error)]
    if suggested is not None:
        rows.append((f"false accept <= {MAX_FALSE_ACCEPT:.0%}", suggested))
    for title, threshold in rows:
        fa, missed, wrong = rates(engine, threshold, genuine, correct, impostor)
        print(f"{title:<28}{fmt.format(threshold):>8}{fa:>14.1%}{missed:>9.1%}{wrong:>15.1%}")
    if suggested is not None:
        print(f"Suggested: {name} = {fmt.format(suggested)}")
    else:
        print("No threshold keeps strangers out on this data.")


def main():
    parser = utils.backend_parser("Evaluate recognition accuracy.")
    parser.add_argument("--data", type=Path, metavar="DIR",
                        help="dataset with one folder per person (default: the registered students)")
    parser.add_argument("--min-images", type=int, default=config.MIN_IMAGES_TO_TRAIN,
                        help="ignore people with fewer images than this")
    parser.add_argument("--max-people", type=int, help="use only the first N people")
    parser.add_argument("--max-images", type=int, help="use only the first N images per person")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    if args.data:
        if not args.data.is_dir():
            sys.exit(f"{args.data} is not a folder.")
        dirs = sorted(p for p in args.data.iterdir() if p.is_dir())
    else:
        dirs = utils.student_dirs()
    dirs = [d for d in dirs if len(utils.image_files(d)) >= args.min_images][:args.max_people]

    try:
        engine = create_engine(args.backend)
    except RuntimeError as e:
        sys.exit(str(e))
    evaluate(engine, dirs, args.max_images, args.folds)


if __name__ == "__main__":
    main()
