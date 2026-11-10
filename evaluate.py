"""Measure recognition accuracy and suggest a threshold.

    python evaluate.py                           # on the registered students in dataset/
    python evaluate.py --data path/to/lfw --min-images 20 --max-people 10 --max-images 30

--data takes any public dataset laid out as one folder per person.

Two experiments:
  identification  k-fold: train on most images of every person, test on the rest.
                  Is the nearest match the right person?
  impostors       leave one person out of training entirely, then show their
                  faces. Every one of them should come back as Unknown.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

import config
import utils
from engines import FaceEngine
from train import load_dataset

MAX_FALSE_ACCEPT = 0.01     # the suggested threshold lets at most this share of strangers through


def run_experiments(engine, descriptors, label_ids, folds):
    """Returns a dict of genuine / impostor distances, ratios and correctness arrays."""
    label_ids = np.array(label_ids)
    index_in_person = np.zeros(len(label_ids), dtype=int)
    for label in np.unique(label_ids):
        mine = np.flatnonzero(label_ids == label)
        index_in_person[mine] = np.arange(len(mine))

    def train_on(mask):
        engine.train([descriptors[i] for i in np.flatnonzero(mask)], label_ids[mask].tolist())

    gd, gr, gc = [], [], []
    for fold in range(folds):
        test = index_in_person % folds == fold
        train_on(~test)
        for i in np.flatnonzero(test):
            m = engine.predict(descriptors[i])
            gd.append(m.score)
            gr.append(m.ratio)
            gc.append(m.label == label_ids[i])

    idd, ir = [], []
    for label in np.unique(label_ids):
        held_out = label_ids == label
        train_on(~held_out)
        for i in np.flatnonzero(held_out):
            m = engine.predict(descriptors[i])
            idd.append(m.score)
            ir.append(m.ratio)
    a = np.array
    return dict(g_d=a(gd), g_r=a(gr), g_correct=a(gc, bool), i_d=a(idd), i_r=a(ir))


def rates(s, t, r):
    """(false accept, missed, wrong student) at distance threshold t and ratio threshold r."""
    fa = ((s["i_d"] <= t) & (s["i_r"] <= r)).mean()
    acc = (s["g_d"] <= t) & (s["g_r"] <= r)
    return fa, 1 - (acc & s["g_correct"]).mean(), (acc & ~s["g_correct"]).mean()


def evaluate(engine, dirs, max_images, folds):
    from benchmark import frr_at_far
    descriptors, label_ids, labels = load_dataset(engine, dirs, max_images)
    people = [s for s in labels.values() if s["images"] >= folds]
    if len(people) < 3 or len(people) < len(labels):
        sys.exit(f"Need at least 3 people, each with {folds}+ usable face images "
                 "(the impostor test removes one person and still needs two to compare).")

    s = run_experiments(engine, descriptors, label_ids, folds)
    openset = engine.recognizer.kind == "openset"
    skipped = sum(v["skipped"] for v in labels.values())
    print(f"\n{len(labels)} people, {len(descriptors)} face images"
          + (f" ({skipped} skipped, no face found)" if skipped else ""))
    print(f"Identification accuracy (nearest match is the right person): {s['g_correct'].mean():.1%}")
    print(f"{'rule':<34}{'dist':>7}{'ratio':>7}{'false accept':>14}{'missed':>9}{'wrong':>8}")
    cur_r = engine.recognizer.ratio_threshold if openset else 9.0
    rows = [("current (config.py)", engine.threshold, cur_r)]
    best = frr_at_far(s, "distance+ratio" if openset else "distance", MAX_FALSE_ACCEPT)
    if best[1] is not None:
        rows.append((f"best with false accept <= {MAX_FALSE_ACCEPT:.0%}", *best[1]))
    for title, t, r in rows:
        fa, missed, wrong = rates(s, t, r)
        print(f"{title:<34}{t:>7.3f}{min(r, 9):>7.2f}{fa:>14.1%}{missed:>9.1%}{wrong:>8.1%}")
    if best[1] is None:
        print("No threshold keeps strangers out on this data.")
        return None
    t, r = best[1]
    if openset:
        print(f"Suggested: DIST_THRESHOLD = {t:.3f}, RATIO_THRESHOLD = {r:.2f}")
    else:
        print(f"Suggested: LBPH_THRESHOLD = {t:.0f}")
    return {"accuracy": float(s["g_correct"].mean()), "dist": t, "ratio": r,
            "far": float(rates(s, t, r)[0]), "frr": float(best[0])}


def main():
    parser = argparse.ArgumentParser(description="Evaluate recognition accuracy.")
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

    evaluate(FaceEngine(), dirs, args.max_images, args.folds)


if __name__ == "__main__":
    main()
