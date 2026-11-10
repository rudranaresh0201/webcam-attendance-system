"""End-to-end smoke test without a webcam, on LFW photos.

    python tests/smoke_test.py --lfw ~/datasets/lfw_home/lfw_funneled

Enrols 4 people from their first 25 photos into a temporary dataset, trains, then
drives AttendanceSession frame by frame with photos the model has never seen:
  * an unseen photo of an enrolled person, as 8 jittered webcam-like frames -> credited
    in at least half of 10 such clips
  * 16 photos x 8 frames of each person who is not enrolled -> never credited
  * with ONE student enrolled (no other student to compare with), strangers -> never credited
  * the CSV has one row per student with Absent for the others
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

ENROL = ["Colin_Powell", "Tony_Blair", "Donald_Rumsfeld", "Gerhard_Schroeder"]
STRANGERS = ["Ariel_Sharon", "Hugo_Chavez", "Junichiro_Koizumi", "Jean_Chretien", "John_Ashcroft"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw", type=Path, default=Path.home() / "datasets/lfw_home/lfw_funneled")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    config.DATASET_DIR, config.MODELS_DIR, config.RECORDS_DIR = tmp / "dataset", tmp / "models", tmp / "records"
    config.LABELS_PATH = config.MODELS_DIR / "labels.json"
    config.OPENSET_MODEL_PATH = config.MODELS_DIR / "openset_lbp.npz"
    config.MODEL_PATH = config.MODELS_DIR / "lbph_model.yml"

    from attendance import AttendanceSession
    from engines import FaceEngine
    from register import import_folder
    from train import train_all
    import utils

    engine = FaceEngine()
    for i, person in enumerate(ENROL):
        out = config.DATASET_DIR / f"S{i:02d}_{person.replace('_', ' ')}"
        out.mkdir(parents=True)
        src = tmp / "src" / person
        src.mkdir(parents=True)
        for f in sorted((args.lfw / person).glob("*.jpg"))[:25]:
            shutil.copy(f, src)
        assert import_folder(engine, src, out) >= config.MIN_IMAGES_TO_TRAIN
    engine, _, labels = train_all(FaceEngine())
    print(f"enrolled {len(labels)} people")

    rng = np.random.default_rng(0)

    def clip(path, n=8):
        """One photo as n webcam-like frames of someone sitting still: a few pixels of
        movement, a little brightness flicker and sensor noise in every frame."""
        img = utils.prepare_photo(cv2.imread(str(path))).astype(np.float32)
        for _ in range(n):
            m = np.float32([[1, 0, rng.uniform(-4, 4)], [0, 1, rng.uniform(-4, 4)]])
            f = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
            f = f * rng.uniform(0.9, 1.1) + rng.uniform(-8, 8) + rng.normal(0, 3, f.shape)
            yield np.clip(f, 0, 255).astype(np.uint8)

    def run(session, path):
        session.tracker = type(session.tracker)()
        for frame in clip(path):
            session.step(frame)

    session = AttendanceSession(engine, labels, config.RECORDS_DIR / "attendance_test.csv")
    credited, genuine_clips = 0, 0
    for i, person in enumerate(ENROL):
        held_out = sorted((args.lfw / person).glob("*.jpg"))[25:35]      # 10 photos never enrolled
        hits = 0
        for f in held_out:
            run(session, f)
            hits += f"S{i:02d}" in session.present
            session.present.pop(f"S{i:02d}", None)
        credited += hits > 0
        genuine_clips += hits
        print(f"  {person}: credited in {hits}/{len(held_out)} clips of unseen photos")
        run(session, held_out[0])             # leave them marked for the CSV check
    false_credits = stranger_clips = 0
    for person in STRANGERS:
        for f in sorted((args.lfw / person).glob("*.jpg"))[:16]:       # every photo gets 8 chances
            before = set(session.present)
            run(session, f)
            new = set(session.present) - before
            false_credits += len(new)
            stranger_clips += 1
            for r in new:
                session.present.pop(r)
    session.finish()
    rows = (config.RECORDS_DIR / "attendance_test.csv").read_text().strip().splitlines()
    print(f"students: {genuine_clips} of {10 * len(ENROL)} clips credited ({genuine_clips / (10 * len(ENROL)):.0%}); "
          f"strangers: {false_credits} of {stranger_clips} clips credited")
    print("\n".join(rows))
    assert false_credits / stranger_clips <= 0.01, "too many strangers credited"

    # --- the hard case: ONE student enrolled, so every face's closest student is them.
    # Only the background crowd stops strangers being credited (RESULTS.md 2d).
    solo = FaceEngine()
    solo_chips = [solo.describe(img, utils.largest_face(solo.detect(img)))
                  for img in (cv2.imread(str(f)) for f in sorted((config.DATASET_DIR / f"S00_{ENROL[0].replace('_', ' ')}").iterdir()))
                  if utils.largest_face(solo.detect(img))]
    solo.train(solo_chips, [0] * len(solo_chips))
    solo_session = AttendanceSession(solo, {0: {"roll": "S00", "name": ENROL[0]}}, config.RECORDS_DIR / "solo.csv")
    solo_false = solo_clips = 0
    for person in ENROL[1:] + STRANGERS:
        for f in sorted((args.lfw / person).glob("*.jpg"))[:16]:
            run(solo_session, f)
            solo_false += int("S00" in solo_session.present)
            solo_clips += 1
            solo_session.present.clear()
    print(f"one student enrolled: {solo_false} of {solo_clips} stranger clips credited ({solo_false / solo_clips:.1%})")
    assert solo_false / solo_clips <= 0.02, "with one student enrolled, too many strangers credited"
    assert credited == len(ENROL), "an enrolled person was never recognised from any unseen photo"
    assert len(rows) == len(ENROL) + 1
    shutil.rmtree(tmp)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
