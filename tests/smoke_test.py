"""End-to-end smoke test without a webcam, on LFW photos.

    python tests/smoke_test.py --lfw ~/datasets/lfw_home/lfw_funneled

Enrols 4 people from their first 25 photos into a temporary dataset, trains, then
drives AttendanceSession frame by frame with photos the model has never seen:
  * a held-out photo of an enrolled person, repeated CONFIRM_FRAMES times -> credited Present
  * photos of people who are not enrolled -> never credited
  * the CSV has one row per student with Absent for the others
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2

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

    def run(session, path, frames=config.CONFIRM_FRAMES + 2):
        frame = utils.prepare_photo(cv2.imread(str(path)))
        session.tracker = type(session.tracker)()        # fresh track per probe photo
        for _ in range(frames):
            session.step(frame)

    session = AttendanceSession(engine, labels, config.RECORDS_DIR / "attendance_test.csv")
    credited = 0
    for i, person in enumerate(ENROL):
        held_out = sorted((args.lfw / person).glob("*.jpg"))[25:45]
        for f in held_out:
            run(session, f)
            if f"S{i:02d}" in session.present:
                credited += 1
                print(f"  {person}: credited from held-out {f.name}")
                break
        else:
            print(f"  {person}: not credited from 20 held-out photos")
    false_credits = 0
    for person in STRANGERS:
        before = set(session.present)
        for f in sorted((args.lfw / person).glob("*.jpg"))[:10]:
            run(session, f)
        false_credits += len(set(session.present) - before)
    session.finish()
    rows = (config.RECORDS_DIR / "attendance_test.csv").read_text().strip().splitlines()
    print(f"credited {credited}/{len(ENROL)} enrolled; {false_credits} strangers credited (from 50 photos)")
    print("\n".join(rows))
    assert false_credits == 0, "a stranger was marked present"
    assert credited >= 3, "enrolled people were not recognised"
    assert len(rows) == len(ENROL) + 1
    shutil.rmtree(tmp)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
