"""Register a student: save face images into dataset/<roll>_<name>/.

    python register.py                          # capture from the webcam
    python register.py --from-folder photos/    # import existing photos instead
"""
import re
import shutil
import sys
from pathlib import Path

import cv2

import config
import utils
from engines import create_engine

WINDOW = "Register - press q to abort"


def ask_student(roll, name):
    roll = (roll or input("Roll number: ")).strip()
    if not re.fullmatch(r"[A-Za-z0-9-]+", roll):
        sys.exit("Roll number may only contain letters, digits and '-'.")
    name = " ".join((name or input("Name: ")).split())
    if not re.fullmatch(r"[A-Za-z .-]+", name):
        sys.exit("Name may only contain letters, spaces, '.' and '-'.")
    return roll, name


def prepare_dir(roll, name, overwrite):
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(config.DATASET_DIR.glob(f"{roll}_*"))
    if existing:
        if not overwrite:
            answer = input(f"Roll {roll} is already registered ({existing[0].name}). Overwrite? [y/n]: ")
            if answer.strip().lower() != "y":
                sys.exit("Cancelled.")
        for path in existing:
            shutil.rmtree(path)
    out_dir = config.DATASET_DIR / f"{roll}_{name}"
    out_dir.mkdir()
    return out_dir


def capture(engine, out_dir):
    cap = utils.open_camera()
    saved = 0
    frame_idx = 0
    try:
        while saved < config.IMAGES_PER_STUDENT:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break
            frame = utils.prepare_frame(frame)
            face = utils.largest_face(engine.detect(frame))
            frame_idx += 1
            view = frame.copy()     # draw on a copy so overlays never end up in saved images

            if face is None:
                utils.draw_hud(view, "No face detected", line=1, color=utils.RED)
            else:
                problem = utils.quality_problem(frame, face.box)
                if problem:
                    utils.draw_hud(view, problem, line=1, color=utils.RED)
                elif frame_idx % config.SAVE_EVERY_N_FRAMES == 0:
                    cv2.imwrite(str(out_dir / f"{saved}.png"), utils.padded_crop(frame, face.box))
                    saved += 1
                color = utils.RED if problem else utils.GREEN
                utils.draw_label(view, face.box, f"{saved}/{config.IMAGES_PER_STUDENT}", color)

            utils.draw_hud(view, "Turn head slightly, vary expression")
            cv2.imshow(WINDOW, view)
            if utils.pressed_key(WINDOW) == "q":
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return saved


def import_folder(engine, folder, out_dir):
    saved = 0
    for path in utils.image_files(folder):
        image = cv2.imread(str(path))
        if image is None:
            print(f"  skip {path.name}: not a readable image")
            continue
        image = utils.prepare_photo(image)
        face = utils.largest_face(engine.detect(image))
        if face is None:
            print(f"  skip {path.name}: no face detected")
            continue
        cv2.imwrite(str(out_dir / f"{saved}.png"), utils.padded_crop(image, face.box))
        saved += 1
    return saved


def main():
    parser = utils.backend_parser("Register a student's face.")
    parser.add_argument("--roll", help="roll number (asked for if omitted)")
    parser.add_argument("--name", help="student name (asked for if omitted)")
    parser.add_argument("--from-folder", type=Path, metavar="DIR",
                        help="import photos of the student from DIR instead of using the webcam")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing registration without asking")
    args = parser.parse_args()

    if args.from_folder and not args.from_folder.is_dir():
        sys.exit(f"{args.from_folder} is not a folder.")
    try:
        engine = create_engine(args.backend)
    except RuntimeError as e:
        sys.exit(str(e))

    roll, name = ask_student(args.roll, args.name)
    out_dir = prepare_dir(roll, name, args.overwrite)
    if args.from_folder:
        saved = import_folder(engine, args.from_folder, out_dir)
        print(f"Saved {saved} images to {out_dir}")
    else:
        saved = capture(engine, out_dir)
        print(f"Saved {saved}/{config.IMAGES_PER_STUDENT} images to {out_dir}")

    if saved < config.MIN_IMAGES_TO_TRAIN:
        print(f"Fewer than {config.MIN_IMAGES_TO_TRAIN} images - register again before training.")
    else:
        print("Now run: python train.py")


if __name__ == "__main__":
    main()
