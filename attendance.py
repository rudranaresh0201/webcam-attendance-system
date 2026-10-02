"""Live recognition: mark recognized students present, everyone else absent on quit.

    python attendance.py                  # backend from config.py
    python attendance.py --backend dnn    # YuNet + SFace, with head-turn liveness check
"""
import json
import sys
import time
from datetime import datetime

import cv2
import pandas as pd

import config
import utils
from engines import create_engine
from liveness import Challenge, yaw_ratio
from tracker import Tracker

WINDOW = "Attendance - press q to finish"
COLUMNS = ["roll", "name", "status", "time"]


def load_model(engine):
    model_dir = config.model_dir(engine.name)
    labels_path = model_dir / "labels.json"
    if not engine.load(model_dir) or not labels_path.exists():
        sys.exit(f"No trained {engine.name} model found. Run: python train.py --backend {engine.name}")
    with open(labels_path) as f:
        return {int(i): s for i, s in json.load(f).items()}


def csv_path_for_today():
    return config.RECORDS_DIR / f"attendance_{datetime.now():%Y-%m-%d}.csv"


def load_present(path):
    """Students already marked Present in today's CSV: roll -> {name, time}."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {
        row["roll"]: {"name": row["name"], "time": row["time"]}
        for row in df.to_dict("records")
        if row["status"] == "Present"
    }


def save_attendance(path, labels, present):
    """Rewrite the whole CSV: one row per student, keyed by roll, so re-runs never duplicate."""
    students = {s["roll"]: s["name"] for s in labels.values()}
    # keep anyone marked earlier today even if they are no longer in the model
    students.update({roll: p["name"] for roll, p in present.items() if roll not in students})
    rows = [
        {
            "roll": roll,
            "name": name,
            "status": "Present" if roll in present else "Absent",
            "time": present[roll]["time"] if roll in present else "",
        }
        for roll, name in sorted(students.items())
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


class AttendanceSession:
    """Per-frame attendance logic, kept separate from the camera loop so it can be tested."""

    def __init__(self, engine, labels, path, liveness=False):
        self.engine = engine
        self.labels = labels
        self.path = path
        self.liveness = liveness
        self.present = load_present(path)
        self.tracker = Tracker()

    def mark_present(self, student):
        now = datetime.now().strftime("%H:%M:%S")
        self.present[student["roll"]] = {"name": student["name"], "time": now}
        save_attendance(self.path, self.labels, self.present)
        print(f"Present: {student['roll']} {student['name']} at {now}")

    def process(self, frame, now=None):
        """Detect, track and recognise every face in the frame. Returns the annotated frame."""
        now = time.monotonic() if now is None else now
        view = frame.copy()
        for track in self.tracker.update(self.engine.detect(frame)):
            match = self.engine.predict(self.engine.describe(frame, track.face))
            self._update_track(view, track, match, now)
        utils.draw_hud(view, f"Present: {self.count_present()}/{len(self.labels)}")
        return view

    def _update_track(self, view, track, match, now):
        box = track.face.box
        score = f"{match.score:.2f}" if self.engine.higher_is_better else f"{match.score:.0f}"

        if track.challenge:
            self._update_challenge(view, track, match, now)
            return
        if now < track.blocked_until:
            utils.draw_label(view, box, "Liveness failed", utils.RED)
            return

        track.observe(match)
        student = self.labels.get(track.label)
        if student is None:
            utils.draw_label(view, box, f"Unknown ({score})", utils.RED)
        elif student["roll"] in self.present:
            utils.draw_label(view, box, f"{student['name']} - Marked", utils.GREEN)
        elif not track.confirmed:
            utils.draw_label(view, box, f"{student['name']} ({score})", utils.YELLOW)
        elif not self.liveness:
            self.mark_present(student)
            utils.draw_label(view, box, f"{student['name']} - Marked", utils.GREEN)
        elif abs(yaw_ratio(track.face.landmarks)) < config.LIVENESS_FRONTAL:
            track.challenge = Challenge(now)
            utils.draw_label(view, box, track.challenge.prompt, utils.YELLOW)
        else:
            # the head turn is measured from a frontal pose, so wait for one
            utils.draw_label(view, box, f"{student['name']} - look at the camera", utils.YELLOW)

    def _update_challenge(self, view, track, match, now):
        box = track.face.box
        student = self.labels[track.label]
        identity_ok = match.ok and match.label == track.label
        result = track.challenge.update(yaw_ratio(track.face.landmarks), identity_ok, now)
        if result == "passed":
            track.challenge = None
            self.mark_present(student)
            utils.draw_label(view, box, f"{student['name']} - Marked", utils.GREEN)
        elif result == "failed":
            track.challenge = None
            track.label, track.streak = None, 0
            track.blocked_until = now + config.LIVENESS_COOLDOWN
            utils.draw_label(view, box, "Liveness failed", utils.RED)
        else:
            utils.draw_label(view, box, f"{student['name']}: {track.challenge.prompt}", utils.YELLOW)

    def count_present(self):
        return sum(1 for s in self.labels.values() if s["roll"] in self.present)

    def finish(self):
        save_attendance(self.path, self.labels, self.present)


def main():
    parser = utils.backend_parser("Take attendance from the webcam.")
    parser.add_argument("--no-liveness", action="store_true", help="skip the head-turn liveness check (dnn)")
    args = parser.parse_args()
    try:
        engine = create_engine(args.backend)
    except RuntimeError as e:
        sys.exit(str(e))
    labels = load_model(engine)

    liveness = config.LIVENESS and not args.no_liveness
    if liveness and engine.name != "dnn":
        print("Note: the liveness check needs facial landmarks and only runs with --backend dnn.")
        liveness = False

    session = AttendanceSession(engine, labels, csv_path_for_today(), liveness)
    cap = utils.open_camera()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break
            cv2.imshow(WINDOW, session.process(utils.prepare_frame(frame)))
            if utils.pressed_key(WINDOW) == "q":
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    session.finish()
    print(f"\nPresent {session.count_present()}/{len(labels)}. Saved to {session.path}")


if __name__ == "__main__":
    main()
