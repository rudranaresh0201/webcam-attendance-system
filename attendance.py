"""Live recognition: mark recognized students present, everyone else absent on quit."""
import json
import sys
from datetime import datetime

import cv2
import pandas as pd

import config
import utils
from engines import FaceEngine
from tracker import Tracker

WINDOW = "Attendance - press q to finish"
COLUMNS = ["roll", "name", "status", "time"]


def load_model(engine):
    if not engine.load() or not config.LABELS_PATH.exists():
        sys.exit("No trained model found. Run train.py first.")
    with open(config.LABELS_PATH) as f:
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

    def __init__(self, engine, labels, path):
        self.engine = engine
        self.labels = labels
        self.path = path
        self.present = load_present(path)
        self.tracker = Tracker()

    def mark_present(self, student):
        now = datetime.now().strftime("%H:%M:%S")
        self.present[student["roll"]] = {"name": student["name"], "time": now}
        save_attendance(self.path, self.labels, self.present)
        print(f"Present: {student['roll']} {student['name']} at {now}")

    def process(self, frame):
        """Detect, track and recognise every face in the frame. Returns the annotated frame."""
        view = frame.copy()
        for track in self.tracker.update(self.engine.detect(frame)):
            match = self.engine.predict(self.engine.describe(frame, track.face))
            track.observe(match)
            box = track.face.box
            student = self.labels.get(track.label)
            if student is None:
                utils.draw_label(view, box, f"Unknown ({match.score:.0f})", utils.RED)
                continue
            if track.confirmed and student["roll"] not in self.present:
                self.mark_present(student)
            if student["roll"] in self.present:
                utils.draw_label(view, box, f"{student['name']} - Marked", utils.GREEN)
            else:
                utils.draw_label(view, box, f"{student['name']} ({match.score:.0f})", utils.YELLOW)
        utils.draw_hud(view, f"Present: {self.count_present()}/{len(self.labels)}")
        return view

    def count_present(self):
        return sum(1 for s in self.labels.values() if s["roll"] in self.present)

    def finish(self):
        save_attendance(self.path, self.labels, self.present)


def main():
    engine = FaceEngine()
    labels = load_model(engine)
    session = AttendanceSession(engine, labels, csv_path_for_today())
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
