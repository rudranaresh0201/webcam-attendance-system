"""Live recognition: mark recognised students present, everyone else absent on quit.

    python attendance.py          # OpenCV window, press q to finish
    python app.py                 # the same session inside the web dashboard
"""
import os
import sys
import threading
from datetime import datetime

import cv2
import pandas as pd

import config
import utils
from engines import FaceEngine
from recognizer import load_labels
from tracker import Tracker

WINDOW = "Attendance - press q to finish"
COLUMNS = ["roll", "name", "status", "time"]


def load_model(engine):
    labels = load_labels()
    if not engine.load() or not labels:
        sys.exit("No trained model found (or settings changed). Run train.py first.")
    return labels


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
    # write a temporary file, then swap it in: a crash mid-write never leaves half a register
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows, columns=COLUMNS).to_csv(tmp, index=False)
    os.replace(tmp, path)


def attendance_report(records_dir=None, labels=None):
    """Every daily register in records/ combined: per student, days present out of days
    held, percentage, and whether they are below config.MIN_ATTENDANCE_PCT."""
    records_dir = records_dir or config.RECORDS_DIR
    files = sorted(records_dir.glob("attendance_*.csv")) if records_dir.exists() else []
    dates, students = [], {}
    for s in (labels or {}).values():
        students[s["roll"]] = {"roll": s["roll"], "name": s["name"], "days": {}}
    for f in files:
        date = f.stem.replace("attendance_", "")
        dates.append(date)
        for row in pd.read_csv(f, dtype=str, keep_default_na=False).to_dict("records"):
            st = students.setdefault(row["roll"], {"roll": row["roll"], "name": row["name"], "days": {}})
            st["days"][date] = row["status"]
    out = []
    for st in students.values():
        held = [d for d in dates if d in st["days"]]          # days this student was on the register
        present = sum(st["days"][d] == "Present" for d in held)
        pct = round(100 * present / len(held), 1) if held else None
        out.append({**st, "present": present, "held": len(held), "pct": pct,
                    "low": pct is not None and pct < config.MIN_ATTENDANCE_PCT})
    out.sort(key=lambda r: (r["pct"] is None, r["pct"] if r["pct"] is not None else 0, r["roll"]))
    return {"dates": dates, "students": out, "min_pct": config.MIN_ATTENDANCE_PCT}


class AttendanceSession:
    """Per-frame attendance logic, kept separate from the camera loop so the CLI,
    the web app and the tests can all drive it."""

    def __init__(self, engine, labels, path):
        self.engine = engine
        self.labels = labels
        self.path = path
        self.present = load_present(path)
        self.tracker = Tracker()
        self.events = []            # (time, roll, name) for the UI feed
        self.unknown_tracks = set()
        self.save_error = None      # set while the CSV can't be written (e.g. open in Excel)
        self._lock = threading.Lock()

    def _save(self):
        """Write the register; on failure keep everything in memory and retry next frame."""
        try:
            save_attendance(self.path, self.labels, self.present)
            self.save_error = None
        except OSError as e:
            self.save_error = (f"Couldn't write {self.path.name} ({e.__class__.__name__}). "
                               "If it's open in Excel, close it; nothing is lost, it will be saved automatically.")

    def mark_present(self, student):
        now = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self.present[student["roll"]] = {"name": student["name"], "time": now}
            self.events.append({"time": now, "roll": student["roll"], "name": student["name"]})
            self._save()
        print(f"Present: {student['roll']} {student['name']} at {now}")

    def step(self, frame):
        """Detect, track, recognise. Returns a list of per-face result dicts."""
        results = []
        if self.save_error:
            with self._lock:
                self._save()
        gray = self.engine.gray(frame)
        for track in self.tracker.update(self.engine.detect(frame)):
            chip = self.engine.describe(frame, track.face)
            match = self.engine.predict(chip)
            track.observe(match)
            if config.LIVENESS:
                track.observe_eyes(self.engine.detector.count_eyes(gray, track.face.box))
            student = self.labels.get(track.label)      # leader of the last-8-frames vote
            state = "unknown"
            if student is not None:
                if track.confirmed and track.live and student["roll"] not in self.present:
                    self.mark_present(student)
                if student["roll"] in self.present:
                    state = "marked"
                elif track.confirmed and not track.live:
                    state = "blink"
                else:
                    state = "verifying"
            elif track.age >= config.CONFIRM_FRAMES:
                self.unknown_tracks.add(track.id)
            results.append({
                "track": track, "chip": chip, "match": match, "student": student, "state": state,
                "box": track.face.box, "sources": track.face.sources,
            })
        return results

    def draw(self, frame, results):
        view = frame.copy()
        for r in results:
            m, s = r["match"], r["student"]
            if r["state"] == "unknown":
                utils.draw_label(view, r["box"], f"Unknown ({m.score:.2f})", utils.RED)
            elif r["state"] == "marked":
                utils.draw_label(view, r["box"], f"{s['name']} - Marked", utils.GREEN)
            elif r["state"] == "blink":
                utils.draw_label(view, r["box"], f"{s['name']} - blink to confirm", utils.YELLOW)
            else:
                utils.draw_label(view, r["box"], f"{s['name']} {r['track'].streak}/{config.CONFIRM_FRAMES}",
                                 utils.YELLOW)
        utils.draw_hud(view, f"Present: {self.count_present()}/{len(self.labels)}")
        return view

    def process(self, frame):
        return self.draw(frame, self.step(frame))

    def count_present(self):
        return sum(1 for s in self.labels.values() if s["roll"] in self.present)

    def finish(self):
        """Final save. Returns False (and keeps save_error) if the file couldn't be written."""
        with self._lock:
            self._save()
        return self.save_error is None


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

    if not session.finish():
        print(session.save_error)
    print(f"\nPresent {session.count_present()}/{len(labels)}. Saved to {session.path}")


if __name__ == "__main__":
    main()
