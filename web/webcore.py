"""The in-browser FaceRoll core, run by Pyodide inside a Web Worker (web/worker.js).

It drives the SAME modules as the desktop app (engines, detector, lbp, recognizer,
tracker, utils, config), so the website and `python app.py` make the same decisions.
Only the I/O differs: frames arrive as RGBA bytes from the page, and enrolments and
registers are kept by the page in the visitor's own browser storage.
"""
import time

import cv2
import numpy as np

import config
import lbp
import utils
from engines import FaceEngine
from recognizer import CROWD
from tracker import Tracker

CHIP = config.FACE_SIZE[0] * config.FACE_SIZE[1]


def _bytes(x):
    """JS typed arrays arrive as JsProxy objects; Python bytes pass through."""
    return x.to_bytes() if hasattr(x, "to_bytes") else bytes(x)
POSES = ["Look straight at the camera", "Turn your head slightly left", "Turn slightly right",
         "Tilt your chin up a little", "Smile", "Neutral face"]


class Core:
    def __init__(self):
        self.engine = FaceEngine()
        self.tracker = Tracker()
        self.students = {}          # id -> {"roll", "name"}
        self.chips = {}             # id -> list of 200x200 uint8 chips
        self.present = {}           # roll -> "HH:MM:SS"
        self.events = []
        self.unknown = set()
        self.mode = "preview"       # preview | attendance | enrol
        self.view = "camera"        # camera | contrast | texture | votes
        self.enrol = None
        self.trained = False

    # ------------------------------------------------------------------ data from the page
    def load_students(self, items):
        """items: [{id, roll, name, chips: bytes}] as stored by the page."""
        self.students, self.chips = {}, {}
        for it in items:
            raw = np.frombuffer(_bytes(it["chips"]), np.uint8)
            self.students[int(it["id"])] = {"roll": it["roll"], "name": it["name"]}
            self.chips[int(it["id"])] = list(raw.reshape(-1, *config.FACE_SIZE))
        self.train()

    def train(self):
        chips, ids = [], []
        for sid, cs in self.chips.items():
            chips += cs
            ids += [sid] * len(cs)
        if chips:
            self.engine.train(chips, ids)
        self.trained = bool(chips)
        self.tracker = Tracker()

    def set_present(self, present):
        self.present = dict(present)

    def configure(self, values):
        for k, v in values.items():
            if hasattr(config, k):
                setattr(config, k, type(getattr(config, k))(v))
        self.engine.recognizer.dist_threshold = config.DIST_THRESHOLD
        self.engine.recognizer.ratio_threshold = config.RATIO_THRESHOLD
        self.engine.detector.mode = config.DETECTOR

    # ------------------------------------------------------------------ enrolment
    def start_enrol(self, sid, roll, name):
        self.enrol = {"id": int(sid), "roll": roll, "name": name, "chips": [], "frame": 0,
                      "hint": "Look at the camera", "thumb": None}
        self.mode = "enrol"

    def _enrol_step(self, frame):
        e = self.enrol
        e["frame"] += 1
        face = utils.largest_face(self.engine.detect(frame))
        box, ok = None, False
        if face is None:
            e["hint"] = "No face found. Look at the camera, with light on your face."
        else:
            box = face.box
            problem = utils.quality_problem(frame, face.box)
            if problem:
                e["hint"] = problem
            else:
                ok = True
                e["hint"] = POSES[len(e["chips"]) * len(POSES) // config.IMAGES_PER_STUDENT]
                if e["frame"] % config.SAVE_EVERY_N_FRAMES == 0:
                    e["chips"].append(self.engine.describe(frame, face))
                    if e["thumb"] is None:
                        x, y, w, h = face.box
                        e["thumb"] = cv2.resize(frame[y:y + h, x:x + w], (96, 96))
        out = {"enrol": {"saved": len(e["chips"]), "target": config.IMAGES_PER_STUDENT, "hint": e["hint"]},
               "faces": [{"box": list(box), "state": "marked" if ok else "unknown", "label": f"{len(e['chips'])}/{config.IMAGES_PER_STUDENT}"}] if box else []}
        if len(e["chips"]) >= config.IMAGES_PER_STUDENT:
            out["enrolled"] = self.finish_enrol()
        return out

    def finish_enrol(self):
        e, self.enrol, self.mode = self.enrol, None, "preview"
        if len(e["chips"]) < config.MIN_IMAGES_TO_TRAIN:
            return None
        self.students[e["id"]] = {"roll": e["roll"], "name": e["name"]}
        self.chips[e["id"]] = e["chips"]
        self.train()
        thumb = cv2.cvtColor(e["thumb"], cv2.COLOR_BGR2RGBA) if e["thumb"] is not None else np.zeros((96, 96, 4), np.uint8)
        return {"id": e["id"], "roll": e["roll"], "name": e["name"],
                "chips": np.stack(e["chips"]).tobytes(), "thumb": thumb.tobytes()}

    def cancel_enrol(self):
        return self.finish_enrol() if self.enrol else None

    def remove(self, sid):
        self.students.pop(int(sid), None)
        self.chips.pop(int(sid), None)
        self.train()

    # ------------------------------------------------------------------ frames
    def frame(self, rgba, w, h, now):
        frame = cv2.cvtColor(np.frombuffer(_bytes(rgba), np.uint8).reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
        t0 = time.perf_counter()
        if self.mode == "enrol":
            out = self._enrol_step(frame)
        else:
            out = self._recognise(frame, now)
        out["ms"] = round((time.perf_counter() - t0) * 1000)
        if self.view != "camera":
            out["view"] = self._view(frame).tobytes()
        return out

    def _recognise(self, frame, now):
        faces, explain = [], None
        gray = self.engine.gray(frame)
        for tr in self.tracker.update(self.engine.detect(frame)):
            if not self.trained:
                faces.append({"box": list(tr.face.box), "state": "face", "sources": tr.face.sources})
                continue
            chip = self.engine.describe(frame, tr.face)
            m = self.engine.predict(chip)
            tr.observe(m)
            if config.LIVENESS:
                tr.observe_eyes(self.engine.detector.count_eyes(gray, tr.face.box))
            student = self.students.get(tr.label)
            state = "unknown"
            if student is not None:
                if self.mode == "attendance" and tr.confirmed and tr.live and student["roll"] not in self.present:
                    self.present[student["roll"]] = now
                    self.events.insert(0, {"time": now, "roll": student["roll"], "name": student["name"]})
                if student["roll"] in self.present:
                    state = "marked"
                elif tr.confirmed and not tr.live:
                    state = "blink"
                else:
                    state = "verifying"
            elif self.mode == "attendance" and tr.age >= config.CONFIRM_FRAMES:
                self.unknown.add(tr.id)
            nearest = self.students.get(m.label)
            runner = "crowd" if m.second == CROWD else self.students.get(m.second, {}).get("name")
            faces.append({"box": list(tr.face.box), "state": state, "sources": tr.face.sources,
                          "name": student["name"] if student else None, "roll": student["roll"] if student else None,
                          "nearest": nearest["name"] if nearest else None, "runner_up": runner,
                          "dist": round(m.score, 3), "ratio": round(m.ratio, 3),
                          "streak": tr.streak, "blinked": bool(tr.blinked)})
            if explain is None or tr.face.box[2] > explain[0]:
                explain = (tr.face.box[2], chip, m)
        out = {"faces": faces, "present": self.present, "events": self.events[:12], "unknown_seen": len(self.unknown)}
        if explain is not None:
            out["explain"] = self._explain(explain[1], explain[2]).tobytes()
        return out

    # ------------------------------------------------------------------ pictures for the page
    def _view(self, frame):
        """320x240 RGBA picture of what the pipeline works on."""
        g = self.engine.detection_image(frame)
        if self.view == "contrast":
            img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        elif self.view == "texture":
            img = cv2.cvtColor(lbp.lbp_codes(cv2.resize(g, (320, 240)), 8, 1).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:   # votes: each detector on its own
            img = (frame * 0.55).astype(np.uint8)
            for x, y, w, h in self.engine.detector.run_haar(g):
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 160, 60), 3)
                cv2.putText(img, "Haar", (x + 6, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 160, 60), 2)
            for x, y, w, h in self.engine.detector.run_lbp(g):
                cv2.rectangle(img, (x, y), (x + w, y + h), (60, 170, 255), 3)
                cv2.putText(img, "LBP", (x + 6, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 170, 255), 2)
        return cv2.cvtColor(cv2.resize(img, (320, 240)), cv2.COLOR_BGR2RGBA)

    def _explain(self, chip, m):
        """Four 120x120 tiles: chip | LBP codes | where it differs from the match | the match."""
        S = 120
        tiles = [cv2.resize(chip, (S, S)), cv2.resize(lbp.lbp_codes(chip, 8, 1).astype(np.uint8), (S, S),
                                                      interpolation=cv2.INTER_NEAREST)]
        tiles = [cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) for t in tiles]
        heat = np.zeros((S, S, 3), np.uint8)
        if m.cells is not None:
            gx, gy = config.LBP_GRID
            planes = len(m.cells) // (gx * gy)
            c = (m.cells.reshape(planes, gy, gx) * self.engine.recognizer.weights.reshape(planes, gy, gx)).mean(0)
            hm = cv2.applyColorMap((np.clip(c / (np.percentile(c, 95) + 1e-9), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            heat = cv2.addWeighted(tiles[0], 0.45, cv2.resize(hm, (S, S), interpolation=cv2.INTER_NEAREST), 0.55, 0)
        tiles.append(heat)
        ref = np.full((S, S, 3), 40, np.uint8)
        if m.label in self.chips and self.chips[m.label]:
            ref = cv2.cvtColor(cv2.resize(self.chips[m.label][0], (S, S)), cv2.COLOR_GRAY2BGR)
        tiles.append(ref)
        return cv2.cvtColor(np.hstack(tiles), cv2.COLOR_BGR2RGBA)


core = Core()
