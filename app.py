"""Web dashboard for the attendance system.

    python app.py            # then open http://127.0.0.1:8000

One background thread owns the webcam and runs the same pipeline as the CLI scripts
(engines.FaceEngine + attendance.AttendanceSession). The browser gets:
  /stream.mjpg      annotated live video (MJPEG)
  /explain.jpg      "why" panel for the main face: chip, LBP codes, per-cell distance map
  /api/...          JSON for status, registration, training, students, records, settings
"""
import json
import shutil
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

import config
import lbp
import utils
from attendance import AttendanceSession, csv_path_for_today, load_present
from engines import FaceEngine
from recognizer import load_labels

ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = config.MODELS_DIR / "settings.json"
SETTABLE = {   # name: (type, needs retrain)
    "DETECTOR": (str, False), "ILLUMINATION": (str, True), "LBP_TERNARY": (bool, True),
    "DIST_THRESHOLD": (float, False), "RATIO_THRESHOLD": (float, False),
    "CONFIRM_FRAMES": (int, False), "LIVENESS": (bool, False), "MIN_NEIGHBORS": (int, False),
}

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")


# ============================================================================ settings
def apply_settings(values):
    retrain = False
    for k, v in values.items():
        if k not in SETTABLE:
            continue
        typ, needs = SETTABLE[k]
        v = typ(v)
        if k == "DETECTOR" and v not in ("haar", "lbp", "union", "consensus"):
            continue
        if k == "ILLUMINATION" and v not in ("clahe", "tantriggs", "none"):
            continue
        if getattr(config, k) != v:
            retrain |= needs
            setattr(config, k, v)
    return retrain


def current_settings():
    return {k: getattr(config, k) for k in SETTABLE}


def save_settings():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(current_settings(), indent=2))


if SETTINGS_PATH.exists():
    apply_settings(json.loads(SETTINGS_PATH.read_text()))


# ============================================================================ drawing
# BGR versions of the UI palette (static/style.css)
COL = {"marked": (127, 211, 143), "verifying": (65, 165, 242), "blink": (255, 180, 111),
       "unknown": (87, 107, 255), "face": (220, 231, 235)}


def draw_face(view, box, color, title, sub=None):
    x, y, w, h = box
    l = max(12, w // 5)
    for (px, py), (dx, dy) in (((x, y), (1, 1)), ((x + w, y), (-1, 1)), ((x, y + h), (1, -1)), ((x + w, y + h), (-1, -1))):
        cv2.line(view, (px, py), (px + dx * l, py), color, 3, cv2.LINE_AA)
        cv2.line(view, (px, py), (px, py + dy * l), color, 3, cv2.LINE_AA)
    cv2.rectangle(view, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(title, font, 0.55, 1)
    ty = y - 10 if y - th - 14 > 0 else y + h + th + 12
    cv2.rectangle(view, (x, ty - th - 6), (x + tw + 12, ty + 6), color, -1)
    cv2.putText(view, title, (x + 6, ty), font, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(view, sub, (x + 4, y + h - 8), font, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, sub, (x + 4, y + h - 8), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def placeholder(text):
    img = np.full((480, 640, 3), 28, np.uint8)
    cv2.putText(img, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def explain_panel(chip, match, recognizer, student_dir):
    """chip | LBP codes | per-cell distance to best template | matched student's photo"""
    S = 180
    tiles = [cv2.resize(chip, (S, S))]
    tiles.append(cv2.resize(lbp.lbp_codes(chip, 8, 1).astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST))
    tiles = [cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) for t in tiles]
    gx, gy = config.LBP_GRID
    heat = np.zeros((S, S, 3), np.uint8)
    if match is not None and match.cells is not None:
        planes = len(match.cells) // (gx * gy)
        cells = match.cells.reshape(planes, gy, gx).mean(0)
        wts = recognizer.weights.reshape(planes, gy, gx).mean(0)
        contrib = cells * wts
        norm = np.clip(contrib / (np.percentile(contrib, 95) + 1e-9), 0, 1)
        hm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        hm = cv2.resize(hm, (S, S), interpolation=cv2.INTER_NEAREST)
        heat = cv2.addWeighted(tiles[0], 0.45, hm, 0.55, 0)
        for i in range(1, gx):
            cv2.line(heat, (i * S // gx, 0), (i * S // gx, S), (60, 60, 60), 1)
            cv2.line(heat, (0, i * S // gy), (S, i * S // gy), (60, 60, 60), 1)
    tiles.append(heat)
    ref = np.full((S, S, 3), 40, np.uint8)
    if student_dir is not None:
        files = utils.image_files(student_dir)
        if files:
            im = cv2.imread(str(files[0]))
            if im is not None:
                ref = cv2.resize(im, (S, S))
    tiles.append(ref)
    return np.hstack(tiles)


# ============================================================================ camera worker
class Worker:
    def __init__(self):
        self.lock = threading.RLock()
        self.engine = FaceEngine()
        self.labels = load_labels()
        self.model_ok = bool(self.labels) and self.engine.load()
        self.mode = "preview"           # preview | attendance | register
        self.session = None
        self.reg = None                 # registration state dict
        self.jpeg = cv2.imencode(".jpg", placeholder("Starting camera..."))[1].tobytes()
        self.explain = cv2.imencode(".jpg", placeholder("No face"))[1].tobytes()
        self.faces = []
        self.fps = 0.0
        self.camera_ok = False
        self.training = False
        self.message = ""
        self.last_eval = None
        self.camera_on = True
        threading.Thread(target=self.loop, daemon=True).start()

    # ---------------------------------------------------------------- frame loop
    def loop(self):
        cap, t_prev = None, time.time()
        while True:
            if not self.camera_on:
                if cap is not None:
                    cap.release()
                    cap = None
                self.camera_ok = False
                self.jpeg = cv2.imencode(".jpg", placeholder("Camera paused"))[1].tobytes()
                time.sleep(0.2)
                continue
            if cap is None:
                try:
                    cap = utils.open_camera()
                    self.camera_ok = True
                except RuntimeError as e:
                    self.camera_ok = False
                    self.jpeg = cv2.imencode(".jpg", placeholder("Camera unavailable"))[1].tobytes()
                    self.message = str(e)
                    time.sleep(2)
                    continue
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                continue
            frame = utils.prepare_frame(frame)
            try:
                with self.lock:
                    view = self.process(frame)
            except Exception as e:      # never kill the camera thread
                view = frame
                self.message = f"Error: {e}"
            self.jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1 / max(now - t_prev, 1e-3))
            t_prev = now

    def process(self, frame):
        if self.mode == "register":
            return self.process_register(frame)
        if self.mode == "attendance" and self.session is not None:
            results = self.session.step(frame)
        elif self.model_ok:
            results = self.preview_step(frame)
        else:
            results = [{"box": f.box, "sources": f.sources, "state": "face", "match": None, "student": None,
                        "chip": None, "track": None} for f in self.engine.detect(frame)]
        view = frame.copy()
        faces = []
        for r in results:
            m, s, st = r["match"], r["student"], r["state"]
            if st == "face":
                title = f"face [{r['sources']}]"
            elif st == "unknown":
                title = "Unknown"
            elif st == "marked":
                title = f"{s['name']}  present"
            elif st == "blink":
                title = f"{s['name']}  blink!"
            else:
                tr = r["track"]
                title = f"{s['name']}  {tr.streak}/{config.CONFIRM_FRAMES}" if tr else s["name"]
            sub = f"d {m.score:.2f}  r {m.ratio:.2f}  [{r['sources']}]" if m is not None else None
            draw_face(view, r["box"], COL[st], title, sub)
            faces.append({
                "state": st, "sources": r["sources"], "box": list(r["box"]),
                "name": s["name"] if s else None, "roll": s["roll"] if s else None,
                "dist": round(m.score, 3) if m else None,
                "ratio": round(m.ratio, 3) if m else None,
                "streak": r["track"].streak if r["track"] else 0,
                "blinked": bool(r["track"].blinked) if r["track"] else False,
            })
        self.faces = faces
        main = max((r for r in results if r["chip"] is not None), key=lambda r: r["box"][2] * r["box"][3],
                   default=None)
        if main is not None:
            sdir = None
            if main["match"] is not None and main["match"].label in self.labels:
                roll = self.labels[main["match"].label]["roll"]
                sdir = next(iter(config.DATASET_DIR.glob(f"{roll}_*")), None)
            panel = explain_panel(main["chip"], main["match"], self.engine.recognizer, sdir)
            self.explain = cv2.imencode(".jpg", panel)[1].tobytes()
        return view

    def preview_step(self, frame):
        """Recognise without recording anything (the 'preview' mode)."""
        if not hasattr(self, "_ptracker"):
            from tracker import Tracker
            self._ptracker = Tracker()
        out = []
        for tr in self._ptracker.update(self.engine.detect(frame)):
            chip = self.engine.describe(frame, tr.face)
            m = self.engine.predict(chip)
            tr.observe(m)
            s = self.labels.get(m.label) if m.ok else None
            out.append({"box": tr.face.box, "sources": tr.face.sources, "match": m, "student": s, "chip": chip,
                        "track": tr, "state": "verifying" if s else "unknown"})
        return out

    def process_register(self, frame):
        reg = self.reg
        view = frame.copy()
        face = utils.largest_face(self.engine.detect(frame))
        reg["frame"] += 1
        if face is None:
            reg["hint"] = "No face detected - look at the camera"
        else:
            problem = utils.quality_problem(frame, face.box)
            if problem:
                reg["hint"] = problem
            else:
                reg["hint"] = ["Look straight", "Turn slightly left", "Turn slightly right",
                               "Tilt chin up a little", "Smile", "Neutral face"][reg["saved"] * 6 // reg["target"]]
                if reg["frame"] % config.SAVE_EVERY_N_FRAMES == 0:
                    cv2.imwrite(str(reg["dir"] / f"{reg['saved']}.png"), utils.padded_crop(frame, face.box))
                    reg["saved"] += 1
            draw_face(view, face.box, COL["unknown"] if problem else COL["marked"],
                      f"{reg['saved']}/{reg['target']}")
        if reg["saved"] >= reg["target"]:
            self.mode = "preview"
            self.message = f"Registered {reg['roll']} {reg['name']} ({reg['saved']} images). Training..."
            self.train_async()
        return view

    # ---------------------------------------------------------------- actions
    def train_async(self):
        if self.training:
            return
        self.training = True

        def job():
            from train import train_all
            try:
                engine = FaceEngine()
                engine, descriptors, labels = train_all(engine)
                with self.lock:
                    self.engine.recognizer = engine.recognizer
                    self.labels = load_labels()
                    self.model_ok = True
                    if self.session is not None:
                        self.session.labels = self.labels
                    if hasattr(self, "_ptracker"):
                        del self._ptracker
                self.message = f"Model trained: {len(labels)} students, {len(descriptors)} images."
            except SystemExit as e:
                self.message = f"Training failed: {e}"
            except Exception as e:
                self.message = f"Training failed: {e}"
            finally:
                self.training = False

        threading.Thread(target=job, daemon=True).start()

    def start_attendance(self):
        with self.lock:
            if not self.model_ok:
                return "Register students and train first."
            if self.session is None:
                self.session = AttendanceSession(self.engine, self.labels, csv_path_for_today())
            self.mode = "attendance"
            self.message = "Attendance session running."

    def stop_attendance(self):
        with self.lock:
            if self.session is not None:
                self.session.finish()
                p, n = self.session.count_present(), len(self.labels)
                self.message = f"Session saved: {p}/{n} present -> {self.session.path.name}"
                self.session = None
            self.mode = "preview"


worker = None


# ============================================================================ routes
@app.route("/")
def index():
    return send_from_directory(ROOT / "static", "index.html")


@app.route("/stream.mjpg")
def stream():
    def gen():
        last = None
        while True:
            frame = worker.jpeg
            if frame is not last:
                last = frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/explain.jpg")
def explain():
    return Response(worker.explain, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


def roster():
    labels = worker.labels
    present = worker.session.present if worker.session else load_present(csv_path_for_today())
    rows = [{"roll": s["roll"], "name": s["name"], "status": "Present" if s["roll"] in present else "Absent",
             "time": present.get(s["roll"], {}).get("time", "")} for s in labels.values()]
    return sorted(rows, key=lambda r: (r["status"] != "Present", r["roll"]))


@app.get("/api/status")
def status():
    w = worker
    reg = None
    if w.reg and w.mode == "register":
        reg = {k: w.reg[k] for k in ("roll", "name", "saved", "target", "hint")}
    return jsonify({
        "mode": w.mode, "fps": round(w.fps, 1), "camera_ok": w.camera_ok, "camera_on": w.camera_on,
        "faces": w.faces, "roster": roster(), "training": w.training, "message": w.message,
        "model_ok": w.model_ok, "students": len(w.labels), "register": reg,
        "events": (w.session.events[-12:][::-1] if w.session else []),
        "unknown_seen": len(w.session.unknown_tracks) if w.session else 0,
        "recognizer": w.engine.recognizer.kind, "settings": current_settings(),
        "date": csv_path_for_today().stem.replace("attendance_", ""),
    })


@app.post("/api/attendance/start")
def att_start():
    err = worker.start_attendance()
    return (jsonify(error=err), 400) if err else jsonify(ok=True)


@app.post("/api/attendance/stop")
def att_stop():
    worker.stop_attendance()
    return jsonify(ok=True)


@app.post("/api/camera")
def camera_toggle():
    worker.camera_on = bool(request.json.get("on", True))
    return jsonify(ok=True)


def _student_args(data):
    import re
    roll = str(data.get("roll", "")).strip()
    name = " ".join(str(data.get("name", "")).split())
    if not re.fullmatch(r"[A-Za-z0-9-]+", roll):
        return None, None, "Roll number may only contain letters, digits and '-'."
    if not re.fullmatch(r"[A-Za-z .-]+", name):
        return None, None, "Name may only contain letters, spaces, '.' and '-'."
    return roll, name, None


def _prepare_dir(roll, name):
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for p in config.DATASET_DIR.glob(f"{roll}_*"):
        shutil.rmtree(p)
    d = config.DATASET_DIR / f"{roll}_{name}"
    d.mkdir()
    return d


@app.post("/api/register/start")
def reg_start():
    roll, name, err = _student_args(request.json or {})
    if err:
        return jsonify(error=err), 400
    exists = any(config.DATASET_DIR.glob(f"{roll}_*")) if config.DATASET_DIR.exists() else False
    if exists and not request.json.get("overwrite"):
        return jsonify(error=f"Roll {roll} is already registered.", exists=True), 409
    with worker.lock:
        if worker.session is not None:
            worker.stop_attendance()
        worker.reg = {"roll": roll, "name": name, "dir": _prepare_dir(roll, name), "saved": 0,
                      "target": config.IMAGES_PER_STUDENT, "frame": 0, "hint": "Look at the camera"}
        worker.mode = "register"
    return jsonify(ok=True)


@app.post("/api/register/cancel")
def reg_cancel():
    with worker.lock:
        if worker.mode == "register":
            worker.mode = "preview"
            if worker.reg and worker.reg["saved"] < config.MIN_IMAGES_TO_TRAIN:
                shutil.rmtree(worker.reg["dir"], ignore_errors=True)
                worker.message = "Registration cancelled."
            else:
                worker.message = "Registration stopped early; training on the images captured."
                worker.train_async()
    return jsonify(ok=True)


@app.post("/api/register/upload")
def reg_upload():
    roll, name, err = _student_args(request.form)
    if err:
        return jsonify(error=err), 400
    files = request.files.getlist("photos")
    if not files:
        return jsonify(error="Choose some photos."), 400
    from register import import_folder
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(files):
            suffix = Path(f.filename or "x.jpg").suffix.lower()
            if suffix in utils.IMAGE_SUFFIXES:
                f.save(Path(tmp) / f"{i:04d}{suffix}")
        out = _prepare_dir(roll, name)
        with worker.lock:
            saved = import_folder(worker.engine, Path(tmp), out)
    if saved < config.MIN_IMAGES_TO_TRAIN:
        shutil.rmtree(out, ignore_errors=True)
        return jsonify(error=f"Only {saved} photos had a usable face; need {config.MIN_IMAGES_TO_TRAIN}."), 400
    worker.message = f"Imported {saved} photos for {roll} {name}. Training..."
    worker.train_async()
    return jsonify(ok=True, saved=saved)


@app.post("/api/train")
def train_route():
    worker.train_async()
    return jsonify(ok=True)


@app.get("/api/students")
def students():
    out = []
    for d in utils.student_dirs():
        roll, name = utils.parse_student_dir(d.name)
        out.append({"roll": roll, "name": name, "images": len(utils.image_files(d)),
                    "trained": any(s["roll"] == roll for s in worker.labels.values())})
    return jsonify(out)


@app.get("/api/students/<roll>/photo")
def student_photo(roll):
    d = next(iter(config.DATASET_DIR.glob(f"{roll}_*")), None) if config.DATASET_DIR.exists() else None
    files = utils.image_files(d) if d else []
    if not files:
        abort(404)
    idx = min(int(request.args.get("i", 0)), len(files) - 1)
    return send_file(files[idx])


@app.delete("/api/students/<roll>")
def student_delete(roll):
    for p in config.DATASET_DIR.glob(f"{roll}_*"):
        shutil.rmtree(p)
    if utils.student_dirs():
        worker.train_async()
    else:
        with worker.lock:
            worker.labels, worker.model_ok = {}, False
            for p in (config.LABELS_PATH, config.OPENSET_MODEL_PATH, config.MODEL_PATH):
                p.unlink(missing_ok=True)
    return jsonify(ok=True)


@app.get("/api/records")
def records():
    files = sorted(config.RECORDS_DIR.glob("attendance_*.csv"), reverse=True) if config.RECORDS_DIR.exists() else []
    out = []
    import pandas as pd
    for f in files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        out.append({"date": f.stem.replace("attendance_", ""), "present": int((df["status"] == "Present").sum()),
                    "total": len(df), "rows": df.to_dict("records")})
    return jsonify(out)


@app.get("/api/records/<date>.csv")
def record_csv(date):
    p = config.RECORDS_DIR / f"attendance_{date}.csv"
    if not p.exists() or not p.resolve().is_relative_to(config.RECORDS_DIR.resolve()):
        abort(404)
    return send_file(p, as_attachment=True)


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        with worker.lock:
            retrain = apply_settings(request.json or {})
            worker.engine = FaceEngine()
            worker.model_ok = bool(worker.labels) and worker.engine.load()
            if worker.session is not None:
                worker.session.engine = worker.engine
        save_settings()
        if retrain or (worker.labels and not worker.model_ok):
            worker.train_async()
    return jsonify(current_settings())


@app.post("/api/calibrate")
def calibrate():
    """Run the open-set evaluation on the registered students and suggest thresholds."""
    import contextlib
    import io
    from evaluate import evaluate
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            res = evaluate(FaceEngine(), utils.student_dirs(), None, 5)
    except SystemExit as e:
        return jsonify(error=str(e) or buf.getvalue()), 400
    worker.last_eval = res
    return jsonify(result=res, log=buf.getvalue())


@app.get("/api/benchmarks")
def benchmarks():
    out = {}
    for f in sorted(config.RESULTS_DIR.glob("*.json")):
        if f.stem != "weights":
            out[f.stem] = json.loads(f.read_text())
    return jsonify(out)


@app.get("/figures/<path:name>")
def figures(name):
    return send_from_directory(config.FIGURES_DIR, name)


def main():
    import argparse
    global worker
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    worker = Worker()
    print(f"Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
