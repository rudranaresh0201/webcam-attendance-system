"""Assemble the static website (GitHub Pages) into web/dist/.

    python web/build.py
The live app runs the repository's own Python modules in the browser (Pyodide), so they
are copied in unchanged next to web/webcore.py.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB, DIST = ROOT / "web", ROOT / "web" / "dist"
MODULES = ["config.py", "lbp.py", "detector.py", "recognizer.py", "tracker.py", "engines.py", "utils.py"]
CASCADES = ["haarcascade_frontalface_default.xml", "haarcascade_eye.xml", "lbpcascade_frontalface_improved.xml"]
ASSETS = ["cell_weights_ulbp_8.1-8.2_g8x8.npy", "cohort_ulbp_8.1-8.2_g8x8_clahe.npz"]

shutil.rmtree(DIST, ignore_errors=True)
(DIST / "py" / "cascades").mkdir(parents=True)
(DIST / "py" / "assets").mkdir()
for f in ["index.html", "app.html", "app.css", "app.js", "worker.js"]:
    shutil.copy(WEB / f, DIST / f)
shutil.copy(WEB / "webcore.py", DIST / "py" / "webcore.py")
for f in MODULES:
    shutil.copy(ROOT / f, DIST / "py" / f)
for f in CASCADES:
    shutil.copy(ROOT / "cascades" / f, DIST / "py" / "cascades" / f)
for f in ASSETS:
    shutil.copy(ROOT / "assets" / f, DIST / "py" / "assets" / f)
(DIST / ".nojekyll").touch()
print("built", DIST, sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file()) // 1024, "KB")
