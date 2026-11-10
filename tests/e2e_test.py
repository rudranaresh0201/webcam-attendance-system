"""End-to-end test: the real web app, driven through a real browser.

    pip install playwright        # once (or: uv run --with playwright python tests/e2e_test.py)
    python tests/e2e_test.py --lfw ~/datasets/lfw_home/lfw_funneled [--chrome /usr/bin/google-chrome]

A video file stands in for the webcam (FACEROLL_CAMERA) and a temporary folder for the
data (FACEROLL_DATA), so your real enrolments and records are never touched.

  run 1  enrol two students by uploading photos in the browser, start attendance, and
         show the "camera" 40 s of four OTHER people -> nobody may be marked present
  run 2  same enrolment; the camera shows the two students (photos never used for
         enrolment) -> both marked present; then check private mode, the welcome
         message, the view switch, Finish and save, the Records table and the CSV
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STUDENTS = [("E2E01", "Colin Powell", "Colin_Powell"), ("E2E02", "Donald Rumsfeld", "Donald_Rumsfeld")]
STRANGERS = ["Ariel_Sharon", "Hugo_Chavez", "Junichiro_Koizumi", "John_Ashcroft"]
PORT = 8765
URL = f"http://127.0.0.1:{PORT}"


def make_video(path, photos, frames_per_photo=20, seed=0):
    """Each photo becomes ~1.3 s of 'someone sitting in front of the camera': enlarged on
    a 640x480 frame with small movement, brightness flicker and sensor noise."""
    rng = np.random.default_rng(seed)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (640, 480))
    blank = np.full((480, 640, 3), 45, np.uint8)
    for _ in range(10):
        out.write(blank)
    for photo in photos:
        face = cv2.resize(cv2.imread(str(photo)), (400, 400)).astype(np.float32)
        for _ in range(frames_per_photo):
            frame = np.full((480, 640, 3), 45, np.float32)
            dx, dy = rng.integers(-6, 7, 2)
            frame[40 + dy:440 + dy, 120 + dx:520 + dx] = face
            frame = frame * rng.uniform(0.9, 1.1) + rng.uniform(-8, 8) + rng.normal(0, 3, frame.shape)
            out.write(np.clip(frame, 0, 255).astype(np.uint8))
        for _ in range(4):
            out.write(blank)
    out.release()


def start_app(data, video):
    env = dict(os.environ, FACEROLL_DATA=str(data), FACEROLL_CAMERA=str(video))
    proc = subprocess.Popen([sys.executable, str(ROOT / "app.py"), "--port", str(PORT)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(80):
        try:
            urllib.request.urlopen(URL + "/api/status", timeout=1)
            return proc
        except OSError:
            time.sleep(0.25)
    proc.kill()
    raise RuntimeError("app did not start")


def status(page):
    return page.evaluate("fetch('/api/status').then(r => r.json())")


def wait_for(page, cond, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = status(page)
        if cond(s):
            return s
        time.sleep(0.5)
    raise AssertionError(f"timed out waiting for: {what}")


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        raise AssertionError(msg)


def main():
    from playwright.sync_api import sync_playwright
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw", type=Path, default=Path.home() / "datasets/lfw_home/lfw_funneled")
    ap.add_argument("--chrome", default=os.environ.get("CHROME_PATH"))
    ap.add_argument("--keep", action="store_true", help="keep the temp folder (screenshots, CSV)")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="faceroll_e2e_"))
    data, shots = tmp / "data", tmp / "screenshots"
    shots.mkdir(parents=True)
    photos = {p: sorted((args.lfw / p).glob("*.jpg")) for _, _, p in STUDENTS}
    enrol_dirs = {}
    for roll, name, p in STUDENTS:
        d = tmp / "upload" / p
        d.mkdir(parents=True)
        for f in photos[p][:15]:                                        # enrolment photos
            shutil.copy(f, d)
        enrol_dirs[roll] = d
    make_video(tmp / "strangers.avi", [f for s in STRANGERS for f in sorted((args.lfw / s).glob("*.jpg"))[:3]])
    make_video(tmp / "students.avi", [f for _, _, p in STUDENTS for f in photos[p][15:23]], seed=1)   # 8 photos never used for enrolment

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.chrome) if args.chrome else pw.chromium.launch()
        # ------------------------------------------------------------------ run 1: strangers only
        print("run 1: two students enrolled, camera shows four strangers")
        app = start_app(data, tmp / "strangers.avi")
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(URL, wait_until="commit")
            page.wait_for_selector("#say")
            page.click(".tabs a[data-tab=register]")
            page.click("#tab-register .seg button[data-src=upload]")
            for i, (roll, name, _) in enumerate(STUDENTS, 1):
                page.fill("#rRoll", roll)
                page.fill("#rName", name)
                page.set_input_files("#rFiles", sorted(str(f) for f in enrol_dirs[roll].iterdir()))
                page.click("#btnUpload")
                wait_for(page, lambda s, i=i: s["students"] == i and not s["training"], 60, f"{name} enrolled")
            check(True, "enrolled 2 students by uploading photos in the browser")
            page.click(".tabs a[data-tab=students]")
            page.wait_for_timeout(1500)
            check(page.locator(".flip").count() == 2, "Students tab shows 2 cards")
            page.click(".tabs a[data-tab=live]")
            page.click("#btnStart")
            s = wait_for(page, lambda s: s["mode"] == "attendance", 10, "session started")
            page.wait_for_function("document.querySelector('#sessionText').textContent.includes('Recording')", timeout=5000)
            check(True, "header says attendance is recording")
            check(s["cohort"], "background crowd is loaded")
            seen_unknown, t0 = 0, time.time()
            while time.time() - t0 < 40:
                s = status(page)
                seen_unknown = max(seen_unknown, s["unknown_seen"])
                assert not any(r["status"] == "Present" for r in s["roster"]), f"stranger credited: {s['roster']}"
                time.sleep(0.5)
            page.screenshot(path=str(shots / "1_strangers.png"))
            check(seen_unknown > 0, f"strangers were seen and labelled unknown ({seen_unknown} sightings)")
            check(True, "40 s of strangers: nobody marked present")
            page.click("#btnStop")
            wait_for(page, lambda s: s["mode"] == "preview", 10, "session stopped")
        finally:
            app.terminate()
            app.wait()
        # the register written by run 1 must list both students as absent
        csv = next((data / "records").glob("attendance_*.csv")).read_text()
        check(csv.count("Absent") == 2 and "Present" not in csv, "run 1 register: both students Absent")
        next((data / "records").glob("attendance_*.csv")).unlink()

        # ------------------------------------------------------------------ run 2: the students
        print("run 2: camera shows the two students (photos never used for enrolment)")
        app = start_app(data, tmp / "students.avi")
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(URL, wait_until="commit")
            page.wait_for_selector("#say")
            page.evaluate("localStorage.setItem('showNames','0')")
            page.reload(wait_until="commit")
            page.click(".tabs a[data-tab=live]") if page.locator("#tab-live.on").count() == 0 else None
            page.click("#btnStart")
            page.wait_for_function("!document.querySelector('#privNote').hidden", timeout=5000)
            check(page.is_hidden("#roster") and page.is_visible("#privNote"), "private mode: names hidden by default")
            welcomed = False
            t0 = time.time()
            while time.time() - t0 < 90:
                s = status(page)
                if page.is_visible("#welcome") and "Welcome" in page.inner_text("#welcome"):
                    welcomed = True
                if sum(r["status"] == "Present" for r in s["roster"]) == 2:
                    break
                time.sleep(0.4)
            page.screenshot(path=str(shots / "2_students.png"))
            present = [r["name"] for r in s["roster"] if r["status"] == "Present"]
            check(len(present) == 2, f"both students marked present within {time.time() - t0:.0f} s (limit 90): {present}")
            check(welcomed or page.is_visible("#welcome"), "private mode shows a 'Welcome, <name>' message")
            page.check("#showNames")
            page.wait_for_timeout(800)
            check(page.locator("#roster .tag.Present").count() == 2, "with names shown, roster lists 2 Present")
            page.click("#viewSeg button[data-view=texture]")
            page.wait_for_timeout(800)
            check(status(page)["view"] == "texture", "view switch: video now shows LBP texture")
            page.screenshot(path=str(shots / "3_texture.png"))
            page.click("#viewSeg button[data-view=camera]")
            check(len(page.inner_text("#say")) > 10, f"narration: \"{page.inner_text('#say').splitlines()[0]}\"")
            page.click("#btnStop")
            wait_for(page, lambda s: s["mode"] == "preview", 10, "session stopped")
            page.click(".tabs a[data-tab=records]")
            page.wait_for_timeout(1500)
            check(page.locator("#recordList tbody tr").count() == 2, "Records tab shows today's 2 rows")
            date = status(page)["date"]
            csv = page.request.get(f"{URL}/api/records/{date}.csv").text()
            rows = csv.strip().splitlines()
            check(rows[0] == "roll,name,status,time" and len(rows) == 3 and csv.count("Present") == 2,
                  "CSV download: header + 2 Present rows with times")
            page.screenshot(path=str(shots / "4_records.png"))
        finally:
            app.terminate()
            app.wait()
        browser.close()
    print(f"\nE2E TEST PASSED  (screenshots: {shots})" if args.keep else "\nE2E TEST PASSED")
    if not args.keep:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
