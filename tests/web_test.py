"""Browser test of the website version (web/), with a fake webcam fed from video.

    python web/build.py && python -m http.server 8899 &      # from the repo root
    python tests/web_test.py /tmp/webtest [URL]

Run 1 enrols Colin Powell (LFW) through the page and expects him marked present, then
downloads the CSV. Run 2 reuses the browser profile and shows four strangers for 40 s:
nobody may be marked present.
"""
import sys, time, json
from pathlib import Path
import cv2, numpy as np
from playwright.sync_api import sync_playwright
S = Path(sys.argv[1]); L = Path.home() / "datasets/lfw_home/lfw_funneled"
URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8899/web/dist/"

def y4m(path, photos, per=20, seed=0):
    rng = np.random.default_rng(seed)
    with open(path, "wb") as f:
        f.write(b"YUV4MPEG2 W640 H480 F15:1 Ip A1:1 C420jpeg\n")
        for ph in photos:
            face = cv2.resize(cv2.imread(str(ph)), (400, 400)).astype(np.float32)
            for _ in range(per):
                fr = np.full((480, 640, 3), 45, np.float32); dx, dy = rng.integers(-6, 7, 2)
                fr[40 + dy:440 + dy, 120 + dx:520 + dx] = face
                fr = np.clip(fr * rng.uniform(.9, 1.1) + rng.uniform(-8, 8) + rng.normal(0, 3, fr.shape), 0, 255).astype(np.uint8)
                f.write(b"FRAME\n" + cv2.cvtColor(fr, cv2.COLOR_BGR2YUV_I420).tobytes())

y4m(S / "powell.y4m", sorted((L / "Colin_Powell").glob("*.jpg"))[:12])
y4m(S / "strangers.y4m", [f for p in ("Ariel_Sharon", "Hugo_Chavez", "John_Ashcroft", "Junichiro_Koizumi") for f in sorted((L / p).glob("*.jpg"))[:4]], seed=1)

def check(c, msg):
    print(("  PASS  " if c else "  FAIL  ") + msg)
    if not c: raise SystemExit(1)

def browser(p, video):
    return p.chromium.launch_persistent_context(str(S / "profile"), executable_path="/usr/bin/google-chrome", headless=True,
        args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", f"--use-file-for-fake-video-capture={video}"],
        permissions=["camera"], viewport={"width": 1440, "height": 1000}, accept_downloads=True)

with sync_playwright() as p:
    print("run 1: camera shows Colin Powell")
    ctx = browser(p, S / "powell.y4m"); pg = ctx.new_page()
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e))); pg.on("response", lambda x: x.status >= 400 and errs.append(f"{x.status} {x.url}"))
    pg.goto(URL); check("Try it live" in pg.inner_text("body"), "project page loads with a 'Try it live' button")
    pg.click("text=Try it live in your browser")
    t0 = time.time(); pg.wait_for_function("document.getElementById('boot').hidden", timeout=180000)
    check(True, f"Python + OpenCV started in the browser in {time.time() - t0:.0f} s")
    if pg.is_visible("#btnCam"): pg.click("#btnCam")
    pg.wait_for_function("document.getElementById('camOff').hidden", timeout=15000); check(True, "camera on")
    time.sleep(3); pg.screenshot(path=str(S / "1_live_empty.png"))
    check("nobody is enrolled" in pg.inner_text("#say"), f"narration with nobody enrolled: \"{pg.inner_text('#say').splitlines()[0]}\"")
    pg.click(".tabs a[data-tab=register]"); pg.fill("#rRoll", "WEB01"); pg.fill("#rName", "Colin Powell"); pg.click("#btnReg")
    pg.wait_for_function("document.getElementById('regHint').textContent.startsWith('Done')", timeout=90000)
    check(True, "enrolled from the webcam in the browser: " + pg.inner_text("#regHint"))
    pg.screenshot(path=str(S / "2_enrolled.png"))
    pg.click(".tabs a[data-tab=live]"); pg.click("#btnStart")
    pg.wait_for_function("document.getElementById('kPresent').textContent === '1'", timeout=90000)
    check(True, "Colin Powell marked present"); time.sleep(1); pg.screenshot(path=str(S / "3_present.png"))
    fps = pg.inner_text("#fps"); check("frames/s" in fps, "speed: " + fps)
    pg.click("#viewSeg button[data-view=texture]"); time.sleep(2); pg.screenshot(path=str(S / "4_texture.png"))
    pg.click("#viewSeg button[data-view=camera]")
    pg.click("#btnStop"); pg.click(".tabs a[data-tab=records]"); time.sleep(1.5)
    with pg.expect_download() as d: pg.click("[data-csv]")
    csv = Path(d.value.path()).read_text(); print("    CSV:", csv.strip().replace("\n", " | "))
    check(csv.startswith("roll,name,status,time") and "WEB01,Colin Powell,Present," in csv, "CSV download from Records")
    check(not errs, f"no page errors {errs[:2]}")
    ctx.close()

    print("run 2: same browser (enrolment kept), camera shows four strangers")
    ctx = browser(p, S / "strangers.y4m"); pg = ctx.new_page()
    pg.goto(URL + "app.html"); pg.wait_for_function("document.getElementById('boot').hidden", timeout=180000)
    if pg.is_visible("#btnCam"): pg.click("#btnCam")
    pg.wait_for_function("document.getElementById('camOff').hidden", timeout=15000)
    check(pg.inner_text("#kTotal") == "1", "enrolment survived a browser restart (stored in IndexedDB)")
    pg.evaluate("localStorage.setItem('faceroll-records','{}')"); pg.reload(); pg.wait_for_function("document.getElementById('boot').hidden", timeout=180000)
    if pg.is_visible("#btnCam"): pg.click("#btnCam")
    pg.click("#btnStart"); bad, seen = 0, 0; t0 = time.time()
    while time.time() - t0 < 40:
        st = pg.evaluate("(last.faces||[]).map(f => f.state)"); seen += len(st)
        bad += pg.inner_text("#kPresent") != "0"
        time.sleep(0.5)
    pg.screenshot(path=str(S / "5_strangers.png"))
    check(seen > 0 and bad == 0, f"40 s of strangers ({seen} face observations): nobody marked present")
    ctx.close()
print("WEB E2E PASSED")
