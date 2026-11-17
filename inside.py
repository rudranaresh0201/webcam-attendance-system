"""'Inside the pipeline': what each classical stage computes, for the dashboard's Inside tab.

    haar_trace     re-runs the Viola-Jones cascade (haarcascade_frontalface_default.xml) in
                   NumPy on one face window: every stage, every weak classifier (a Haar-like
                   feature, its value, its threshold and its vote). OpenCV does exactly this
                   inside detectMultiScale but never shows it.
    stage_tiles    the face at each step of the pipeline, side by side
    fingerprint    the LBP descriptor of the face (what is actually compared) and its
                   distance to every enrolled student and to the background crowd
"""
import xml.etree.ElementTree as ET
from functools import lru_cache

import cv2
import numpy as np

import config
import lbp
from detector import _cascade


# --------------------------------------------------------------------------- Haar cascade
@lru_cache(maxsize=1)
def haar_model():
    """Parse the cascade XML: window size, feature rectangles, and stages of stumps.
    Returns (W, H, features, stages); features[i] = [(x, y, w, h, weight), ...];
    stages[s] = (stage_threshold, [(feature_idx, node_threshold, left_leaf, right_leaf), ...])."""
    root = ET.parse(str(_cascade("haarcascade_frontalface_default.xml"))).getroot()
    c = root.find("cascade")
    W, H = int(c.find("width").text), int(c.find("height").text)
    features = []
    for f in c.find("features"):
        rects = []
        for r in f.find("rects"):
            x, y, w, h, wt = r.text.split()
            rects.append((int(x), int(y), int(w), int(h), float(wt)))
        features.append(rects)
    stages = []
    for s in c.find("stages"):
        weak = []
        for wc in s.find("weakClassifiers"):
            _, _, idx, thr = wc.find("internalNodes").text.split()
            left, right = (float(v) for v in wc.find("leafValues").text.split())
            weak.append((int(idx), float(thr), left, right))
        stages.append((float(s.find("stageThreshold").text), weak))
    return W, H, features, stages


def haar_window(gray, box, cascade=None):
    """A 24x24 window that the cascade actually accepted for this face.

    The Haar box drawn on screen is the average of many overlapping windows that each
    passed all stages (their number is the minNeighbors vote). The average itself was
    never tested, so we ask OpenCV for the raw accepted windows around the box
    (minNeighbors=0), take the one closest to it, and rebuild it exactly as
    detectMultiScale saw it: the image shrunk by 1.1^k, then a 24x24 crop.
    Returns (window, votes) where votes = accepted raw windows around this face."""
    W, H, _, _ = haar_model()
    x, y, w, h = box
    fallback = cv2.resize(gray[y:y + h, x:x + w], (W, H), interpolation=cv2.INTER_LINEAR)
    if cascade is None:
        return fallback, 0
    m = int(0.25 * w)
    x0, y0 = max(x - m, 0), max(y - m, 0)
    roi = gray[y0:y + h + m, x0:x + w + m]
    raw = cascade.detectMultiScale(roi, scaleFactor=config.SCALE_FACTOR, minNeighbors=0,
                                   minSize=(int(w * 0.7),) * 2, maxSize=(int(w * 1.45),) * 2)
    if not len(raw):
        return fallback, 0
    from detector import iou
    rel = (x - x0, y - y0, w, h)
    raw = sorted((tuple(int(v) for v in r) for r in raw), key=lambda r: -iou(r, rel))
    for rx, ry, rw, rh in raw[:3]:
        k = min(range(60), key=lambda k: abs(round(W * config.SCALE_FACTOR ** k) - rw))
        f = config.SCALE_FACTOR ** k
        small = cv2.resize(roi, (int(round(roi.shape[1] / f)), int(round(roi.shape[0] / f))), interpolation=cv2.INTER_LINEAR)
        sx, sy = int(round(rx / f)), int(round(ry / f))
        for dy, dx in ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            win = small[sy + dy:sy + dy + H, sx + dx:sx + dx + W]
            if win.shape == (H, W) and haar_trace(win, detail_stages=0)["passed"] == len(haar_model()[3]):
                return win, len(raw)
    return fallback, len(raw)


def haar_trace(window, detail_stages=3):
    """Evaluate the cascade on one 24x24 window the way OpenCV's HaarEvaluator does:
    rectangle sums from the integral image, scaled by the window's standard deviation
    (variance normalisation over the inner 22x22), one stump per feature, leaf votes
    summed per stage, reject as soon as a stage sum falls below its threshold.

    Returns {"passed": n stages passed, "stages": [...]}; the first `detail_stages`
    stages also list every test (feature rectangles, value, threshold, vote)."""
    W, H, features, stages = haar_model()
    win = window.astype(np.float64)
    ii = cv2.integral(win)

    def rsum(x, y, w, h):
        return ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x]

    inner = win[1:H - 1, 1:W - 1]
    nf = inner.size * np.square(inner).sum() - inner.sum() ** 2
    nf = np.sqrt(nf) if nf > 0 else 1.0
    out, passed = [], 0
    for si, (stage_thr, weak) in enumerate(stages):
        total, tests = 0.0, []
        for fi, thr, left, right in weak:
            value = sum(wt * rsum(x, y, w, h) for x, y, w, h, wt in features[fi]) / nf
            vote = left if value < thr else right
            total += vote
            if si < detail_stages:
                tests.append({"rects": features[fi], "value": round(value, 4), "threshold": round(thr, 4),
                              "vote": round(vote, 4), "face_like": vote > 0})
        ok = total >= stage_thr
        out.append({"stage": si + 1, "tests": len(weak), "sum": round(total, 3), "threshold": round(stage_thr, 3),
                    "pass": bool(ok), "detail": tests})
        if not ok:
            break
        passed += 1
    return {"passed": passed, "total": len(stages), "window": W, "stages": out,
            "features_evaluated": sum(s["tests"] for s in out)}


# --------------------------------------------------------------------------- pipeline tiles
STAGE_NAMES = ["Camera", "Grey + blur", "CLAHE", "Face chip 200×200", "LBP codes", "Uniform LBP (59 bins)"]


def stage_tiles(engine, frame, face, size=160):
    """One image: the face region at each step of the pipeline, left to right."""
    x, y, w, h = face.box
    pad = int(0.25 * max(w, h))
    y0, y1, x0, x1 = max(y - pad, 0), y + h + pad, max(x - pad, 0), x + w + pad
    crop = lambda img: cv2.resize(img[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
    gray, clahe = engine.gray(frame), engine.detection_image(frame)
    chip = engine.describe(frame, face)
    codes = lbp.lbp_codes(chip, 8, 1)
    uni = (lbp.uniform_table(8)[codes] * (255 / 58)).astype(np.uint8)
    tiles = [crop(frame), crop(gray), crop(clahe), cv2.resize(chip, (size, size)),
             cv2.resize(codes.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST),
             cv2.resize(uni, (size, size), interpolation=cv2.INTER_NEAREST)]
    tiles = [t if t.ndim == 3 else cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) for t in tiles]
    return np.hstack(tiles), chip


# --------------------------------------------------------------------------- descriptor + race
def fingerprint(recognizer, chip, labels):
    """The (C, B) LBP descriptor of the chip, plus how far it is from every enrolled
    student (min over their templates) and from the closest background-crowd face."""
    desc = recognizer.desc(chip) if hasattr(recognizer, "desc") else lbp.LBPDescriptor(config.LBP_SCALES, config.LBP_GRID)(chip)
    out = {"cells": config.LBP_GRID[0] * config.LBP_GRID[1], "bins": int(desc.shape[1]),
           "hist": np.round(desc[:config.LBP_GRID[0] * config.LBP_GRID[1]], 4).tolist(),   # radius-1 plane
           "race": [], "crowd": None}
    if hasattr(recognizer, "identity_distances") and len(getattr(recognizer, "labels", [])):
        ids, d, _, _ = recognizer.identity_distances(desc)
        out["race"] = sorted(({"name": labels.get(int(i), {}).get("name", str(i)), "roll": labels.get(int(i), {}).get("roll"),
                               "d": round(float(v), 4)} for i, v in zip(ids, d)), key=lambda r: r["d"])
        if getattr(recognizer, "has_cohort", False):
            dc = lbp.weighted_distance(lbp.chi2_cells(desc, recognizer.cohort), recognizer.weights, config.TRIM_CELLS)
            out["crowd"] = round(float(dc.min()), 4)
    return out
