"""Face detection with two classical cascades that make different mistakes.

  haar       Viola-Jones cascade on Haar-like rectangle features (OpenCV frontal default)
  lbp        cascade on LBP features (OpenCV lbpcascade_frontalface_improved):
             integer comparisons instead of sums, ~4-6x faster, a different error profile
  union      every box either detector returns (highest recall, most false boxes)
  consensus  a box is kept when both cascades agree on it (IoU >= CONSENSUS_IOU), or when
             only one of them found it but the Haar eye cascade finds an eye pair in its
             upper half ("eye rescue"), or a Haar box has >= HAAR_STRONG_VOTES neighbours.
             Background false positives of the two feature families rarely coincide, so agreement removes most of them while the eye
             check keeps faces that only one detector saw.

Each returned Face carries `sources` ("H", "L", "HL", "H+eyes", ...) so the UI and the
benchmark can show which path accepted it.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import config

CASCADE_DIR = Path(__file__).resolve().parent / "cascades"


@dataclass
class Face:
    box: tuple                  # (x, y, w, h)
    sources: str = "H"          # which detector(s) produced / confirmed it
    eyes: tuple = field(default=None, repr=False)   # ((x, y), (x, y)) when found


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def _load(path):
    c = cv2.CascadeClassifier(str(path))
    if c.empty():
        raise RuntimeError(f"Could not load cascade {path}")
    return c


class Detector:
    MODES = ("haar", "lbp", "union", "consensus")

    def __init__(self, mode=None):
        self.mode = mode or config.DETECTOR
        if self.mode not in self.MODES:
            raise ValueError(f"DETECTOR must be one of {self.MODES}")
        self.haar = _load(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.lbp = _load(CASCADE_DIR / "lbpcascade_frontalface_improved.xml")
        self.eye = _load(cv2.data.haarcascades + "haarcascade_eye.xml")

    # ------------------------------------------------------------------ single cascades
    def _run(self, cascade, gray, min_neighbors):
        boxes = cascade.detectMultiScale(
            gray, scaleFactor=config.SCALE_FACTOR, minNeighbors=min_neighbors,
            minSize=config.MIN_FACE_SIZE)
        return [tuple(int(v) for v in b) for b in boxes]

    def run_haar(self, gray):
        return self._run(self.haar, gray, config.MIN_NEIGHBORS)

    def run_haar_scored(self, gray):
        """Haar boxes with their neighbour count: how many overlapping pyramid windows
        voted for the box. Real faces collect many votes, accidental matches few."""
        boxes, votes = self.haar.detectMultiScale2(
            gray, scaleFactor=config.SCALE_FACTOR, minNeighbors=config.MIN_NEIGHBORS,
            minSize=config.MIN_FACE_SIZE)
        return [(tuple(int(v) for v in b), int(n)) for b, n in zip(boxes, votes)]

    def run_lbp(self, gray):
        return self._run(self.lbp, gray, config.LBP_MIN_NEIGHBORS)

    # ------------------------------------------------------------------ eye check
    def find_eyes(self, gray, box):
        """Best plausible eye pair in the upper 60% of the box, image coords, left first."""
        x, y, w, h = box
        roi = gray[y:y + int(h * 0.6), x:x + w]
        if roi.size == 0:
            return None
        eyes = self.eye.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4,
                                         minSize=(w // 9, w // 9), maxSize=(w // 3, w // 3))
        centres = [(ex + ew / 2, ey + eh / 2) for ex, ey, ew, eh in eyes]
        best = None
        for l in (c for c in centres if c[0] < w / 2):
            for r in (c for c in centres if c[0] >= w / 2):
                dx, dy = r[0] - l[0], r[1] - l[1]
                if not 0.25 * w <= dx <= 0.65 * w:
                    continue
                if abs(np.degrees(np.arctan2(dy, dx))) > config.MAX_EYE_ANGLE:
                    continue
                if best is None or abs(dy) < best[0]:
                    best = (abs(dy), l, r)
        if best is None:
            return None
        return tuple((x + cx, y + cy) for cx, cy in best[1:])

    def count_eyes(self, gray, box):
        """Number of eye detections in the upper face (used by the blink check)."""
        x, y, w, h = box
        roi = gray[y + int(h * 0.15):y + int(h * 0.6), x:x + w]
        if roi.size == 0:
            return 0
        return len(self.eye.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3,
                                             minSize=(w // 9, w // 9), maxSize=(w // 3, w // 3)))

    # ------------------------------------------------------------------ fusion
    @staticmethod
    def _dedupe(boxes, thr=0.4):
        out = []
        for b in sorted(boxes, key=lambda b: -b[2] * b[3]):
            if all(iou(b, o) < thr for o in out):
                out.append(b)
        return out

    def detect(self, gray, mode=None):
        """gray: the enhanced (blur + gray + CLAHE) frame. Returns [Face]."""
        mode = mode or self.mode
        if mode == "haar":
            return [Face(b, "H") for b in self.run_haar(gray)]
        if mode == "lbp":
            return [Face(b, "L") for b in self.run_lbp(gray)]

        scored = self.run_haar_scored(gray)
        haar, lbpb = [b for b, _ in scored], self.run_lbp(gray)
        if mode == "union":
            faces = [Face(b, "H") for b in haar]
            faces += [Face(b, "L") for b in lbpb if all(iou(b, h) < config.CONSENSUS_IOU for h in haar)]
            return faces

        # consensus
        faces, used = [], set()
        for h, votes in scored:
            best = max(range(len(lbpb)), key=lambda i: iou(h, lbpb[i]), default=None)
            if best is not None and best not in used and iou(h, lbpb[best]) >= config.CONSENSUS_IOU:
                used.add(best)
                faces.append(Face(h, "HL"))         # keep the Haar box: same geometry as training
                continue
            if votes >= config.HAAR_STRONG_VOTES:
                faces.append(Face(h, "H*"))         # Haar alone, but with overwhelming support
                continue
            eyes = self.find_eyes(gray, h) if config.EYE_RESCUE else None
            if eyes is not None:
                faces.append(Face(h, "H+eyes", eyes))
        for i, b in enumerate(lbpb):
            if i in used:
                continue
            # LBP boxes are tighter than Haar's; grow to Haar-like framing before the eye check
            b = self._lbp_to_haar_box(b, gray.shape)
            if any(iou(b, f.box) >= config.CONSENSUS_IOU for f in faces):
                continue
            eyes = self.find_eyes(gray, b) if config.EYE_RESCUE else None
            if eyes is not None:
                faces.append(Face(b, "L+eyes", eyes))
        return faces

    @staticmethod
    def _lbp_to_haar_box(b, shape):
        """The improved LBP cascade frames the face more tightly than the Haar
        cascade; scale it so chips cut from either look alike."""
        x, y, w, h = b
        s = config.LBP_BOX_SCALE
        nw, nh = int(w * s), int(h * s)
        nx, ny = x - (nw - w) // 2, y - int((nh - h) * 0.6)
        H, W = shape[:2]
        nx, ny = max(nx, 0), max(ny, 0)
        nw, nh = min(nw, W - nx), min(nh, H - ny)
        return (nx, ny, nw, nh)
