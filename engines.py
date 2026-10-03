"""The image processing pipeline: detection, face normalisation and recognition.

    blur -> grayscale -> CLAHE -> Haar cascade -> crop (optional eye alignment)
    -> illumination normalisation -> (optional ellipse mask) -> LBPH

Every script goes through the same FaceEngine methods, so registration,
training, evaluation and live attendance all see identically processed faces.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class Face:
    box: tuple      # (x, y, w, h)


@dataclass
class Match:
    label: int      # id of the closest registered student
    score: float    # LBPH distance to that student (lower = closer)
    ok: bool        # distance is within LBPH_THRESHOLD


def tan_triggs(chip, gamma=0.2, sigma0=1.0, sigma1=2.0, alpha=0.1, tau=10.0):
    """Tan & Triggs illumination normalisation: gamma correction, difference of
    Gaussians (band-pass, removes shading gradients), then contrast equalisation."""
    x = np.power(chip.astype(np.float32) / 255.0 + 1e-6, gamma)
    dog = cv2.GaussianBlur(x, (0, 0), sigma0) - cv2.GaussianBlur(x, (0, 0), sigma1)
    dog /= np.mean(np.abs(dog) ** alpha) ** (1.0 / alpha) + 1e-6
    dog /= np.mean(np.minimum(np.abs(dog), tau) ** alpha) ** (1.0 / alpha) + 1e-6
    dog = tau * np.tanh(dog / tau)
    return cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def lbp_image(gray):
    """Basic 3x3 LBP code of every pixel (what LBPH histograms are built from).
    Used for visualisation only - recognition uses OpenCV's implementation."""
    g = gray.astype(np.int16)
    center = g[1:-1, 1:-1]
    codes = np.zeros(center.shape, dtype=np.uint8)
    # clockwise from the top-left neighbour
    offsets = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    for bit, (dy, dx) in enumerate(offsets):
        neighbour = g[dy:dy + center.shape[0], dx:dx + center.shape[1]]
        codes |= (neighbour >= center).astype(np.uint8) << (7 - bit)
    return codes


class FaceEngine:
    def __init__(self, align=None, illumination=None, mask=None, threshold=None):
        self.align = config.ALIGN_EYES if align is None else align
        self.illumination = config.ILLUMINATION if illumination is None else illumination
        self.mask = config.ELLIPSE_MASK if mask is None else mask
        self.threshold = config.LBPH_THRESHOLD if threshold is None else threshold

        self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self._clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP, tileGridSize=config.CLAHE_GRID)
        self._recognizer = None
        self._gray_cache = (None, None)

        w, h = config.FACE_SIZE
        self._ellipse = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(self._ellipse, (w // 2, h // 2), (int(w * 0.42), int(h * 0.52)), 0, 0, 360, 255, -1)

    # -- detection ---------------------------------------------------------
    def gray(self, frame):
        """Gaussian blur + grayscale, cached so detect() and describe() share one pass per frame."""
        if self._gray_cache[0] is not frame:
            blurred = cv2.GaussianBlur(frame, config.BLUR_KERNEL, 0)
            self._gray_cache = (frame, cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))
        return self._gray_cache[1]

    def detection_image(self, frame):
        return self._clahe.apply(self.gray(frame))

    def detect(self, frame):
        boxes = self._face_cascade.detectMultiScale(
            self.detection_image(frame),
            scaleFactor=config.SCALE_FACTOR,
            minNeighbors=config.MIN_NEIGHBORS,
            minSize=config.MIN_FACE_SIZE,
        )
        return [Face(tuple(int(v) for v in box)) for box in boxes]

    # -- chip normalisation ------------------------------------------------
    def find_eyes(self, gray, box):
        """Returns ((x, y), (x, y)) eye centres in image coordinates, left one first, or None."""
        x, y, w, h = box
        roi = gray[y:y + int(h * 0.6), x:x + w]     # eyes are in the upper part of the face
        eyes = self._eye_cascade.detectMultiScale(
            roi, scaleFactor=1.1, minNeighbors=5, minSize=(w // 8, w // 8), maxSize=(w // 3, w // 3))
        centres = [(ex + ew / 2, ey + eh / 2) for ex, ey, ew, eh in eyes]
        best = None
        for left in (c for c in centres if c[0] < w / 2):
            for right in (c for c in centres if c[0] >= w / 2):
                dx, dy = right[0] - left[0], right[1] - left[1]
                if not 0.25 * w <= dx <= 0.65 * w:
                    continue
                if abs(math.degrees(math.atan2(dy, dx))) > config.MAX_EYE_ANGLE:
                    continue
                if best is None or abs(dy) < best[0]:
                    best = (abs(dy), left, right)
        if best is None:
            return None
        return tuple((x + cx, y + cy) for cx, cy in best[1:])

    def align_to_eyes(self, gray, eyes):
        """Similarity transform (rotate + scale + shift) putting the eyes on EYE_POSITIONS."""
        (lx, ly), (rx, ry) = eyes
        w, h = config.FACE_SIZE
        (tlx, tly), (trx, _) = config.EYE_POSITIONS
        angle = math.degrees(math.atan2(ry - ly, rx - lx))
        scale = (trx - tlx) * w / math.hypot(rx - lx, ry - ly)
        mid = ((lx + rx) / 2, (ly + ry) / 2)
        m = cv2.getRotationMatrix2D(mid, angle, scale)
        m[0, 2] += (tlx + trx) / 2 * w - mid[0]
        m[1, 2] += tly * h - mid[1]
        return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def normalise(self, chip):
        if self.illumination == "clahe":
            chip = self._clahe.apply(chip)
        elif self.illumination == "tantriggs":
            chip = tan_triggs(chip)
        if self.mask:
            chip = cv2.bitwise_and(chip, self._ellipse)
        return chip

    def raw_chip(self, frame, face):
        """Geometric normalisation only (crop or eye alignment), before illumination + mask."""
        gray = self.gray(frame)
        eyes = self.find_eyes(gray, face.box) if self.align else None
        if eyes is not None:
            return self.align_to_eyes(gray, eyes)
        x, y, w, h = face.box
        return cv2.resize(gray[y:y + h, x:x + w], config.FACE_SIZE)

    def describe(self, frame, face):
        """What the recognizer consumes: the normalised 200x200 face chip."""
        return self.normalise(self.raw_chip(frame, face))

    # -- recognition -------------------------------------------------------
    def train(self, descriptors, label_ids):
        gx, gy = config.LBPH_GRID
        self._recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=config.LBPH_RADIUS, neighbors=config.LBPH_NEIGHBORS, grid_x=gx, grid_y=gy)
        self._recognizer.train(descriptors, np.array(label_ids, dtype=np.int32))

    def predict(self, descriptor):
        label, distance = self._recognizer.predict(descriptor)
        return Match(label, distance, distance <= self.threshold)

    def save(self):
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._recognizer.write(str(config.MODEL_PATH))

    def load(self):
        if not config.MODEL_PATH.exists():
            return False
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._recognizer.read(str(config.MODEL_PATH))
        return True
