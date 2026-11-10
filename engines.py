"""The image processing pipeline shared by every script and the web app.

    frame -> Gaussian blur -> grayscale -> CLAHE -> face detection (detector.py)
          -> crop / optional eye alignment -> illumination normalisation
          -> optional ellipse mask -> LBP features + open-set matching (recognizer.py)

Registration, training, evaluation, benchmarking and live attendance all go through
the same FaceEngine methods, so every face is processed identically.
"""
import math

import cv2
import numpy as np

import config
from detector import Detector, Face  # noqa: F401  (Face re-exported for older imports)
from recognizer import Match, make_recognizer  # noqa: F401


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
    """Basic 3x3 LBP code image, for visualisation."""
    import lbp
    return lbp.lbp_codes(gray, 8, 1).astype(np.uint8)


class FaceEngine:
    def __init__(self, align=None, illumination=None, mask=None, threshold=None,
                 detector=None, recognizer=None):
        self.align = config.ALIGN_EYES if align is None else align
        self.illumination = config.ILLUMINATION if illumination is None else illumination
        self.mask = config.ELLIPSE_MASK if mask is None else mask
        self.detector = Detector(detector)
        self.recognizer = make_recognizer(recognizer)
        if threshold is not None:
            self.recognizer.dist_threshold = threshold
        self._clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP, tileGridSize=config.CLAHE_GRID)
        self._cache = (None, None, None)

        w, h = config.FACE_SIZE
        self._ellipse = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(self._ellipse, (w // 2, h // 2), (int(w * 0.42), int(h * 0.52)), 0, 0, 360, 255, -1)

    @property
    def threshold(self):
        return self.recognizer.threshold

    # -- frame preparation (cached per frame: detect() and describe() share one pass)
    def _prepare(self, frame):
        if self._cache[0] is not frame:
            blurred = cv2.GaussianBlur(frame, config.BLUR_KERNEL, 0)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY) if blurred.ndim == 3 else blurred
            self._cache = (frame, gray, self._clahe.apply(gray))
        return self._cache

    def gray(self, frame):
        return self._prepare(frame)[1]

    def detection_image(self, frame):
        return self._prepare(frame)[2]

    def detect(self, frame, mode=None):
        return self.detector.detect(self.detection_image(frame), mode)

    # -- chip normalisation ---------------------------------------------------
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
        """Geometric normalisation only (crop or eye alignment)."""
        gray = self.gray(frame)
        eyes = None
        if self.align:
            eyes = face.eyes or self.detector.find_eyes(gray, face.box)
        if eyes is not None:
            return self.align_to_eyes(gray, eyes)
        x, y, w, h = face.box
        return cv2.resize(gray[y:y + h, x:x + w], config.FACE_SIZE)

    def describe(self, frame, face):
        """What the recognizer consumes: the normalised 200x200 face chip."""
        return self.normalise(self.raw_chip(frame, face))

    # -- recognition -----------------------------------------------------------
    def train(self, chips, label_ids):
        self.recognizer.train(chips, label_ids)

    def predict(self, chip):
        return self.recognizer.predict(chip)

    def save(self):
        self.recognizer.save()

    def load(self):
        return self.recognizer.load()
