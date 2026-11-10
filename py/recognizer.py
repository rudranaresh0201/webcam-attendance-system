"""Face recognition from LBP histograms, as an open-set problem.

Two back-ends with the same interface (train / predict / save / load):

  LBPHRecognizer   OpenCV's cv2.face LBPH (basic 256-bin LBP, unweighted chi-square,
                   absolute distance threshold). Kept as the baseline from stage 1.

  OpenSetLBP       our implementation (lbp.py):
                   * uniform, optionally multi-scale LBP or LTP histograms
                   * per-cell weights learned from data (lbp.learn_cell_weights)
                   * per-identity distance = min over that person's templates
                   * decision uses TWO tests:
                       d1 <= DIST_THRESHOLD           (close enough to someone)
                       d1 / d2 <= RATIO_THRESHOLD     (clearly closer to them than to
                                                       the runner-up identity)
                     The ratio test (Lowe 2004, used for SIFT matching) works because a
                     stranger tends to be roughly equally far from everybody, while a
                     registered student is much closer to themselves than to anyone
                     else. It does not depend on the absolute scale of the distance,
                     which shifts with lighting and camera.
"""
import json
from dataclasses import dataclass, field

import cv2
import numpy as np

import config
import lbp


@dataclass
class Match:
    label: int              # closest registered identity (-1 if gallery empty)
    score: float            # distance to it (lower = closer)
    ok: bool                # accepted as that identity
    ratio: float = 1.0      # d1 / d2 (1.0 when there is no runner-up)
    second: int = -1        # runner-up identity
    cells: np.ndarray = field(default=None, repr=False)   # per-cell distances to best template


def make_descriptor():
    return lbp.LBPDescriptor(scales=config.LBP_SCALES, grid=config.LBP_GRID,
                             uniform=True, ternary=config.LBP_TERNARY, ltp_t=config.LTP_T)


def default_weights(desc):
    """Shipped cell weights learned on LFW identities that are never used for
    testing (see benchmark.py). Falls back to uniform weights."""
    path = config.WEIGHTS_DIR / f"cell_weights_{desc.signature}.npy"
    if config.CELL_WEIGHTS == "learned" and path.exists():
        w = np.load(path)
        if w.shape == (desc.n_cells,):
            return w
    return np.ones(desc.n_cells, np.float32)


CROWD = -2     # Match.second when the runner-up is a background face, not a student


def load_cohort(desc):
    """Background faces (texture histograms of LFW people who are never students and
    never in any test; see benchmark.make_cohort_assets). Returns (features, person ids)
    or (None, None) when switched off or not available for these settings."""
    if not config.USE_COHORT:
        return None, None
    path = config.WEIGHTS_DIR / f"cohort_{desc.signature}_{config.ILLUMINATION}.npz"
    if not path.exists():
        return None, None
    z = np.load(path)
    return z["features"].astype(np.float32), z["people"]


class OpenSetLBP:
    """Open-set matcher. The gallery is the enrolled students PLUS a silent background
    crowd: a student is accepted only if they are the closest identity of all, close
    enough (d1 <= DIST_THRESHOLD) and clearly closer than the runner-up, student or
    crowd (d1 / d2 <= RATIO_THRESHOLD). Without the crowd, a class of one student would
    accept most strangers: every face is 'closest' to the only student (RESULTS.md 2d)."""
    kind = "openset"

    def __init__(self, dist_threshold=None, ratio_threshold=None, weights=None):
        self.desc = make_descriptor()
        self.dist_threshold = config.DIST_THRESHOLD if dist_threshold is None else dist_threshold
        self.ratio_threshold = config.RATIO_THRESHOLD if ratio_threshold is None else ratio_threshold
        self.weights = default_weights(self.desc) if weights is None else weights
        self.features = np.zeros((0, self.desc.n_cells, 1), np.float32)
        self.labels = np.zeros(0, np.int32)
        self.cohort, self.cohort_people = load_cohort(self.desc)

    @property
    def has_cohort(self):
        return self.cohort is not None

    @property
    def threshold(self):
        return self.dist_threshold

    def features_of(self, chips):
        return np.stack([self.desc(c) for c in chips]) if len(chips) else self.features[:0]

    def train(self, chips, label_ids):
        labels = np.asarray(label_ids, np.int32)
        if config.MIRROR_ENROL and len(chips):
            # faces are nearly symmetric: each enrolment photo also counts mirrored,
            # which covers the head turned the other way (RESULTS.md, robustness)
            chips = list(chips) + [cv2.flip(c, 1) for c in chips]
            labels = np.concatenate([labels, labels])
        self.features = self.features_of(chips)
        self.labels = labels

    def identity_distances(self, feature):
        """(ids, per-identity min distance, index of best template per identity)"""
        cell_d = lbp.chi2_cells(feature, self.features)
        d = lbp.weighted_distance(cell_d, self.weights, config.TRIM_CELLS)
        ids = np.unique(self.labels)
        best_idx = np.array([np.flatnonzero(self.labels == i)[np.argmin(d[self.labels == i])] for i in ids])
        return ids, d[best_idx], best_idx, cell_d

    def predict(self, chip):
        if not len(self.labels):
            return Match(-1, float("inf"), False)
        feature = self.desc(chip)
        ids, d, best_idx, cell_d = self.identity_distances(feature)
        order = np.argsort(d)
        d1 = float(d[order[0]])
        # runner-up: the next student, or the closest background face, whichever is nearer
        d2, second = (float(d[order[1]]), int(ids[order[1]])) if len(ids) > 1 else (float("inf"), -1)
        if self.has_cohort:
            dc = lbp.weighted_distance(lbp.chi2_cells(feature, self.cohort), self.weights, config.TRIM_CELLS)
            if dc.min() < d2:
                d2, second = float(dc.min()), CROWD
        ratio = d1 / d2 if np.isfinite(d2) else 0.0      # 0 = nothing to compare with
        ok = d1 <= self.dist_threshold and ratio <= self.ratio_threshold
        return Match(int(ids[order[0]]), d1, ok, ratio, second, cell_d[best_idx[order[0]]])

    def save(self, path=None):
        path = path or config.OPENSET_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=self.features, labels=self.labels, weights=self.weights,
                            signature=self.desc.signature)

    def load(self, path=None):
        path = path or config.OPENSET_MODEL_PATH
        if not path.exists():
            return False
        data = np.load(path)
        if str(data["signature"]) != self.desc.signature:
            return False        # descriptor settings changed: retrain
        self.features, self.labels, self.weights = data["features"], data["labels"], data["weights"]
        return True


class LBPHRecognizer:
    """OpenCV LBPH baseline (stage-1 system)."""
    kind = "lbph"

    def __init__(self, threshold=None):
        self.dist_threshold = config.LBPH_THRESHOLD if threshold is None else threshold
        self._r = None

    @property
    def threshold(self):
        return self.dist_threshold

    def train(self, chips, label_ids):
        gx, gy = config.LBPH_GRID
        self._r = cv2.face.LBPHFaceRecognizer_create(
            radius=config.LBPH_RADIUS, neighbors=config.LBPH_NEIGHBORS, grid_x=gx, grid_y=gy)
        self._r.train(list(chips), np.array(label_ids, dtype=np.int32))

    def predict(self, chip):
        label, distance = self._r.predict(chip)
        return Match(int(label), float(distance), distance <= self.dist_threshold)

    def save(self, path=None):
        path = path or config.MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self._r.write(str(path))

    def load(self, path=None):
        path = path or config.MODEL_PATH
        if not path.exists():
            return False
        self._r = cv2.face.LBPHFaceRecognizer_create()
        self._r.read(str(path))
        return True


def make_recognizer(kind=None):
    kind = kind or config.RECOGNIZER
    return OpenSetLBP() if kind == "openset" else LBPHRecognizer()


def save_labels(labels):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.LABELS_PATH, "w") as f:
        json.dump({str(i): {"roll": s["roll"], "name": s["name"]} for i, s in labels.items()}, f, indent=2)


def load_labels():
    if not config.LABELS_PATH.exists():
        return {}
    with open(config.LABELS_PATH) as f:
        return {int(i): s for i, s in json.load(f).items()}
