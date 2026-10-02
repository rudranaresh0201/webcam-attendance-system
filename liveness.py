"""Head-turn liveness check against printed-photo / phone-photo spoofing.

A real head is 3-D: the nose tip sticks out in front of the eyes, so when the
head turns, the nose slides sideways relative to the midpoint between the eyes.
A photo is flat: eyes and nose lie in one plane, so however the photo is
rotated or tilted the nose stays (almost exactly) centred between the eyes.
Asking for a turn in a random direction and measuring that shift therefore
separates a live face from a picture of one.

It does not stop a replayed video of the person turning their head.
"""
import random

import numpy as np

import config


def yaw_ratio(landmarks):
    """Offset of the nose tip from the eye midpoint, measured along the eye
    line and divided by the distance between the eyes (so it is independent of
    face size and of head tilt). About 0 when facing the camera; positive when
    the nose points towards the right of the image, negative towards the left.
    """
    a, b, nose = landmarks[0], landmarks[1], landmarks[2]
    if a[0] > b[0]:
        a, b = b, a
    axis = b - a
    length_sq = float(np.dot(axis, axis))
    if length_sq < 1e-6:
        return 0.0
    return float(np.dot(nose - (a + b) / 2, axis) / length_sq)


class Challenge:
    """One 'turn your head this way' request for one tracked face."""

    def __init__(self, now, direction=None):
        self.direction = direction or random.choice((-1, 1))    # -1 = image left, +1 = image right
        self.started = now

    @property
    def prompt(self):
        return "<-- turn your head" if self.direction < 0 else "turn your head -->"

    def update(self, yaw, identity_ok, now):
        """Returns 'passed', 'failed' or 'pending'.

        Passing needs the turned face to be recognised as the same student in
        the very same frame - otherwise someone could show a photo to get
        recognised and then turn their own head in its place.
        """
        if yaw * self.direction >= config.LIVENESS_TURN and identity_ok:
            return "passed"
        if now - self.started > config.LIVENESS_TIMEOUT:
            return "failed"
        return "pending"
