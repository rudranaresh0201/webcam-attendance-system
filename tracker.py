"""Follow each face from frame to frame by bounding-box overlap (IoU).

Per-face state lives on the Track, so two people in view never share a counter:
  * recognition vote: the accepted identity in each of the last CONFIRM_WINDOW frames; a
    student is confirmed once they win CONFIRM_FRAMES of them (5 of the last 8 by default),
    so one bad frame (a blink, a head turn) does not throw the evidence away
  * blink liveness: a printed photo or a phone screen held up to the camera does not blink.
    The Haar eye cascade only fires on OPEN eyes, so the eye count per frame gives an
    open / closed signal; a blink is  open (>= BLINK_MIN_OPEN frames) -> closed
    (1..BLINK_MAX_CLOSED frames) -> open again.
"""
from collections import Counter, deque

import config
from detector import iou  # noqa: F401  (re-exported)


class Track:
    def __init__(self, track_id, face):
        self.id = track_id
        self.face = face
        self.misses = 0
        self.label = None           # student currently leading the vote on this track
        self.streak = 0             # how many of the recent frames voted for self.label
        self.votes = deque(maxlen=max(config.CONFIRM_WINDOW, config.CONFIRM_FRAMES))
        self.match = None           # last Match
        self.age = 0
        # blink state machine
        self.blinked = False
        self._open_run = 0
        self._closed_run = 0
        self._armed = False         # saw enough open frames, waiting for a closure

    def observe(self, match):
        """Update the consecutive-frame counter with this frame's recognition result."""
        self.match = match
        self.age += 1
        self.votes.append(match.label if match.ok else None)
        counts = Counter(v for v in self.votes if v is not None)
        if counts:
            self.label, self.streak = counts.most_common(1)[0]
        else:
            self.label, self.streak = None, 0

    def observe_eyes(self, n_eyes):
        if n_eyes >= 1:                                   # eyes open
            if self._closed_run and self._armed and self._closed_run <= config.BLINK_MAX_CLOSED:
                self.blinked = True
            self._closed_run = 0
            self._open_run += 1
            if self._open_run >= config.BLINK_MIN_OPEN:
                self._armed = True
        else:                                             # eyes closed (or not found)
            self._closed_run += 1
            if self._closed_run > config.BLINK_MAX_CLOSED:
                self._armed = False
            self._open_run = 0

    @property
    def confirmed(self):
        return self.label is not None and self.streak >= config.CONFIRM_FRAMES

    @property
    def live(self):
        return self.blinked or not config.LIVENESS


class Tracker:
    def __init__(self):
        self.tracks = []
        self._next_id = 0

    def update(self, faces):
        """Match this frame's faces to existing tracks. Returns the tracks seen in this frame."""
        pairs = sorted(
            ((iou(t.face.box, f.box), ti, fi) for ti, t in enumerate(self.tracks) for fi, f in enumerate(faces)),
            reverse=True,
        )
        matched_tracks, matched_faces = set(), set()
        for overlap, ti, fi in pairs:
            if overlap < config.TRACK_IOU:
                break
            if ti in matched_tracks or fi in matched_faces:
                continue
            self.tracks[ti].face = faces[fi]
            self.tracks[ti].misses = 0
            matched_tracks.add(ti)
            matched_faces.add(fi)

        seen = [self.tracks[ti] for ti in sorted(matched_tracks)]
        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= config.TRACK_MAX_MISSES]

        for fi, face in enumerate(faces):
            if fi not in matched_faces:
                track = Track(self._next_id, face)
                self._next_id += 1
                self.tracks.append(track)
                seen.append(track)
        return seen
