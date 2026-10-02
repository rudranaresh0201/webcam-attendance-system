"""Follow each face from frame to frame by bounding-box overlap (IoU).

Per-face state (who it has been recognised as, for how many frames in a row,
any liveness challenge in progress) lives on the Track, so two people in view
never share a counter.
"""
import config


def iou(a, b):
    """Intersection over union of two (x, y, w, h) boxes."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


class Track:
    def __init__(self, track_id, face):
        self.id = track_id
        self.face = face
        self.misses = 0
        self.label = None           # student currently being recognised on this track
        self.streak = 0             # consecutive frames recognised as self.label
        self.challenge = None       # liveness.Challenge in progress
        self.blocked_until = 0.0    # after a failed liveness check

    def observe(self, match):
        """Update the consecutive-frame counter with this frame's recognition result."""
        if not match.ok:
            self.label, self.streak = None, 0
        elif match.label == self.label:
            self.streak += 1
        else:
            self.label, self.streak = match.label, 1

    @property
    def confirmed(self):
        return self.label is not None and self.streak >= config.CONFIRM_FRAMES


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
