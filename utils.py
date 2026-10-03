"""Shared helpers: camera, frame preparation, dataset layout, face quality, drawing."""
import sys

import cv2

import config

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".pgm"}
GREEN, RED, YELLOW, WHITE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)


def open_camera():
    # DirectShow opens much faster than the default backend on Windows
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {config.CAMERA_INDEX}. "
            "Close other apps using the webcam or change CAMERA_INDEX in config.py."
        )
    return cap


def prepare_frame(frame, mirror=None):
    """Resize a camera frame to FRAME_WIDTH (keeping aspect) and mirror it."""
    h, w = frame.shape[:2]
    scale = config.FRAME_WIDTH / w
    frame = cv2.resize(frame, (config.FRAME_WIDTH, int(h * scale)))
    if config.MIRROR if mirror is None else mirror:
        frame = cv2.flip(frame, 1)
    return frame


def prepare_photo(image):
    """Photos from disk: shrink if wider than FRAME_WIDTH, never enlarge or mirror."""
    if image.shape[1] > config.FRAME_WIDTH:
        return prepare_frame(image, mirror=False)
    return image


def largest_face(faces):
    return max(faces, key=lambda f: f.box[2] * f.box[3]) if faces else None


def padded_crop(frame, box):
    """The face plus CROP_PADDING of context on every side, clipped to the frame.
    This is what gets stored in dataset/: training re-detects and normalises the
    face from it, so normalisation settings can change without re-registering."""
    x, y, w, h = box
    pad = int(config.CROP_PADDING * max(w, h))
    return frame[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]


def face_quality(frame, box):
    """Returns (sharpness, brightness) of a face: variance of the Laplacian
    (few edges = blurry = low variance) and mean gray level."""
    x, y, w, h = box
    gray = cv2.cvtColor(cv2.resize(frame[y:y + h, x:x + w], config.FACE_SIZE), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var(), gray.mean()


def quality_problem(frame, box):
    """Why this face should not be used for registration, or None if it is fine."""
    sharpness, brightness = face_quality(frame, box)
    if sharpness < config.MIN_SHARPNESS:
        return "Too blurry - hold still / move closer"
    if brightness < config.MIN_BRIGHTNESS:
        return "Too dark"
    if brightness > config.MAX_BRIGHTNESS:
        return "Too bright"
    return None


def student_dirs():
    """Sorted dataset/<roll>_<name> folders."""
    if not config.DATASET_DIR.exists():
        return []
    return sorted(p for p in config.DATASET_DIR.iterdir() if p.is_dir())


def parse_student_dir(dirname):
    """'<roll>_<name>' -> (roll, name)"""
    roll, _, name = dirname.partition("_")
    return roll, name


def image_files(directory):
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def draw_label(frame, box, text, color):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, text, (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_hud(frame, text, line=0, color=WHITE):
    origin = (10, 25 + 25 * line)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)   # outline
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def pressed_key(window):
    """Key pressed this frame as a character ('' if none); 'q' if the window was closed."""
    key = cv2.waitKey(1) & 0xFF
    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
        return "q"
    return chr(key) if key != 255 else ""
