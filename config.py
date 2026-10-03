"""All tunable parameters in one place."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"      # dataset/<roll>_<name>/0.png ... 29.png
MODELS_DIR = BASE_DIR / "models"
RECORDS_DIR = BASE_DIR / "records"      # records/attendance_YYYY-MM-DD.csv
DOCS_DIR = BASE_DIR / "docs"            # figures saved by visualize.py
MODEL_PATH = MODELS_DIR / "lbph_model.yml"      # trained recognizer
LABELS_PATH = MODELS_DIR / "labels.json"        # {int_id: {"roll": ..., "name": ...}}

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
MIRROR = True                    # flip horizontally so the preview behaves like a mirror

# Detection
MIN_FACE_SIZE = (80, 80)         # ignore tiny / far-away faces
BLUR_KERNEL = (3, 3)             # Gaussian blur before grayscale
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
SCALE_FACTOR = 1.1               # Haar image pyramid step
MIN_NEIGHBORS = 6                # higher = fewer spurious Haar boxes

# Face chip normalisation (applied to every face before LBPH).
# ALIGN_EYES and ELLIPSE_MASK are off because neither improved accuracy on LFW
# (the Haar eye detector only finds both eyes in about half the faces).
# Re-test them on webcam data with evaluate.py before turning them on.
FACE_SIZE = (200, 200)
ALIGN_EYES = False               # rotate/scale so both eyes land on fixed positions
EYE_POSITIONS = ((0.30, 0.38), (0.70, 0.38))   # where the eyes go, as a fraction of FACE_SIZE
MAX_EYE_ANGLE = 20               # degrees; a steeper eye line is treated as a bad eye detection
ILLUMINATION = "clahe"           # "clahe", "tantriggs" or "none"
ELLIPSE_MASK = False             # black out hair/background in the corners of the chip
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID = (8, 8)

# Registration
IMAGES_PER_STUDENT = 30
SAVE_EVERY_N_FRAMES = 3          # spread the captures out so they are not near-duplicates
CROP_PADDING = 0.5               # context kept around the face box in saved images
MIN_SHARPNESS = 40               # variance of Laplacian on the face; below this = too blurry
MIN_BRIGHTNESS = 50              # mean gray level of the face
MAX_BRIGHTNESS = 205
MIN_IMAGES_TO_TRAIN = 10

# Recognition
LBPH_THRESHOLD = 70              # LBPH distance, lower = stricter. Above this -> Unknown
CONFIRM_FRAMES = 5               # consecutive matching frames before a face is accepted

# Tracking (follows each face from frame to frame)
TRACK_IOU = 0.3                  # minimum box overlap to count as the same face
TRACK_MAX_MISSES = 5             # frames a face may go undetected before its track is dropped
