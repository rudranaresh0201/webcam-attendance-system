"""All tunable parameters in one place.

Values marked (benchmark) were chosen from the experiments in benchmark.py;
see docs/RESULTS.md for the numbers behind them.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"      # dataset/<roll>_<name>/0.png ... 29.png
MODELS_DIR = BASE_DIR / "models"
RECORDS_DIR = BASE_DIR / "records"      # records/attendance_YYYY-MM-DD.csv
DOCS_DIR = BASE_DIR / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
RESULTS_DIR = DOCS_DIR / "results"
WEIGHTS_DIR = BASE_DIR / "assets"       # learned LBP cell weights (committed)
MODEL_PATH = MODELS_DIR / "lbph_model.yml"          # OpenCV LBPH baseline
OPENSET_MODEL_PATH = MODELS_DIR / "openset_lbp.npz" # our recognizer's gallery
LABELS_PATH = MODELS_DIR / "labels.json"            # {int_id: {"roll": ..., "name": ...}}

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
MIRROR = True                    # flip horizontally so the preview behaves like a mirror

# Preprocessing
BLUR_KERNEL = (3, 3)             # Gaussian blur before grayscale
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

# Detection
DETECTOR = "consensus"           # "haar", "lbp", "union" or "consensus" (see detector.py)
MIN_FACE_SIZE = (80, 80)         # ignore tiny / far-away faces
SCALE_FACTOR = 1.1               # image pyramid step
MIN_NEIGHBORS = 6                # Haar: higher = fewer spurious boxes
LBP_MIN_NEIGHBORS = 4            # LBP cascade
CONSENSUS_IOU = 0.3              # Haar and LBP boxes this similar count as agreement
EYE_RESCUE = True                # single-detector box survives if an eye pair is found in it
LBP_BOX_SCALE = 1.37             # LBP cascade boxes are tighter than Haar's (benchmark)
HAAR_STRONG_VOTES = 20           # a Haar box this well supported needs no second opinion (benchmark)
MAX_EYE_ANGLE = 20               # degrees; steeper eye line = bad eye detection

# Face chip normalisation
FACE_SIZE = (200, 200)
ALIGN_EYES = False               # rotate/scale so both eyes land on EYE_POSITIONS
EYE_POSITIONS = ((0.30, 0.38), (0.70, 0.38))
ILLUMINATION = "clahe"           # "clahe", "tantriggs" or "none"
ELLIPSE_MASK = False             # black out hair/background in the chip corners

# Recognition
RECOGNIZER = "openset"           # "openset" (lbp.py + ratio test) or "lbph" (OpenCV baseline)
LBP_SCALES = ((8, 1), (8, 2))    # (points, radius) per scale: multi-scale uniform LBP
LBP_GRID = (8, 8)
LBP_TERNARY = False              # Local Ternary Patterns instead of LBP
LTP_T = 5                        # LTP tolerance in gray levels
CELL_WEIGHTS = "learned"         # "learned" (assets/) or "uniform"
DIST_THRESHOLD = 0.38            # weighted chi-square per cell, in [0, 2] (benchmark: LFW, FAR <= 1%)
RATIO_THRESHOLD = 0.88           # d(best) / d(runner-up identity) (benchmark: LFW, FAR <= 1%)

# OpenCV LBPH baseline
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID = (8, 8)
LBPH_THRESHOLD = 70              # LBPH distance, lower = stricter

# Registration
IMAGES_PER_STUDENT = 30
SAVE_EVERY_N_FRAMES = 3          # spread the captures out so they are not near-duplicates
CROP_PADDING = 0.5               # context kept around the face box in saved images
MIN_SHARPNESS = 40               # variance of Laplacian on the face; below = too blurry
MIN_BRIGHTNESS = 50              # mean gray level of the face
MAX_BRIGHTNESS = 205
MIN_IMAGES_TO_TRAIN = 10

# Attendance decision
CONFIRM_FRAMES = 5               # consecutive accepted frames before a face is credited
LIVENESS = False                 # also require a blink (Haar eye cascade) before crediting
BLINK_MIN_OPEN = 2               # frames with eyes visible before / after the blink
BLINK_MAX_CLOSED = 8             # a closed-eye run longer than this is not a blink

# Tracking
TRACK_IOU = 0.3
TRACK_MAX_MISSES = 10            # frames a face may go undetected before its track ends
