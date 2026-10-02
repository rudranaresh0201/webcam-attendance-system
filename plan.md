# Face Detection Attendance System — plan.md

Software-only attendance system using the laptop webcam. Students register their face once, then the system detects and recognizes faces live and marks present/absent in a CSV.

## Goal
- Register students via webcam (roll no + name + face images)
- Train a recognizer on registered faces
- Run live recognition, mark each recognized student present once per day
- At session end, mark all unrecognized registered students absent
- Show "Unknown" for unregistered faces instead of forcing a match

## Stack
- Python 3.10+
- `opencv-contrib-python` (webcam, preprocessing, Haar cascade detection, LBPH recognizer). Use contrib, not plain `opencv-python`, since LBPH lives in `cv2.face`.
- `numpy`, `pandas`
- No dlib / face_recognition (avoids Windows install pain). Pure OpenCV pipeline.

```
pip install opencv-contrib-python numpy pandas
```

## Project Structure
```
attendance/
├── dataset/                 # dataset/<roll>_<name>/0.jpg ... 29.jpg
├── models/
│   ├── lbph_model.yml       # trained recognizer
│   └── labels.json          # {int_id: {"roll": ..., "name": ...}}
├── attendance/
│   └── attendance_YYYY-MM-DD.csv
├── config.py                # all tunable params in one place
├── utils.py                 # shared preprocessing + detection
├── register.py
├── train.py
├── attendance.py
├── requirements.txt
└── README.md
```

## Pipeline (maps to case-study terms)
| Stage | Implementation |
|---|---|
| Image acquisition | `cv2.VideoCapture(0)` laptop webcam |
| Preprocessing | resize, Gaussian blur (3x3), CLAHE for lighting |
| Grayscale conversion | `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` |
| Face detection | Haar cascade `haarcascade_frontalface_default.xml` via `detectMultiScale` |
| Feature extraction | LBP histograms (inside LBPH recognizer) |
| Classification | LBPH nearest match + confidence threshold |
| Output | CSV attendance with timestamp |

## config.py
```
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FACE_SIZE = (200, 200)          # all face crops resized to this
IMAGES_PER_STUDENT = 30
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 6
MIN_FACE_SIZE = (80, 80)         # ignore tiny detections
LBPH_THRESHOLD = 70              # LBPH distance; lower = stricter. Above this -> Unknown
CONFIRM_FRAMES = 5               # consecutive matching frames before marking present
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
```

## utils.py
- `preprocess(frame) -> gray`: resize to FRAME_WIDTH (keep aspect), Gaussian blur, grayscale, CLAHE. Return gray image.
- `detect_faces(gray) -> list[(x,y,w,h)]`: Haar cascade with config params. Load cascade from `cv2.data.haarcascades`.
- `crop_face(gray, box) -> face`: crop + resize to FACE_SIZE.
- `largest_face(boxes)`: helper for registration (only use the biggest face).

## register.py
1. Prompt in terminal for roll number and name. Reject if `dataset/<roll>_*` already exists (offer overwrite y/n).
2. Open webcam. Each frame: preprocess, detect faces.
3. If exactly one face (or take largest), crop and save every ~3rd frame until IMAGES_PER_STUDENT saved.
4. Show live preview with box + counter "12/30". On-screen hint: "Turn head slightly, vary expression".
5. `q` aborts. Release camera, close windows.

## train.py
1. Walk `dataset/`. Each folder -> integer label id. Build `labels.json` mapping id -> {roll, name}.
2. Load each image as grayscale, ensure FACE_SIZE.
3. `cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)`, train, save to `models/lbph_model.yml`.
4. Print students trained + image counts. Error clearly if dataset is empty or a student has < 10 images.

## attendance.py
1. Load model + labels. Error if missing ("run train.py first").
2. Load or create today's `attendance/attendance_YYYY-MM-DD.csv` with columns: `roll, name, status, time`.
3. Webcam loop, each frame:
   - preprocess, detect faces (apply MIN_FACE_SIZE)
   - for each face: crop, `predict()` -> (label, distance)
   - distance <= LBPH_THRESHOLD -> candidate match, else "Unknown"
   - track per-label consecutive-frame counter; mark present only after CONFIRM_FRAMES
   - draw green box + name (+ distance) for matches, red box "Unknown" otherwise
   - already-marked students show "Marked" and are not re-logged
4. On `q`: for every student in labels.json not marked present, write status `Absent`. Save CSV. Print summary (present count / total).

## False-Detection Reduction (implement + mention in report)
- `MIN_NEIGHBORS = 6`: fewer spurious Haar boxes
- `MIN_FACE_SIZE`: drop small background detections
- `LBPH_THRESHOLD`: unknown faces shown as Unknown, never forced to a student
- `CONFIRM_FRAMES`: needs a stable match across consecutive frames, kills one-frame misfires
- CLAHE: consistent contrast across lighting
- 30 varied registration images per student

## Acceptance Checks
- Register 2–3 people, train, run attendance: each is recognized and logged once with timestamp
- A non-registered person shows "Unknown" and is not logged
- Registered person who never appears is marked Absent on quit
- Re-running on same day appends to / respects existing CSV without duplicates
- Runs at usable real-time FPS on a normal laptop

## Optional Upgrades (only if time permits)
- Simple Tkinter GUI with Register / Train / Start buttons
- Swap LBPH for `face_recognition` 128-D embeddings for better accuracy
- Blink-based liveness check to block photo spoofing

## Report Notes
- Explain Haar features, integral image, AdaBoost, cascade
- Explain LBP: each pixel compared to 8 neighbors -> binary code -> histogram per grid cell -> concatenated feature vector
- Limitations: frontal faces work best, strong side lighting and heavy occlusion reduce accuracy
