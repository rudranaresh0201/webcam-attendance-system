# Face Detection Attendance System

Webcam attendance: each student registers their face once, then the system
recognises faces live and writes present/absent to a CSV for the day.
Unregistered faces show as **Unknown** and are never logged.

Built with classical image processing only (OpenCV): no neural networks.

The original spec is in [plan.md](plan.md). Where this README and the plan
differ, the README describes what the code actually does.

## Setup (once)

Needs Python 3.10+ and a webcam.

**Windows:** double-click `setup.bat`, or run it from a terminal.

**Manual / macOS / Linux:**

```
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python check_setup.py             # checks packages and the webcam
```

If `check_setup.py` says OpenCV has no `cv2.face`, you have plain
`opencv-python` installed. This project needs `opencv-contrib-python` 4.x
(OpenCV 5 removed the Haar cascade from the main package). Use the venv.

## Use

```
python register.py        # roll number + name, then look at the camera (30 images)
python train.py           # after registering everyone
python attendance.py      # press q to finish; absentees are written on quit
```

Output: `records/attendance_YYYY-MM-DD.csv` with `roll, name, status, time`.
Running again on the same day keeps earlier Present marks and adds new ones.

Register at least two people. With one student the recognizer has nobody else
to confuse them with, and only the threshold keeps strangers out.

## Pipeline

| Stage | Implementation |
|---|---|
| Image acquisition | `cv2.VideoCapture`, resize to 640 px wide, mirror |
| Noise reduction | Gaussian blur 3x3 |
| Grayscale conversion | `cv2.cvtColor(..., COLOR_BGR2GRAY)` |
| Contrast enhancement | CLAHE (adaptive histogram equalisation) |
| Face detection | Haar cascade, `detectMultiScale` |
| Geometric normalisation | crop to 200x200; optional eye alignment (`ALIGN_EYES`, off) |
| Photometric normalisation | CLAHE on the face chip; optional Tan-Triggs and elliptical mask (off) |
| Feature extraction | LBP histograms over an 8x8 grid |
| Classification | LBPH nearest neighbour, distance <= `LBPH_THRESHOLD` |
| Output | CSV attendance with timestamp |

False-detection reduction:

- `MIN_NEIGHBORS` and a minimum face size, so spurious and background boxes are dropped
- a distance threshold, so strangers are Unknown instead of the nearest student
- each face is tracked across frames (box overlap) and must be recognised as
  the same student for `CONFIRM_FRAMES` consecutive frames
- registration rejects blurry (variance of Laplacian), too dark and too bright captures

## Evaluating and tuning

```
python evaluate.py                    # on your registered students
python evaluate.py --data path/to/dataset --min-images 20 --max-people 10 --max-images 30
```

`--data` accepts any public dataset laid out as one folder per person (LFW is
already in that layout). It reports identification accuracy, how many
strangers get accepted, how many students get missed, and suggests a
threshold. **Set `LBPH_THRESHOLD` in `config.py` from a run on your own
registered students** - the default is a starting point.

First numbers, LFW (funneled), 10 people x 30 images, default settings:

| Identification accuracy | Strangers accepted at `LBPH_THRESHOLD = 70` | At 50 |
|---|---|---|
| 76.6% | 91.2% | 0.7%, but 82% of genuine faces are then rejected |

LFW is unconstrained news photos, much harder than one webcam in one room, so
expect better on your own registrations. The threshold still has to be
measured there.

## Files

| File | Purpose |
|---|---|
| `config.py` | every tunable parameter |
| `engines.py` | the pipeline: `detect`, `describe`, `train`, `predict` |
| `register.py` | capture from webcam, or `--from-folder` to import photos |
| `train.py` | build the recognizer from `dataset/` |
| `attendance.py` | live loop, CSV writing |
| `tracker.py` | follows faces between frames |
| `evaluate.py` | accuracy + threshold suggestion |
| `utils.py` | camera, quality checks, drawing |
| `check_setup.py`, `setup.bat` | environment |

`dataset/`, `models/` and `records/` are git-ignored: they hold face photos and
attendance sheets. Share them privately if both of you need the same data.

The `dnn-comparison` branch holds an earlier version with an extra neural-network
backend. It is not part of the project; it is there only as a reference point.

## Not done yet

- benchmark on two public datasets (`evaluate.py --data` is the hook)
- threshold tuning on real registrations
- pipeline-stage figures for the report (`engines.lbp_image` computes the LBP code image)
- GUI
