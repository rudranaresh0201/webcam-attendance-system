# Face Detection Attendance System

Webcam attendance: each student registers their face once, then the system
recognises faces live and writes present/absent to a CSV for the day.
Unregistered faces show as **Unknown** and are never logged.

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
python download_models.py         # only needed for the dnn backend (39 MB)
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

## Two backends

Every script takes `--backend classic` or `--backend dnn`. The default is set
by `BACKEND` in `config.py`. Each backend needs its own `train.py` run; the
registered images in `dataset/` are shared.

| Stage | `classic` (case-study pipeline) | `dnn` |
|---|---|---|
| Acquisition | `cv2.VideoCapture`, resize to 640 px, mirror | same |
| Preprocessing | Gaussian blur 3x3, grayscale, CLAHE | none needed |
| Face detection | Haar cascade (`detectMultiScale`) | YuNet CNN, also gives 5 landmarks |
| Geometric normalisation | crop to 200x200; optional eye alignment (`ALIGN_EYES`, off) | warp landmarks onto SFace's reference positions |
| Photometric normalisation | CLAHE on the face chip; optional Tan-Triggs and elliptical mask (off) | none needed |
| Features | LBP histograms over an 8x8 grid | SFace 128-D embedding |
| Classification | LBPH nearest neighbour, distance <= `LBPH_THRESHOLD` | cosine similarity to each student's mean embedding, >= `SFACE_THRESHOLD` |
| Liveness | not available | head-turn challenge |

False-detection reduction, both backends:

- minimum face size, so background faces are ignored
- a threshold, so strangers are Unknown instead of the nearest student
- each face is tracked across frames (box overlap) and must be recognised as
  the same student for `CONFIRM_FRAMES` consecutive frames
- registration rejects blurry (variance of Laplacian), too dark and too bright captures

### Liveness check (dnn only)

Before marking someone present, the system asks them to turn their head left
or right (random). A real head is 3-D, so the nose tip shifts sideways relative
to the midpoint between the eyes. A printed or on-screen photo is flat, so the
nose stays centred however the photo is turned. See `liveness.py`.
It does not stop a replayed video. Disable with `--no-liveness` or
`LIVENESS = False`.

## Evaluating and tuning

```
python evaluate.py                    # on your registered students
python evaluate.py --backend dnn
python evaluate.py --data path/to/dataset --min-images 20 --max-people 10 --max-images 30
```

`--data` accepts any public dataset laid out as one folder per person (LFW is
already in that layout). It reports identification accuracy, how many
strangers get accepted, how many students get missed, and suggests a
threshold. **Set the thresholds in `config.py` from a run on your own
registered students** - the defaults are starting points.

First numbers, LFW (funneled), 10 people x 30 images, default settings:

| Backend | Identification accuracy | Strangers accepted at the default threshold |
|---|---|---|
| classic | 76.6% | 91.2% (`LBPH_THRESHOLD = 70` is far too lenient on this data; 50 gave 0.7%) |
| dnn | 100% | 5.1% (0.7% at 0.410) |

LFW is unconstrained news photos, much harder than one webcam in one room, so
expect classic to do better on your own registrations. The threshold still
has to be measured there.

## Files

| File | Purpose |
|---|---|
| `config.py` | every tunable parameter |
| `engines.py` | the two backends behind one interface: `detect`, `describe`, `train`, `predict` |
| `register.py` | capture from webcam, or `--from-folder` to import photos |
| `train.py` | build the recognizer from `dataset/` |
| `attendance.py` | live loop, CSV writing |
| `tracker.py` | follows faces between frames |
| `liveness.py` | head-turn check |
| `evaluate.py` | accuracy + threshold suggestion |
| `utils.py` | camera, quality checks, drawing |
| `check_setup.py`, `download_models.py`, `setup.bat` | environment |

`dataset/`, `models/` and `records/` are git-ignored: they hold face photos and
attendance sheets. Share them privately if both of you need the same data.

## Not done yet

- benchmark on two public datasets (`evaluate.py --data` is the hook)
- threshold tuning on real registrations
- pipeline-stage figures for the report (`engines.lbp_image` computes the LBP code image)
- GUI
