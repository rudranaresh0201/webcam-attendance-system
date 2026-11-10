# Walkthrough: how stage 2 was built

This walks through the work in order, with the reason for each step. Use it to
prepare the viva or presentation.

## 0. Where stage 1 left off

The repository already had a working command-line pipeline: Gaussian blur →
grayscale → CLAHE → Haar → 200×200 chip → OpenCV LBPH → distance threshold, plus
an IoU tracker requiring 5 consecutive frames.

The stage-1 report's feasibility trial (10 LFW people) found:

* 76.6% rank-1 identification, but
* **91.2% of strangers accepted** at the default threshold 70, and
* at threshold 50 strangers dropped to 0.7%, but **82% of genuine faces were
  rejected**.

So the threshold alone cannot separate students from strangers. Everything in
stage 2 attacks that problem from a different side, while staying entirely within
classical image processing.

## 1. Set up and get the data

* Python 3.12 virtual environment, `opencv-contrib-python` 4.14.
* The pip wheel ships Haar cascades but **not** the LBP cascades, so
  `lbpcascade_frontalface_improved.xml` was taken from OpenCV's repository into
  `cascades/`.
* **LFW**, via scikit-learn's mirror (deep-funneled, 5,749 people).
* **Extended Yale B**: the UCSD download link is dead, so the cropped set came from
  a GitHub mirror (38 subjects × 64 lighting conditions).

## 2. Write LBP ourselves (`lbp.py`)

OpenCV's LBPH is a black box: 256 bins, no weights, one distance out. To do
anything new, we needed the cell histograms. `lbp.py` implements:

* circular LBP(P, R) with bilinear interpolation, checked against a worked example
  (centre 100 → code 71)
* the uniform mapping (59 bins for P = 8)
* Local Ternary Patterns
* spatial histograms
* per-cell χ²
* Fisher-score weight learning

It runs in about 8 ms per face in vectorised NumPy.

## 3. Build the benchmark before changing anything (`benchmark.py`)

The protocol is the one the stage-1 report promised (§6.5): 5-fold identification,
a leave-one-person-out impostor test, and FRR at FAR ≤ 1%. Distances are computed
once as a full matrix across 12 processes, so every decision rule can be tested
without re-running feature extraction.

The first run replicated the stage-1 baseline (70.9% / 83.9% FAR at 70; see
RESULTS.md for why this differs slightly from 76.6% / 91.2%).

## 4. Detection: make the two cascades vote (`detector.py`)

**Measured first:** on LFW mosaics, Haar found 98.4% of faces with 0.31 false boxes
per frame. The LBP cascade found 71.3% with 0.02, at 2.5× the speed. *They make
different mistakes.*

**Analysed:** for each Haar box we recorded three pieces of evidence: whether the
LBP cascade agreed, whether an eye pair was found, and its neighbour count. LBP
agreement was present on 72% of true boxes but only 6.5% of false ones.

**Rule:** keep a box if the LBP cascade agrees, or if an eye pair is present, or if
it has ≥ 20 Haar neighbours. The first attempt used agreement or eyes only; it
had too little recall (80%), so the neighbour-count rule was added.

**Result:** recall 94.6%, false boxes 0.10 per frame.

## 5. Recognition: three changes, measured one at a time

| change | why | LFW-62 rank-1 | FRR @ FAR 1% |
|---|---|---|---|
| OpenCV LBPH (start) | | 56.6% | 88.1% |
| uniform LBP + second radius | fewer, more stable bins; two scales of texture | 56.7% | 86.4% |
| + learned cell weights | stop counting hair and background | **67.3%** | 80.2% |
| + ratio test d₁/d₂ | open-set: a stranger has no clear winner | | **66.5%** |
| + fuse 2 frames | noise averages out, ambiguity doesn't | | **51.4%** |

The weights are learned on 96 LFW people who never appear in a test set, so the
gain is not over-fitting. The learned map closely matches the region weights
Ahonen et al. set by hand.

## 6. Lighting: Tan–Triggs and LTP (Yale B)

With subset 1 as the gallery, plain LBP collapses under side light: 48% on subset
4 and 7% on subset 5. The Tan–Triggs chain (gamma → difference of Gaussians →
contrast equalisation) together with Local Ternary Patterns reaches **97%** and
**59%**. This is a setting in the UI, for rooms with strong window light.

## 7. Liveness: the eye cascade as a blink sensor

The Haar eye cascade fires on open eyes. Per tracked face, a small state machine
(open → closed for 1–8 frames → open) detects a blink. When enabled, nobody is
credited until they blink, so a printed photograph is never marked present.

## 8. The live system and UI (`app.py`, `static/`)

* One thread owns the webcam and runs the shared `FaceEngine` and
  `AttendanceSession`. Flask serves an MJPEG stream and a JSON API.
* The UI is a single page in plain HTML/CSS/JS with no build step.
* The UI is meant to *show how it works* while it runs, in plain language:
  * a sentence above the video saying what it sees and why (*"closest is X, but
    not close enough: 0.58 against a limit of 0.50"*)
  * **What the computer sees**: the video itself switches to the pipeline's own
    images: contrast-fixed (CLAHE), LBP texture, and the raw votes of the Haar and
    LBP detectors
  * **How it decided**: the chip, texture codes, difference map and closest student
    stacked in 3D, with meters against each limit; the χ² formula is under "The maths"
* After first use the live distance limit was raised from 0.38 to 0.50: a real
  student measured 0.38–0.47 against their own enrolment, which LFW's limit rejected
  (RESULTS.md §2c). Confirmation changed from 5 frames in a row to 5 of the last 8.
* **Calibrate** runs the same open-set protocol on the enrolled class and offers
  the thresholds that keep FAR ≤ 1%.

## 9. Verification

* `tests/smoke_test.py` enrols 4 people, then drives the attendance session with
  photos they never trained on. All 4 are credited, 0 of 50 stranger photos are,
  and the CSV is correct.
* Every API path was exercised: webcam enrolment start and cancel, photo upload,
  retrain on a settings change, calibration, session start and stop, CSV download
  (including a path-traversal check) and student deletion.
* Live at 17–27 fps. An unenrolled person stayed Unknown (d₁ ≈ 0.62 vs τ = 0.38).

## Demo script (≈ 5 minutes)

1. **Live, nobody enrolled.** The sentence says it sees a face but nobody is enrolled.
   Flip *What the computer sees* through Contrast → Texture → Detector votes.
2. **Enrol** yourself (30 captures, about 10 s). Mention the quality gate:
   variance of the Laplacian, brightness.
3. **Live → Start attendance.** The box turns amber with a 1/5 … 5/5 count, then
   green, and the log records the time.
4. Have a classmate who is **not** enrolled sit in. They stay red "Unknown". Show
   the gauges: d₁/d₂ is close to 1 because they are equally far from everyone.
   That is the ratio test working.
5. **How it decided**: hover each numbered step to lift its layer, then click *Lay
   the layers flat*. The χ² map shows *where* on the face the match differs.
6. **Settings → Ask for a blink**, then hold up a phone photo of an enrolled
   student. It is never credited. Blink in person and you are.
7. **How well it works.** Turn over Fig. 3 (the ratio test) and Fig. 5 (learned weights
   vs Ahonen).
8. **Finish and save → Records.** Download the CSV.
