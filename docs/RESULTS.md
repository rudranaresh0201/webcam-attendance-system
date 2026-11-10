# Results

Everything here is reproduced by `python benchmark.py` (about 10 minutes on an
18-thread laptop CPU, no GPU). Raw numbers are in `docs/results/*.json` and the
figures in `docs/figures/`.

**Data.** LFW (deep-funneled, 250×250, from the scikit-learn mirror) and Extended
Yale B, cropped (38 subjects × 64 lighting directions, 2,414 readable images).

**Protocol** (as planned in the stage-1 report, §6.5):
* *Identification:* 5-fold per person. Train on 4 folds, test on the 5th.
* *Impostor test:* leave one person out of the gallery entirely; all their images
  are strangers.
* FAR = strangers accepted / all stranger probes. FRR = genuine probes that are
  rejected *or* given to the wrong student (eq. 6 of the report).
* The operating point is the most lenient threshold(s) with FAR ≤ 1%.

---

## 1. Detection: Haar vs LBP cascade vs consensus

250 frames of 640×500, each a 2×2 mosaic of random LFW photos (1,000 faces at known
positions). A box counts as correct if its centre is within 35 px of a face centre
and its width is 70–200 px. Any other box is a false box.

| detector | recall | false boxes / frame | ms / frame |
|---|---|---|---|
| Haar (`frontalface_default`, minNeighbors 6) | 98.4% | 0.31 | 27.9 |
| LBP cascade (`lbpcascade_frontalface_improved`) | 71.3% | 0.02 | **11.3** |
| Union | 98.7% | 0.32 | 39.2 |
| **Consensus (ours)** | **94.6%** | **0.10** | 39.6 |

Consensus removes two-thirds of Haar's false boxes and costs 3.8 points of recall.
In live use the tracker and the recognition threshold remove most of the
remaining false boxes. The LBP cascade's boxes are tighter than Haar's: the
median width ratio is 1.37, which is used to re-frame LBP-only boxes.

**Under lighting (Yale B, faces padded with a flat border):**

| subset (light angle) | 1 (≤12°) | 2 (≤25°) | 3 (≤50°) | 4 (≤77°) | 5 (>77°) |
|---|---|---|---|---|---|
| Haar | 54% | 54% | 40% | 42% | 4% |
| Haar + CLAHE | 35% | 39% | 30% | 42% | 8% |
| Consensus | 99% | 98% | 64% | 40% | 3% |
| Consensus + CLAHE | 98% | 95% | 62% | 45% | 8% |

*Caveat:* the Yale crops are tight (forehead to chin) and padded with a flat grey
border. The Haar cascade, trained on looser framing, finds them only half the time
even under frontal light, while the tighter LBP cascade does not have this
problem. This table therefore says more about framing than lighting for subsets
1–3. Subsets 4–5 show the real limit: past about 50° neither cascade sees half a
face in shadow, and CLAHE recovers only a few points.

## 2. Recognition on LFW (open set)

### 2a. Stage-1 replication: 10 people × 30 images (261 faces found)

| method | rank-1 | FRR @ FAR ≤ 1%, distance only | + ratio test | + 4-frame fusion |
|---|---|---|---|---|
| OpenCV LBPH (stage 1) | 70.9% | 84.3% | – | – |
| uLBP (8,1) | 70.9% | 85.1% | 77.0% | 40.0% |
| uLBP (8,1)+(8,2) | 73.2% | 84.3% | 74.3% | 30.0% |
| **uLBP (8,1)+(8,2), learned weights** | **77.0%** | 78.5% | **66.3%** | 42.0% |
| LTP (8,1)+(8,2), Tan–Triggs, weighted | 74.7% | 80.1% | 76.2% | **24.0%** |

The stage-1 report measured 76.6% rank-1 and 91.2% FAR at threshold 70 with the
same settings. Our re-run gives 70.9% and 83.9%. The difference most likely comes
from the LFW copy (the scikit-learn deep-funneled mirror) and the detector picking
a slightly different face set (4 images skipped). All rows in this table share
the same chips, so the comparison between methods is fair. With only about 65
groups of 4 frames, the fusion column is noisy (±6 points).

### 2b. Larger set: all 62 LFW people with ≥ 20 images, 20 each (1,201 faces)

| method | rank-1 | FRR @ FAR ≤ 1%, distance | ratio | distance ∧ ratio | 2-frame fusion |
|---|---|---|---|---|---|
| OpenCV LBPH (stage 1) | 56.6% | 88.1% | – | – | – |
| uLBP (8,1) | 52.5% | 88.2% | 82.7% | 82.2% | 68.9% |
| uLBP (8,1)+(8,2) | 56.7% | 86.4% | 80.8% | 78.9% | 64.6% |
| **uLBP (8,1)+(8,2), learned weights** | **67.3%** | 80.2% | 69.0% | **66.5%** | **51.4%** |
| LTP (8,1)+(8,2), Tan–Triggs, weighted | 59.9% | 83.3% | 84.3% | 78.7% | 68.6% |

At the chosen operating point the **wrong-student rate is 0.1%**. Almost every
error is a rejection ("not sure"), not a misidentification, which is the right
failure mode for attendance: a missed student can be marked by hand, but a false
mark is never noticed.

The best operating point here was τ_d = 0.38 and τ_r = 0.88.

### 2c. Choosing the live distance limit

On LFW, the distance alone barely separates people: the median d₁ of a correct
match is 0.35, and the median d₁ of a stranger is 0.40. **The ratio test does the
real work.** Loosening τ_d with τ_r fixed at 0.88:

| τ_d | strangers accepted (1 frame) | (2-frame fusion) | students missed (1 frame) |
|---|---|---|---|
| 0.38 | 0.8% | 0.2% | 67.4% |
| 0.45 | 1.6% | 0.2% | 65.0% |
| **0.50** | **1.7%** | **0.2%** | **64.7%** |

On the webcam, a student's live face against their own enrolment photos (taken a
little earlier) measured 0.38–0.47. The same person against other people's photos
measured 0.60–0.64. So 0.38 turned real students away. The live default is
**τ_d = 0.50**: a loose sanity bound, with the ratio test and the 5-of-8-frame
vote keeping strangers out.

*Caveat:* with **only one** student enrolled there is no runner-up, so the ratio
test cannot run and τ_d is the only guard. The UI says so. In a real class this
never happens. Run **Settings → Tune for this class** (or `python evaluate.py`)
once three or more students are enrolled.

![FAR vs FRR](figures/far_frr_all62_20.png)

### 2d. Small classes: the background crowd

The ratio test needs a runner-up. With **one** student enrolled, every face's closest
student is that student, so only the distance limit stands in the way. With few
students the runner-up is often far away. Measured on LFW-62 (15 enrolment photos
per student, 40 random classes per size, τ_d = 0.50, τ_r = 0.88):

| students enrolled | strangers accepted, no crowd | **with crowd** | students missed, no crowd | with crowd |
|---|---|---|---|---|
| 1 | **55.5%** | **0.3%** | 5.0% | 53.3% |
| 2 | 22.9% | 0.5% | 20.7% | 61.3% |
| 5 | 7.0% | 0.7% | 40.8% | 59.2% |
| 10 | 3.9% | 1.1% | 50.6% | 59.8% |

The **crowd** is 192 faces (2 each of the 96 weight-learning people, never tested on),
kept silently in the gallery as texture histograms. No photos are stored. A face is
accepted only if a student beats every crowd face *and* every other student by the
ratio. This is the cohort / background-model idea from speaker verification.

### 2e. Robustness: occlusion, mirror enrolment, trimmed χ², final limit

Probes are occluded with flat grey regions: an eye bar (sunglasses), the lower face
(a mask), and a hand over one cheek. The crowd is in the gallery. Rows are per
frame, averaged over classes of 1, 5 and 10 students:

| probe | baseline: students missed | **mirror enrolment + trim 20%** | strangers accepted |
|---|---|---|---|
| unoccluded | 56.6% | **52.8%** | 0.5% |
| eye bar | 92.5% | **65.4%** | 0.7% |
| mask (lower face) | **100%** | **62.1%** | 0.9% |
| hand on cheek | 76.7% | **55.9%** | 0.4% |

*Trimmed χ²* ignores the worst-matching 20% of grid cells, so an occluder can only
spoil a bounded part of the distance. *Mirror enrolment* adds each enrolment photo
flipped. Five extra shifted/scaled crops per photo gained only 2 points, so they
were not adopted.

**Final runner-up limit.** A live end-to-end test found one stranger photo whose ratio
wobbled between 0.85 and 0.89 across frames, so the 5-of-8 vote passed at 0.88.
Sweeping the limit with the full pipeline (classes of 1–30 students):

| τ_r | strangers accepted per frame | students missed per frame (LFW) |
|---|---|---|
| 0.88 | 0.22–0.74% | 56% |
| **0.84** | **0.02–0.12%** | 72% |
| 0.80 | ≤ 0.03% | 83% |

The shipped value is **0.84**. "Students missed" here compares news photos taken
years apart. A student enrolled this week on the same webcam scores about 0.7, well
inside the limit.

## 3. Recognition under lighting: Extended Yale B

Gallery: subset 1 only (7 near-frontal-light images per person). Probes: subsets
2–5. Nearest neighbour, unweighted χ². Faces are the dataset's own aligned crops,
so this isolates illumination from detection.

| descriptor, normalisation | S2 | S3 | S4 | S5 |
|---|---|---|---|---|
| LBP 256-bin, none (≈ OpenCV LBPH) | 99.6% | 100% | 48.1% | 7.0% |
| uLBP (8,1)+(8,2), CLAHE | 99.6% | 100% | 62.7% | 13.7% |
| uLBP (8,1)+(8,2), Tan–Triggs | 100% | 100% | 86.2% | 50.8% |
| **LTP (8,1)+(8,2), Tan–Triggs** | **100%** | **100%** | **97.0%** | **58.8%** |

Subsets come from the light angle θ = arccos(cos A · cos E), with boundaries at
12°/25°/50°/77°. The probe counts (456 / 380 / 601 / 714) differ slightly from the
standard split for subsets 3 and 4.

CLAHE is the right default for a classroom: it is cheap and improves both
detection and recognition. Under strong side light, switch to **Tan–Triggs + LTP**
in Controls. That trades a few points on LFW-like variation for a very large gain
on shadows.

## 4. Tests

**`tests/smoke_test.py`** (no camera, no browser). Four LFW people are enrolled from
25 photos each. Every probe photo becomes an 8-frame clip with webcam-like jitter
(a few pixels of movement, brightness flicker, sensor noise) and is fed to the real
`AttendanceSession`:

* students: all 4 credited; 12 of 40 single-photo clips of *unseen* photos (30%)
* strangers, 4 students enrolled: **0 of 80** clips credited
* strangers, **1** student enrolled: **0 of 128** clips credited

**`tests/e2e_test.py`** (the real app in a real browser; a video file stands in for the
webcam and a temporary folder for the data). 16 checks, all passing:

* run 1: enrol 2 students by uploading photos → start attendance → 40 s of video of
  4 other people → **nobody marked present**, register says both Absent
* run 2: video of the 2 students (photos never enrolled) → **both marked present in
  25 s** → private mode hides names and shows "Welcome, …" → view switch → Finish
  and save → Records table → CSV download

## 5. Live

On this laptop (built-in 640×480 webcam, CPU only), the full pipeline runs at
17–27 fps in the web app: consensus detection, recognition of every face, tracking
and the explanation panel.

With 4 LFW identities enrolled, an unenrolled person at the camera was held at
d₁ ≈ 0.60–0.64 and d₁/d₂ ≈ 0.96–0.99, against τ_d = 0.38 and τ_r = 0.88. They
stayed **Unknown** throughout.
