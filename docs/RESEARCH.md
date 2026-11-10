# Haar and LBP: the theory, and how this project uses them

This note covers the two feature families that the whole system is built on. It
explains why each works, what its known limits are, and the five ways this project
uses them beyond the textbook pipeline. Every claim about our system is backed by a
measurement in [RESULTS.md](RESULTS.md).

---

## 1. Haar-like features and the Viola–Jones cascade

### 1.1 The feature
A Haar-like feature is the difference between the pixel sums of adjacent
rectangles: two-rectangle (edge), three-rectangle (line) and four-rectangle
(diagonal) masks, placed anywhere inside a 24×24 window at any size. Over 160,000
such features fit in one window. Each one is a crude, very cheap test of a fact
about faces. For example, the eye band is darker than the cheeks below it, and the
bridge of the nose is brighter than the eyes on either side.

### 1.2 The integral image
With $ii(x, y) = \sum_{x' \le x,\, y' \le y} i(x', y')$, computed in one pass using
$s(x,y) = s(x,y-1) + i(x,y)$ and $ii(x,y) = ii(x-1,y) + s(x,y)$, the sum over **any**
rectangle needs four look-ups:

$$\textstyle\sum_{\text{rect}} = ii(D) - ii(B) - ii(C) + ii(A)$$

A two-rectangle feature therefore costs 6 references, a three-rectangle feature 8,
and a four-rectangle feature 9, whatever the scale. Scanning at a new scale means
scaling the feature, not the image.

Lienhart & Maydt (2002) added 45°-rotated features evaluated on a *tilted* integral
image. The `haarcascade_frontalface_*` files shipped with OpenCV come from their work.

### 1.3 AdaBoost as feature selection
Each weak classifier is a single feature with a threshold and a polarity,
$h_j(x) = [\,p_j f_j(x) < p_j \theta_j\,]$. In each boosting round, AdaBoost picks
the one feature whose stump has the lowest weighted error, then re-weights the
training windows so the next round concentrates on the windows it got wrong. The
strong classifier is a weighted vote,
$\sum_t \alpha_t h_t(x) \ge \tfrac12\sum_t\alpha_t$ with $\alpha_t = \log\frac{1-\epsilon_t}{\epsilon_t}$.
Boosting here does two jobs at once: it *selects* features and it *combines* them.

### 1.4 The attentional cascade
Stages are chained, and each stage only sees windows the previous one accepted. If
stage $i$ has detection rate $d_i$ and false-positive rate $f_i$:

$$D = \prod_i d_i, \qquad F = \prod_i f_i$$

With ten stages at $d_i = 0.99$ and $f_i = 0.30$: $D \approx 0.904$ and
$F \approx 5.9\times10^{-6}$. Most windows in an image are background and are
rejected by the first stage or two after only a handful of features. That is why
the detector of Viola & Jones (2004) evaluated about 8 features per window on
average, out of 6,060 in the full cascade.

### 1.5 From windows to boxes: `minNeighbors`
`detectMultiScale` scans an image pyramid (step `scaleFactor`) and collects every
accepted window. It then groups overlapping windows (`groupRectangles`) and keeps a
group only if it has at least `minNeighbors` members. A true face fires in many
windows that are shifted and scaled slightly. A background patch that happens to
pass the cascade usually fires in only a few. **The size of the group is therefore
a confidence score.** OpenCV exposes it as `numDetections` from `detectMultiScale2`,
and our consensus detector uses it (§5.1).

### 1.6 Limits
Haar features are sums over regions, so they respond to *global* contrast
structure. That makes them fast and sensitive, but also easy to fool with any
texture that has a dark band over a lighter one: curtains, shelves, posters. They
are also trained on frontal faces, so profile views fall outside the model.

---

## 2. Local Binary Patterns

### 2.1 The operator
For a centre pixel $g_c$ and $P$ neighbours $g_p$ sampled on a circle of radius
$R$ (with bilinear interpolation for off-grid points):

$$\mathrm{LBP}_{P,R} = \sum_{p=0}^{P-1} s(g_p - g_c)\,2^p, \qquad s(x) = \begin{cases}1 & x \ge 0\\ 0 & x<0\end{cases}$$

`lbp.py` implements this directly with vectorised NumPy. `lbp_codes` builds the
$P$ shifted, interpolated copies of the image and compares them with the centre.

### 2.2 Why it tolerates lighting: monotonic invariance
Let $T$ be any strictly increasing grey-level transform: a gain, an offset, a gamma
curve, or a camera's auto-exposure. Then $T(g_p) \ge T(g_c) \iff g_p \ge g_c$, so
**every LBP code is unchanged**. This is a stronger guarantee than normalising mean
and variance, and it is the formal reason LBP beat Eigenfaces on FERET's
different-lighting probe set (Ahonen et al. 2006).

What it does *not* cover is lighting that changes the *order* of neighbouring
pixels: cast shadows and strong side light. That is the failure we measure on
Extended Yale B subsets 4–5, and the reason for §2.5.

### 2.3 Uniform patterns
A pattern is *uniform* if its circular bit string has at most two 0↔1 transitions.
Examples are 00000000 (flat), 00111000 (an edge or line end) and 11111111 (a spot).
There are $P(P-1)+2$ of them; for $P = 8$ that is 58, and every non-uniform code
shares one more bin, giving **59 bins instead of 256**. Ojala et al. (2002) found
uniform patterns account for about 90% of texture at (8,1). The rare non-uniform
codes are mostly noise, so pooling them makes each cell histogram both shorter and
more stable. (`uniform_table` in `lbp.py`; checked: 59 bins for P = 8, 243 for P = 16.)

### 2.4 The face descriptor
Ahonen, Hadid & Pietikäinen split the face into a grid, take a histogram of codes
in each cell, and concatenate the histograms. The histogram captures *what*
texture is present, and the grid keeps *where* it is. Faces are compared with the
(weighted) chi-square distance:

$$\chi^2_w(S, M) = \sum_j w_j \sum_b \frac{(S_{j,b} - M_{j,b})^2}{S_{j,b} + M_{j,b}}$$

Ahonen et al. also hand-set the weights $w_j$ on a 7×7 grid: highest at the eyes,
zero at some cheek and corner cells.

Our descriptor uses two radii, (8,1) and (8,2), on an 8×8 grid. That gives
128 cells × 59 bins = 7,552 values. Each cell is normalised to sum to 1, so each
cell's χ² lies in [0, 2], and the reported distance is the weighted *mean* over
cells.

### 2.5 Local Ternary Patterns and the Tan–Triggs chain
In near-uniform regions, sensor noise flips the sign of $g_p - g_c$ at random.
Tan & Triggs (2010) replace the hard threshold with a dead zone of width $t$:

$$s'(g_p, g_c, t) = \begin{cases} +1 & g_p \ge g_c + t\\ 0 & |g_p - g_c| < t \\ -1 & g_p \le g_c - t\end{cases}$$

The result is split into an "upper" and a "lower" binary pattern, each histogrammed
separately (`ltp_codes`). They pair it with a preprocessing chain:

1. gamma $\gamma = 0.2$, which compresses the dynamic range so shadows are lifted
2. a difference of Gaussians ($\sigma_0 = 1$, $\sigma_1 = 2$), a band-pass filter
   that removes the slow shading gradient a side lamp produces
3. two-stage robust contrast equalisation, then $\tau\tanh(\cdot/\tau)$ to clip outliers

This is the configuration that takes Yale B subset 4 from 48% to 97%.

### 2.6 LBP as a *detection* feature
Liao et al. (2007) proposed *multi-block* LBP: the same 8-neighbour comparison, but
on the *mean* of 3×3 blocks of pixels instead of single pixels. These blocks are
computed from an integral image, just like Haar sums. The resulting codes are
categorical (256 values), so each weak classifier is a 256-entry look-up table
rather than a threshold. OpenCV's `lbpcascade_frontalface_improved.xml` is a
cascade of these features. It uses integer comparisons instead of real-valued
sums, so it runs in **11 ms instead of 28 ms per frame** on this laptop and,
inheriting LBP's monotonic invariance, behaves differently under lighting.

---

## 3. Why the two families complement each other

| | Haar | LBP |
|---|---|---|
| Measures | signed **differences of sums** over rectangles | **signs** of differences between pixels/blocks |
| Sensitive to | global contrast layout, magnitude | local texture micro-structure, order only |
| Invariant to | additive offset (difference of sums) | any monotonic grey-level change |
| Typical false positive | high-contrast layouts that look like a face: shelves, patterned clothing | busy textures with locally face-like micro-patterns |
| Cost | floating-point sums | integer comparisons, ~2.5× faster |

Because the two features measure different things, **their false positives are
nearly independent**. In our mosaic test, 72% of true Haar boxes were confirmed by
the LBP cascade, against **6.5%** of false Haar boxes. That observation is the
basis of the consensus detector.

---

## 4. What this project adds to the textbook pipeline

The stage-1 report found that the textbook pipeline (Haar → CLAHE → OpenCV LBPH →
distance threshold) admitted **91% of strangers** at its default threshold, and that
no single threshold could separate students from strangers. The five additions
below each target one link in that chain, and each is measured separately.

### 4.1 Consensus detection: two cascades vote (`detector.py`)
Haar and LBP cascades run on the same CLAHE frame. A Haar box is kept if

* an LBP box overlaps it (IoU ≥ 0.3), **or**
* the Haar eye cascade finds a plausible eye pair in its upper 60% (horizontal
  separation 25–65% of the width, tilt ≤ 20°), **or**
* its Haar group has ≥ 20 neighbours (§1.5).

An LBP-only box survives only with an eye pair. The rules were chosen from
measured evidence rates on true and false boxes (Haar-only, 150 mosaics):

| evidence on a Haar box | true faces | false boxes |
|---|---|---|
| LBP cascade agrees | 72% | 6.5% |
| eye pair found | 58% | 15% |
| ≥ 20 Haar neighbours | 91% | 30% |

**Result:** false boxes fall from 0.31 to 0.10 per frame (−67%), and recall goes
from 98.4% to 94.6%.

### 4.2 Learned cell weights: a data-driven version of Ahonen's map (`lbp.learn_cell_weights`)
For each of the 128 cells we compute χ² distances over 20,000 same-person and
20,000 different-person pairs and score the cell by the Fisher criterion

$$J_j = \frac{(\mu_{b,j} - \mu_{w,j})^2}{\sigma_{b,j}^2 + \sigma_{w,j}^2}$$

The weights are normalised to mean 1. They are learned once, on 96 LFW people
(937 images) **that never appear in any test set**, and shipped in `assets/`.

The learned map (`docs/figures/cell_weights.png`) reproduces Ahonen's hand-set
intuition without being told it: brows and eyes reach 2–3.8, the nose and mouth
sit around 1–1.8, and the four corners, where hair and background sit, fall to
0.0–0.2.

**Result:** LFW rank-1 accuracy rises from 56.7% to **67.3%** on 62 people.

### 4.3 Open-set decision with a ratio test (`recognizer.OpenSetLBP`)
Attendance is *open-set*: the right answer for a stranger is "nobody". An absolute
threshold on $d_1$ (the distance to the nearest student) fails because $d_1$ shifts
with lighting, pose and camera for everyone at once.

We add Lowe's ratio test (from SIFT matching) at the **identity** level:

$$\text{accept } k^* \iff d_1 \le \tau_d \;\wedge\; \frac{d_1}{d_2} \le \tau_r$$

Here $d_1$ and $d_2$ are the distances to the nearest and second-nearest
*identities* (the minimum over each person's templates). A registered student is
much closer to their own templates than to anyone else's, while a stranger tends
to be equally far from everyone. In biometrics this is a form of *cohort
normalisation*: the other enrolled students act as the cohort.

**Result:** on LFW at FAR ≤ 1%, the share of genuine probes missed drops from
**88.1% (stage-1 LBPH) to 66.5%**.

### 4.4 Temporal score fusion (`tracker.py`, `benchmark.fuse`)
The tracker already follows each face across frames. In the benchmark we average
the per-identity distance vectors over $k$ frames (the sum rule of Kittler et al.,
1998) before deciding. Independent noise in each frame averages out, but a
stranger's lack of a clear winner does not.

**Result:** FRR at FAR ≤ 1% goes from 66.5% to **51.4%** with only 2 frames (LFW-62),
and to 24–42% with 4 frames on the 10-person set. The live system credits a student only when they win **5 of the last 8 frames**
(`CONFIRM_FRAMES`, `CONFIRM_WINDOW`). That is a vote rather than a strict streak, so a
single blink or downward glance doesn't discard the evidence.

### 4.5 Blink liveness from the Haar eye cascade (`tracker.Track.observe_eyes`)
The stage-1 report listed photo spoofing as out of scope. The Haar eye cascade was
trained on *open* eyes and largely stops firing when they close. Per track, we run
a small state machine: open for ≥ 2 frames → closed for 1–8 frames → open again =
a blink. With `LIVENESS` on, a student is credited only after their track has
blinked. A printed photograph never does.

This is a lightweight classical counterpart to the LBP micro-texture anti-spoofing
of Määttä, Hadid & Pietikäinen (2011), which needs a trained SVM and a spoof
dataset. It is not a formal anti-spoofing evaluation, and a video replay would
defeat it.

### 4.6 Explainability
Because every stage is hand-designed, the UI can show *why* a decision was made.
The "Why this decision" panel shows the chip, the LBP code image, and the weighted
per-cell χ² contribution against the nearest identity (an 8×8 heat map of where
the two faces differ), next to live gauges of $d_1$ and $d_1/d_2$ against
their thresholds. A neural network cannot show its reasoning this directly.

---

## 5. Other uses of Haar and LBP, not implemented here

* **LBP-TOP** (Zhao & Pietikäinen 2007): LBP on three orthogonal planes (XY, XT, YT)
  of a video volume. It captures facial *motion* texture and is used for expression
  recognition and video liveness.
* **Integral histograms** (Porikli 2005): an integral image per histogram bin gives
  any rectangle's LBP histogram in $O(\text{bins})$. This would let the cell grid
  adapt to the detected box at no cost.
* **Rotation-invariant LBP$^{riu2}$** ($P + 2$ bins): useless for whole faces,
  where orientation carries identity, but useful for detecting skin texture or
  occluders such as masks and hands.
* **Haar cascades for state rather than identity**: `haarcascade_smile.xml` and the
  eye cascades as per-region state detectors, for example to check that a face is
  attentive or that the mouth is not covered.
* **CS-LBP / LBP variance**: centre-symmetric pairs (16 bins) or a contrast-weighted
  histogram. Both are cheaper or more contrast-aware variants worth testing against
  uniform LBP.

---

## References

* P. Viola, M. Jones. *Robust real-time face detection.* IJCV 57(2), 2004.
* R. Lienhart, J. Maydt. *An extended set of Haar-like features for rapid object detection.* ICIP 2002.
* Y. Freund, R. Schapire. *A decision-theoretic generalization of on-line learning…* JCSS 55(1), 1997.
* T. Ojala, M. Pietikäinen, T. Mäenpää. *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns.* IEEE TPAMI 24(7), 2002.
* T. Ahonen, A. Hadid, M. Pietikäinen. *Face description with local binary patterns: application to face recognition.* IEEE TPAMI 28(12), 2006.
* S. Liao, X. Zhu, Z. Lei, L. Zhang, S. Z. Li. *Learning multi-scale block local binary patterns for face recognition.* ICB 2007.
* X. Tan, B. Triggs. *Enhanced local texture feature sets for face recognition under difficult lighting conditions.* IEEE TIP 19(6), 2010.
* D. Lowe. *Distinctive image features from scale-invariant keypoints.* IJCV 60(2), 2004.
* J. Kittler, M. Hatef, R. Duin, J. Matas. *On combining classifiers.* IEEE TPAMI 20(3), 1998.
* J. Määttä, A. Hadid, M. Pietikäinen. *Face spoofing detection from single images using micro-texture analysis.* IJCB 2011.
* G. Zhao, M. Pietikäinen. *Dynamic texture recognition using local binary patterns with an application to facial expressions.* IEEE TPAMI 29(6), 2007.
* F. Porikli. *Integral histogram: a fast way to extract histograms in Cartesian spaces.* CVPR 2005.
* K. Zuiderveld. *Contrast limited adaptive histogram equalization.* Graphics Gems IV, 1994.
* G. Huang et al. *Labeled Faces in the Wild.* UMass TR 07-49, 2007.
* A. Georghiades, P. Belhumeur, D. Kriegman. *From few to many: illumination cone models…* IEEE TPAMI 23(6), 2001.
