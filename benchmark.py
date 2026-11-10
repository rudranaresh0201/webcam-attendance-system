"""Stage-2 experiments on public datasets. Writes docs/results/*.json and docs/figures/*.png.

    python benchmark.py --lfw ~/datasets/lfw_home/lfw_funneled --yale ~/datasets/yale_src/CroppedYale

Experiments
  detection    Haar vs LBP cascade vs union vs consensus on 2x2 LFW mosaics (4 faces per
               640x500 frame, known positions) -> recall, false boxes per frame, ms per frame;
               and on Extended Yale B per lighting subset, with and without CLAHE.
  weights      learn per-cell LBP weights on LFW people who are NEVER used for testing.
  lfw          identification (5-fold) + open-set impostor test (leave-one-person-out), the
               protocol of the stage-1 report, for the OpenCV LBPH baseline and our variants;
               FRR at FAR <= 1% for three decision rules and with 5-frame fusion.
  yale         illumination: gallery = Yale B subset 1, probes = subsets 2-5, accuracy per
               subset for each illumination normalisation x descriptor.
"""
import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import config
import lbp
from engines import FaceEngine, tan_triggs
from detector import iou

CACHE = Path.home() / "datasets" / "cache"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


# =============================================================================== data
def lfw_people(root, min_images, max_images=None, max_people=None, max_count=None):
    people = []
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        files = sorted(d.glob("*.jpg"))
        if len(files) >= min_images and (max_count is None or len(files) <= max_count):
            people.append((d.name, files[:max_images]))
    return people[:max_people]


def lfw_chips(engine, people, tag):
    """Detect (largest face, as the live system does) and normalise; cached on disk."""
    path = CACHE / f"lfw_{tag}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["chips"], z["labels"], list(z["names"]), int(z["skipped"])
    chips, labels, names, skipped = [], [], [], 0
    for li, (name, files) in enumerate(people):
        names.append(name)
        for f in files:
            img = cv2.imread(str(f))
            faces = engine.detect(img)
            if not faces:
                skipped += 1
                continue
            face = max(faces, key=lambda fc: fc.box[2] * fc.box[3])
            chips.append(engine.raw_chip(img, face))
            labels.append(li)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, chips=np.array(chips), labels=np.array(labels), names=np.array(names),
                        skipped=skipped)
    return np.array(chips), np.array(labels), names, skipped


YALE_RE = re.compile(r"yaleB(\d+)_P00A([+-]\d+)E([+-]\d+)\.pgm$")


def yale_subset(az, el):
    """Standard Yale B subsets by angle between light source and camera axis."""
    theta = np.degrees(np.arccos(np.cos(np.radians(az)) * np.cos(np.radians(el))))
    for s, hi in ((1, 12), (2, 25), (3, 50), (4, 77)):
        if theta <= hi + 0.5:
            return s
    return 5


def yale_images(root):
    imgs, subj, subset = [], [], []
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.pgm")):
            m = YALE_RE.search(f.name)
            if not m:
                continue        # ambient image, logs
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            imgs.append(img)
            subj.append(int(m.group(1)))
            subset.append(yale_subset(int(m.group(2)), int(m.group(3))))
    return imgs, np.array(subj), np.array(subset)


# =============================================================================== detection
def mosaics(people, n, rng):
    """n frames of 2x2 LFW images (4 known faces) on a 640x500 canvas."""
    files = [f for _, fs in people for f in fs]
    frames = []
    for _ in range(n):
        tiles = [cv2.imread(str(files[i])) for i in rng.choice(len(files), 4, replace=False)]
        canvas = np.full((500, 640, 3), 40, np.uint8)
        centres = []
        for k, t in enumerate(tiles):
            y, x = 250 * (k // 2), 70 + 250 * (k % 2)
            canvas[y:y + 250, x:x + 250] = t
            centres.append((x + 125, y + 125))
        frames.append((canvas, centres))
    return frames


def score_boxes(boxes, centres, tol=35):
    hit = set()
    false = 0
    for (x, y, w, h) in boxes:
        cx, cy = x + w / 2, y + h / 2
        j = next((j for j, c in enumerate(centres) if abs(cx - c[0]) < tol and abs(cy - c[1]) < tol
                  and 70 <= w <= 200), None)
        if j is None or j in hit:
            false += 1
        else:
            hit.add(j)
    return len(hit), false


def exp_detection(lfw_root, yale_root, out):
    rng = np.random.default_rng(1)
    people = lfw_people(lfw_root, 1)
    frames = mosaics(people, 250, rng)
    res = {"mosaic": {}, "yale": {}}

    # calibrate LBP box scale against Haar on agreeing detections
    eng = FaceEngine(detector="haar")
    ratios = []
    for canvas, _ in frames[:120]:
        g = eng.detection_image(canvas)
        H, L = eng.detector.run_haar(g), eng.detector.run_lbp(g)
        for h in H:
            for b in L:
                if iou(h, b) > 0.3:
                    ratios.append(h[2] / b[2])
    res["lbp_box_scale"] = float(np.median(ratios))
    config.LBP_BOX_SCALE = round(res["lbp_box_scale"], 2)
    print(f"LBP->Haar box scale: {res['lbp_box_scale']:.3f} (n={len(ratios)})")

    for mode in ("haar", "lbp", "union", "consensus"):
        hits = false = 0
        t = time.perf_counter()
        for canvas, centres in frames:
            faces = eng.detect(canvas, mode)
            h, fa = score_boxes([f.box for f in faces], centres)
            hits, false = hits + h, false + fa
            eng._cache = (None, None, None)
        ms = (time.perf_counter() - t) / len(frames) * 1000
        res["mosaic"][mode] = {"recall": hits / (4 * len(frames)), "false_per_frame": false / len(frames),
                               "ms_per_frame": ms}
        print(f"  {mode:10s} recall {hits / (4 * len(frames)):.3f}  false/frame {false / len(frames):.3f}  {ms:.1f} ms")

    # Yale B: detection under lighting, CLAHE on/off
    imgs, _, subset = yale_images(yale_root)
    for clahe in (False, True):
        for mode in ("haar", "consensus"):
            key = f"{mode}{'+clahe' if clahe else ''}"
            found = np.zeros(len(imgs), bool)
            for i, img in enumerate(imgs):
                pad = cv2.copyMakeBorder(img, 70, 70, 70, 70, cv2.BORDER_CONSTANT, value=int(np.median(img)))
                g = cv2.GaussianBlur(pad, config.BLUR_KERNEL, 0)
                if clahe:
                    g = eng._clahe.apply(g)
                faces = eng.detector.detect(g, mode)
                cy, cx = pad.shape[0] / 2, pad.shape[1] / 2
                found[i] = any(abs(f.box[0] + f.box[2] / 2 - cx) < 50 and abs(f.box[1] + f.box[3] / 2 - cy) < 60
                               for f in faces)
            res["yale"][key] = {str(s): float(found[subset == s].mean()) for s in range(1, 6)}
            print(f"  yale {key:16s}", " ".join(f"{found[subset == s].mean():.2f}" for s in range(1, 6)))
    (out / "detection.json").write_text(json.dumps(res, indent=2))
    return res


# =============================================================================== features & distances
VARIANTS = {
    # name: (descriptor kwargs, illumination, weighted)
    "uLBP(8,1)":                 (dict(scales=((8, 1),)), "clahe", False),
    "uLBP(8,1)+(8,2)":           (dict(scales=((8, 1), (8, 2))), "clahe", False),
    "uLBP(8,1)+(8,2) weighted":  (dict(scales=((8, 1), (8, 2))), "clahe", True),
    "LTP(8,1)+(8,2) Tan-Triggs weighted": (dict(scales=((8, 1), (8, 2)), ternary=True), "tantriggs", True),
}


def illum(chips, how, clahe):
    if how == "clahe":
        return [clahe.apply(c) for c in chips]
    if how == "tantriggs":
        return [tan_triggs(c) for c in chips]
    return list(chips)


def _feat_worker(args):
    chips, kw = args
    d = lbp.LBPDescriptor(**kw)
    return np.stack([d(c) for c in chips])


def features(chips, kw, workers=12):
    parts = np.array_split(np.arange(len(chips)), workers)
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_feat_worker, [([chips[i] for i in p], kw) for p in parts]))
    return np.concatenate(out)


_G = {}


def _dist_init(F, w):
    _G["F"], _G["w"] = F, w / w.sum()


def _dist_rows(rows):
    F, w = _G["F"], _G["w"]
    return np.stack([lbp.chi2_cells(F[i], F) @ w for i in rows])


def distance_matrix(F, w, workers=12):
    parts = np.array_split(np.arange(len(F)), workers * 4)
    with ProcessPoolExecutor(workers, initializer=_dist_init, initargs=(F, w)) as ex:
        return np.concatenate(list(ex.map(_dist_rows, parts)))


def cross_distance(Q, G, w):
    w = w / w.sum()
    return np.stack([lbp.chi2_cells(q, G) @ w for q in Q])


# =============================================================================== open-set protocol
def identity_min(D, labels, gallery_mask_fn, n_ids):
    """M[i, k] = min distance from probe i to identity k's gallery images (inf if none)."""
    M = np.full((len(D), n_ids), np.inf)
    for i in range(len(D)):
        g = gallery_mask_fn(i)
        for k in range(n_ids):
            sel = g & (labels == k)
            if sel.any():
                M[i, k] = D[i, sel].min()
    return M


def folds_of(labels, k=5):
    f = np.zeros(len(labels), int)
    for l in np.unique(labels):
        idx = np.flatnonzero(labels == l)
        f[idx] = np.arange(len(idx)) % k
    return f


def decisions(M):
    """From per-identity distances -> (best id, d1, ratio d1/d2)."""
    order = np.argsort(M, 1)
    i = np.arange(len(M))
    d1, d2 = M[i, order[:, 0]], M[i, order[:, 1]]
    return order[:, 0], d1, d1 / (d2 + 1e-12)


def fuse(M, groups):
    """Temporal fusion: average per-identity distances over a group of probes (frames)."""
    return np.stack([M[g].mean(0) for g in groups])


def groups_of(labels, folds, size, by_fold=True):
    out = []
    for l in np.unique(labels):
        for f in (np.unique(folds) if by_fold else [None]):
            idx = np.flatnonzero((labels == l) & ((folds == f) if by_fold else True))
            for s in range(0, len(idx) - size + 1, size):
                out.append(idx[s:s + size])
    return out


def openset_scores(D, labels, k=5, fusion=1):
    n_ids = labels.max() + 1
    folds = folds_of(labels, k)
    Mg = identity_min(D, labels, lambda i: folds != folds[i], n_ids)
    Mi = identity_min(D, labels, lambda i: labels != labels[i], n_ids)   # own identity absent
    gl = labels
    if fusion > 1:
        gg = groups_of(labels, folds, fusion)
        Mg, gl = fuse(Mg, gg), labels[[g[0] for g in gg]]
        ig = groups_of(labels, folds, fusion, by_fold=False)
        Mi = fuse(Mi, ig)
    gid, gd, gr = decisions(Mg)
    _, idd, ir = decisions(Mi)
    return dict(g_correct=gid == gl, g_d=gd, g_r=gr, i_d=idd, i_r=ir)


def frr_at_far(s, rule, far_max=0.01):
    """Most lenient threshold(s) keeping FAR <= far_max; returns (FRR, thresholds, wrong-student rate).
    FRR counts genuine probes that are rejected OR given to the wrong student (eq. 6 of the report)."""
    gd, gr, gc, idd, ir = s["g_d"], s["g_r"], s["g_correct"], s["i_d"], s["i_r"]
    best = (1.0, None, 0.0)
    td = np.quantile(np.concatenate([gd, idd]), np.linspace(0, 1, 201))
    tr = np.linspace(0.5, 1.0, 101)
    cand = {"distance": [(t, 9) for t in td], "ratio": [(9e9, r) for r in tr],
            "distance+ratio": [(t, r) for t in td for r in tr]}[rule]
    for t, r in cand:
        far = ((idd <= t) & (ir <= r)).mean()
        if far > far_max:
            continue
        acc = (gd <= t) & (gr <= r)
        frr = 1 - (acc & gc).mean()
        if frr < best[0]:
            best = (frr, (float(t), float(r)), float((acc & ~gc).mean()))
    return best


def far_frr_curve(s, rule, n=200):
    gd, gr, gc, idd, ir = s["g_d"], s["g_r"], s["g_correct"], s["i_d"], s["i_r"]
    if rule == "distance":
        g, i = gd, idd
    else:
        g, i = gr, ir
    ts = np.quantile(np.concatenate([g, i]), np.linspace(0, 1, n))
    far = np.array([(i <= t).mean() for t in ts])
    frr = np.array([1 - ((g <= t) & gc).mean() for t in ts])
    return far, frr


def lbph_baseline(chips, labels, clahe, k=5, threshold=70):
    """The stage-1 system exactly: OpenCV LBPH, absolute threshold."""
    chips = illum(chips, "clahe", clahe)
    folds = folds_of(labels, k)
    gd, gc, idd = [], [], []
    for f in range(k):
        r = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        tr = folds != f
        r.train([chips[i] for i in np.flatnonzero(tr)], labels[tr].astype(np.int32))
        for i in np.flatnonzero(~tr):
            l, d = r.predict(chips[i])
            gd.append(d)
            gc.append(l == labels[i])
    for l in np.unique(labels):
        r = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        tr = labels != l
        r.train([chips[i] for i in np.flatnonzero(tr)], labels[tr].astype(np.int32))
        idd += [r.predict(chips[i])[1] for i in np.flatnonzero(~tr)]
    gd, gc, idd = np.array(gd), np.array(gc), np.array(idd)
    s = dict(g_correct=gc, g_d=gd, g_r=np.zeros_like(gd), i_d=idd, i_r=np.zeros_like(idd))
    return s, {"accuracy": float(gc.mean()),
               "far_at_70": float((idd <= threshold).mean()),
               "far_at_50": float((idd <= 50).mean()),
               "frr_at_50": float(1 - ((gd <= 50) & gc).mean())}


# =============================================================================== experiments
def exp_weights(lfw_root, engine, out):
    people = lfw_people(lfw_root, 10, max_images=10, max_count=19)   # disjoint from >=20 sets
    chips, labels, names, skipped = lfw_chips(engine, people, "weights")
    clahe = engine._clahe
    res = {"people": len(names), "images": int(len(chips))}
    for kw in ({"scales": ((8, 1), (8, 2))}, {"scales": ((8, 1), (8, 2)), "ternary": True}):
        desc = lbp.LBPDescriptor(**kw)
        how = "tantriggs" if kw.get("ternary") else "clahe"
        F = features(illum(chips, how, clahe), kw)
        w = lbp.learn_cell_weights(F, labels, pairs=20000)
        config.WEIGHTS_DIR.mkdir(exist_ok=True)
        np.save(config.WEIGHTS_DIR / f"cell_weights_{desc.signature}.npy", w)
        res[desc.signature] = w.tolist()
        print(f"weights {desc.signature}: learned on {len(names)} people / {len(chips)} images")
    (out / "weights.json").write_text(json.dumps(res))
    return res


def weights_for(kw):
    desc = lbp.LBPDescriptor(**kw)
    return np.load(config.WEIGHTS_DIR / f"cell_weights_{desc.signature}.npy")


def exp_lfw(lfw_root, engine, out, tag, people):
    chips, labels, names, skipped = lfw_chips(engine, people, tag)
    print(f"\nLFW[{tag}]: {len(names)} people, {len(chips)} chips, {skipped} skipped (no face)")
    res = {"people": len(names), "images": int(len(chips)), "skipped": skipped, "variants": {}}
    s, base = lbph_baseline(chips, labels, engine._clahe)
    base["frr@far1"] = {r: frr_at_far(s, r)[0] for r in ("distance",)}
    res["baseline_lbph"] = base
    print(f"  OpenCV LBPH: acc {base['accuracy']:.3f}  FAR@70 {base['far_at_70']:.3f}  "
          f"FAR@50 {base['far_at_50']:.3f} FRR@50 {base['frr_at_50']:.3f}  FRR@FAR1% {base['frr@far1']['distance']:.3f}")
    curves = {"OpenCV LBPH (distance)": far_frr_curve(s, "distance")}
    for name, (kw, how, weighted) in VARIANTS.items():
        F = features(illum(chips, how, engine._clahe), kw)
        w = weights_for(kw) if weighted else np.ones(F.shape[1], np.float32)
        D = distance_matrix(F, w)
        s1 = openset_scores(D, labels)
        folds = folds_of(labels)
        nf = min(5, min(int(((labels == l) & (folds == f)).sum()) for l in np.unique(labels) for f in range(5)))
        s5 = openset_scores(D, labels, fusion=nf)
        res["fusion_frames"] = nf
        v = {"accuracy": float(s1["g_correct"].mean())}
        for rule in ("distance", "ratio", "distance+ratio"):
            frr, th, wrong = frr_at_far(s1, rule)
            v[f"frr@far1 {rule}"] = frr
            v[f"thresholds {rule}"] = th
            v[f"wrong {rule}"] = wrong
        frr5, th5, _ = frr_at_far(s5, "distance+ratio")
        v["frr@far1 distance+ratio multi-frame"] = frr5
        v["thresholds multi-frame"] = th5
        v["accuracy multi-frame"] = float(s5["g_correct"].mean())
        res["variants"][name] = v
        print(f"  {name:38s} acc {v['accuracy']:.3f} | FRR@FAR1%  dist {v['frr@far1 distance']:.3f}  "
              f"ratio {v['frr@far1 ratio']:.3f}  both {v['frr@far1 distance+ratio']:.3f}  {nf}-frame {frr5:.3f}")
        curves[f"{name} (distance)"] = far_frr_curve(s1, "distance")
        curves[f"{name} (ratio)"] = far_frr_curve(s1, "ratio")
        curves[f"{name} (ratio, multi-frame)"] = far_frr_curve(s5, "ratio")
    (out / f"lfw_{tag}.json").write_text(json.dumps(res, indent=2))
    return res, curves


def exp_yale(yale_root, engine, out):
    imgs, subj, subset = yale_images(yale_root)
    chips = [cv2.resize(im, config.FACE_SIZE) for im in imgs]
    labels = np.unique(subj, return_inverse=True)[1]
    gallery = subset == 1
    res = {"images": len(chips), "subjects": int(labels.max() + 1),
           "per_subset_count": {str(s): int((subset == s).sum()) for s in range(1, 6)}, "variants": {}}
    print(f"\nYale B: {len(chips)} images, {labels.max() + 1} subjects; gallery = subset 1 ({gallery.sum()})")
    configs = {
        "LBP 256-bin, no normalisation (LBPH-like)": (dict(scales=((8, 1),), uniform=False), "none"),
        "uLBP(8,1)+(8,2), CLAHE": (dict(scales=((8, 1), (8, 2))), "clahe"),
        "uLBP(8,1)+(8,2), Tan-Triggs": (dict(scales=((8, 1), (8, 2))), "tantriggs"),
        "LTP(8,1)+(8,2), Tan-Triggs": (dict(scales=((8, 1), (8, 2)), ternary=True), "tantriggs"),
    }
    for name, (kw, how) in configs.items():
        F = features(illum(chips, how, engine._clahe), kw)
        D = cross_distance(F[~gallery], F[gallery], np.ones(F.shape[1], np.float32))
        pred = labels[gallery][np.argmin(D, 1)]
        ok = pred == labels[~gallery]
        sub = subset[~gallery]
        res["variants"][name] = {str(s): float(ok[sub == s].mean()) for s in range(2, 6)}
        print(f"  {name:44s}", " ".join(f"S{s} {ok[sub == s].mean():.3f}" for s in range(2, 6)))
    (out / "yale.json").write_text(json.dumps(res, indent=2))
    return res


# =============================================================================== figures
def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#b5b4ae")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def fig_detection(det, figs):
    import matplotlib.pyplot as plt
    modes = ["haar", "lbp", "union", "consensus"]
    labels = ["Haar", "LBP cascade", "Union", "Consensus\n(ours)"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, key, title, fmt in zip(axes, ("recall", "false_per_frame", "ms_per_frame"),
                                   ("Faces found (recall)", "False boxes per frame", "Time per 640x500 frame (ms)"),
                                   ("{:.1%}", "{:.2f}", "{:.1f}")):
        vals = [det["mosaic"][m][key] for m in modes]
        bars = ax.bar(labels, vals, color=PALETTE[0], width=0.55, edgecolor="white", linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom",
                    fontsize=9, color=INK)
        _style(ax, title)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Detection on 250 LFW mosaics (4 faces each, 1000 faces)", x=0.01, ha="left", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "detection_modes.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    subsets = [str(s) for s in range(1, 6)]
    for k, (key, lab) in enumerate((("haar", "Haar"), ("haar+clahe", "Haar + CLAHE"),
                                    ("consensus", "Consensus"), ("consensus+clahe", "Consensus + CLAHE"))):
        ys = [det["yale"][key][s] for s in subsets]
        ax.plot(range(1, 6), ys, color=PALETTE[k], lw=2, marker="o", ms=6, label=lab,
                ls="-" if "clahe" in key else "--")
    _style(ax, "Detection vs lighting angle (Extended Yale B)", "lighting subset (1 = frontal ... 5 = > 77°)",
           "faces found")
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 1.03)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "detection_lighting.png", dpi=150)
    plt.close(fig)


def fig_det_curves(curves, figs, tag):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    pick = [("OpenCV LBPH (distance)", "OpenCV LBPH, distance threshold (stage 1)", 0, "--"),
            ("uLBP(8,1)+(8,2) weighted (distance)", "ours, distance threshold", 1, "--"),
            ("uLBP(8,1)+(8,2) weighted (ratio)", "ours, ratio test", 2, "-"),
            ("uLBP(8,1)+(8,2) weighted (ratio, multi-frame)", "ours, ratio test + 4-frame fusion", 6, "-")]
    for key, lab, c, ls in pick:
        far, frr = curves[key]
        ax.plot(np.maximum(far, 1e-3), frr, color=PALETTE[c], lw=2, ls=ls, label=lab)
    ax.axvline(0.01, color="#9a998f", lw=1, ls=":")
    ax.text(0.0105, 0.97, "FAR = 1%", color=INK2, fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_ylim(0, 1)
    _style(ax, "Strangers accepted vs students missed (LFW)", "FAR: strangers accepted (log)",
           "FRR: students rejected or misidentified")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(figs / f"far_frr_{tag}.png", dpi=150)
    plt.close(fig)


def fig_yale(yale, figs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    names = list(yale["variants"])
    x = np.arange(2, 6)
    wbar = 0.2
    for k, n in enumerate(names):
        ys = [yale["variants"][n][str(s)] for s in x]
        ax.bar(x + (k - 1.5) * wbar, ys, wbar, color=PALETTE[k], label=n, edgecolor="white", linewidth=1.5)
    _style(ax, "Recognition under lighting (Yale B, gallery = subset 1)", "probe subset (harsher lighting ->)",
           "rank-1 accuracy")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    fig.savefig(figs / "yale_recognition.png", dpi=150)
    plt.close(fig)


def fig_weights(engine, lfw_root, figs):
    import matplotlib.pyplot as plt
    kw = {"scales": ((8, 1), (8, 2))}
    w = weights_for(kw).reshape(2, *config.LBP_GRID)
    z = np.load(CACHE / "lfw_weights.npz")
    mean_face = engine._clahe.apply(np.mean(z["chips"], 0).astype(np.uint8))
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    axes[0].imshow(mean_face, cmap="gray")
    axes[0].set_title("mean training face", fontsize=10, color=INK, loc="left")
    for ax, k, t in zip(axes[1:], range(2), ("weights, LBP(8,1)", "weights, LBP(8,2)")):
        ax.imshow(mean_face, cmap="gray", extent=(0, 8, 8, 0))
        im = ax.imshow(w[k], cmap="Blues", alpha=0.75, extent=(0, 8, 8, 0), vmin=0, vmax=w.max())
        for (i, j), v in np.ndenumerate(w[k]):
            ax.text(j + 0.5, i + 0.5, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > w.max() * 0.55 else INK)
        ax.set_title(t, fontsize=10, color=INK, loc="left")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[1:], shrink=0.8, label="Fisher weight (mean 1)")
    fig.savefig(figs / "cell_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_pipeline(engine, lfw_root, figs):
    """Each stage of the pipeline on one LFW image."""
    import matplotlib.pyplot as plt
    img = cv2.imread(str(Path(lfw_root) / "George_W_Bush" / "George_W_Bush_0010.jpg"))
    gray = engine.gray(img)
    det = engine.detection_image(img)
    view = cv2.cvtColor(det, cv2.COLOR_GRAY2RGB)
    for b in engine.detector.run_haar(det):
        cv2.rectangle(view, b[:2], (b[0] + b[2], b[1] + b[3]), (42, 120, 214), 2)
    for b in engine.detector.run_lbp(det):
        cv2.rectangle(view, b[:2], (b[0] + b[2], b[1] + b[3]), (235, 104, 52), 2)
    face = max(engine.detect(img), key=lambda f: f.box[2] * f.box[3])
    chip = engine.describe(img, face)
    codes = lbp.lbp_codes(chip, 8, 1)
    ucodes = lbp.uniform_table(8)[codes]
    hist = lbp.LBPDescriptor()(chip)
    panels = [(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "(a) input"), (gray, "(b) blur + grayscale"),
              (det, "(c) CLAHE"), (view, "(d) Haar (blue) / LBP (orange)"),
              (chip, "(e) 200x200 chip + CLAHE"), (codes, "(f) LBP(8,1) codes"),
              (ucodes, "(g) uniform LBP (59 bins)")]
    fig, axes = plt.subplots(1, 8, figsize=(17, 2.6))
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t, fontsize=8.5, color=INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[7].bar(np.arange(59), hist[27], color=PALETTE[0], width=0.8)
    _style(axes[7], "")
    axes[7].set_title("(h) one cell, 59 bins", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(figs / "pipeline_stages.png", dpi=150)
    plt.close(fig)


# =============================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw", type=Path, default=Path.home() / "datasets/lfw_home/lfw_funneled")
    ap.add_argument("--yale", type=Path, default=Path.home() / "datasets/yale_src/CroppedYale")
    ap.add_argument("--only", nargs="*", default=["detection", "weights", "lfw", "yale", "figures"])
    args = ap.parse_args()
    out, figs = config.RESULTS_DIR, config.FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    # LFW was assembled with a Viola-Jones detector: use Haar for chips so the
    # recognition experiments isolate the recogniser (and match the stage-1 trial).
    engine = FaceEngine(detector="haar")
    if "detection" in args.only:
        det = exp_detection(args.lfw, args.yale, out)
    if "weights" in args.only:
        exp_weights(args.lfw, engine, out)
    curves = {}
    if "lfw" in args.only:
        exp_lfw(args.lfw, engine, out, "stage1_10x30", lfw_people(args.lfw, 20, max_images=30, max_people=10))
        _, curves = exp_lfw(args.lfw, engine, out, "all62_20", lfw_people(args.lfw, 20, max_images=20))
        np.savez(CACHE / "curves.npz", **{k: np.stack(v) for k, v in curves.items()})
    if "yale" in args.only:
        exp_yale(args.yale, engine, out)
    if "figures" in args.only:
        det = json.loads((out / "detection.json").read_text())
        fig_detection(det, figs)
        if not curves:
            z = np.load(CACHE / "curves.npz")
            curves = {k: tuple(z[k]) for k in z.files}
        fig_det_curves(curves, figs, "all62_20")
        fig_yale(json.loads((out / "yale.json").read_text()), figs)
        fig_weights(engine, args.lfw, figs)
        fig_pipeline(engine, args.lfw, figs)
        print(f"figures -> {figs}")


if __name__ == "__main__":
    main()
