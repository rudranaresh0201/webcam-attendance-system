// FaceRoll dashboard. Polls /api/status; no framework, no build step.
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const TABS = ["live", "register", "students", "records", "insights", "settings"];
let state = null, lastMsg = "", tab = null;

async function api(path, opts = {}) {
  const o = { ...opts };
  if (o.json !== undefined) { o.body = JSON.stringify(o.json); o.headers = { "Content-Type": "application/json" }; o.method ??= "POST"; delete o.json; }
  const r = await fetch(path, o);
  const data = r.headers.get("content-type")?.includes("json") ? await r.json() : null;
  if (!r.ok) throw Object.assign(new Error(data?.error || r.statusText), { data, status: r.status });
  return data;
}
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove("show"), 3800); }

// ------------------------------------------------------------------ tabs as faces of a cube
function show(name) {
  if (name === tab) return;
  const prev = tab ? $("#tab-" + tab) : null, next = $("#tab-" + name);
  const fwd = !tab || TABS.indexOf(name) > TABS.indexOf(tab);
  tab = name;
  $$(".channels a").forEach((a) => a.classList.toggle("on", a.dataset.tab === name));
  $$(".face").forEach((f) => f.classList.remove("in-r", "in-l", "out-r", "out-l"));
  if (prev) {
    prev.classList.add(fwd ? "out-r" : "out-l");
    prev.addEventListener("animationend", () => prev.classList.remove("on", "out-r", "out-l"), { once: true });
    setTimeout(() => prev.classList.remove("on", "out-r", "out-l"), 800);   // reduced-motion fallback
  }
  next.classList.add("on", fwd ? "in-r" : "in-l");
  next.addEventListener("animationend", () => next.classList.remove("in-r", "in-l"), { once: true });
  // keep a single MJPEG connection open
  $("#video").src = name === "live" ? "/stream.mjpg" : "";
  $("#regVideo").src = name === "register" ? "/stream.mjpg" : "";
  ({ students: loadStudents, records: loadRecords, insights: loadBench, settings: loadSettings })[name]?.();
  try { localStorage.setItem("tab", name); } catch {}
}
$$(".channels a").forEach((a) => (a.onclick = () => show(a.dataset.tab)));
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select, textarea")) return;
  const n = Number(e.key);
  if (n >= 1 && n <= TABS.length) show(TABS[n - 1]);
});

// ------------------------------------------------------------------ signal chain
function renderChain(s) {
  const st = s.settings, live = s.mode === "attendance";
  const det = { consensus: "Haar ∧ LBP", haar: "Haar V–J", lbp: "LBP cascade", union: "Haar ∨ LBP" }[st.DETECTOR];
  const ill = { clahe: "CLAHE 2.0 / 8×8", tantriggs: "Tan–Triggs", none: "none" }[st.ILLUMINATION];
  const blocks = [
    ["acquire", "640 px · mirror"], ["gauss", "3×3 σ auto"], ["luma", ".299R+.587G+.114B"], ["clahe", "clip 2 · 8×8"],
    ["detect", `${det} · n≥${st.MIN_NEIGHBORS}`], ["chip", "200² · " + ill],
    [st.LBP_TERNARY ? "ltp" : "u-lbp", "(8,1)+(8,2) · 59 bins"], ["χ² · w", "learned weights"],
    ["open set", `d≤${st.DIST_THRESHOLD} ∧ r≤${st.RATIO_THRESHOLD}`], ["track", `${st.CONFIRM_FRAMES} frames${st.LIVENESS ? " + blink" : ""}`],
    ["register", live ? "writing CSV" : "idle"],
  ];
  const html = blocks.map(([b, v], i) => `<div class="blk${(live && i === blocks.length - 1) ? " hot" : ""}${i === 4 && s.faces.length ? " hot" : ""}"><b>${esc(b)}</b><span>${esc(v)}</span></div>`).join('<div class="wire"></div>');
  if (html !== renderChain.h) { renderChain.h = html; $("#chain").innerHTML = html; }
}

// ------------------------------------------------------------------ polling
async function poll() {
  try { state = await api("/api/status"); render(state); } catch (e) { /* server restarting */ }
  setTimeout(poll, 400);
}

function gauge(label, v, thr, max) {
  const ok = v <= thr;
  return `<div class="gauge ${ok ? "ok" : "no"}"><span>${label}</span>
    <div class="g"><em style="width:${Math.min(100, (v / max) * 100)}%;background:var(${ok ? "--green" : "--red"})"></em><i style="left:${Math.min(100, (thr / max) * 100)}%"></i></div>
    <b>${v.toFixed(3)}</b></div>`;
}

function render(s) {
  $("#modePill").textContent = s.mode;
  $("#modePill").className = s.mode;
  $("#modelInfo").textContent = s.model_ok ? `${s.students} enrolled · ${s.recognizer === "openset" ? "open-set LBP" : "LBPH"}` : (s.training ? "training…" : "untrained");
  $("#kFps").textContent = s.camera_ok ? `${s.fps.toFixed(1)} fps` : "no camera";
  $("#dateLine").textContent = s.date;
  $("#btnStart").hidden = s.mode === "attendance";
  $("#btnStop").hidden = s.mode !== "attendance";
  $("#recDot").hidden = s.mode !== "attendance";
  $("#confN").textContent = s.settings.CONFIRM_FRAMES;
  renderChain(s);

  const present = s.roster.filter((r) => r.status === "Present").length;
  $("#kPresent").innerHTML = `${present}<small>/${s.roster.length}</small>`;
  $("#kFaces").textContent = s.faces.length;
  $("#kUnknown").textContent = s.unknown_seen;
  $("#rosterCount").textContent = s.roster.length ? `${present} of ${s.roster.length}` : "";

  const rk = JSON.stringify(s.roster);
  if (rk !== render.rk) {
    render.rk = rk;
    $("#roster").innerHTML = s.roster.length ? s.roster.map((r) => `
      <li class="${r.status}"><img class="avatar" src="/api/students/${encodeURIComponent(r.roll)}/photo" alt="" onerror="this.style.visibility='hidden'">
      <div><b>${esc(r.name)}</b><span class="r">${esc(r.roll)}${r.time ? " · " + esc(r.time) : ""}</span></div>
      <span class="st ${r.status}">${r.status}</span></li>`).join("") : `<li class="dim">Nobody enrolled yet &mdash; see <b>02 Enrol</b>.</li>`;
  }
  const fk = JSON.stringify(s.events);
  if (fk !== render.fk) {
    render.fk = fk;
    $("#feed").innerHTML = s.events.length ? s.events.map((e) => `<li><time>${esc(e.time)}</time>${esc(e.name)} credited present</li>`).join("")
      : `<li class="dim">${s.mode === "attendance" ? "Session open. Waiting for the first face…" : "Begin a session to record arrivals."}</li>`;
  }

  // decision readout for each recognised face
  const st = s.settings, faces = s.faces.filter((f) => f.dist !== null);
  $("#faceTable").innerHTML = faces.length ? faces.map((f) => `
    <div class="fr">
      <div class="fr-head"><b>${esc(f.name || "Unknown")}</b><span>${esc(f.state)} · ${esc(f.sources)} · ${f.streak}/${st.CONFIRM_FRAMES}${st.LIVENESS ? (f.blinked ? " · blink ✓" : " · no blink") : ""}</span></div>
      ${gauge("d₁", f.dist, st.DIST_THRESHOLD, Math.max(0.8, f.dist))}
      ${gauge("d₁ / d₂", f.ratio, st.RATIO_THRESHOLD, 1.0)}
    </div>`).join("") : `<p class="dim">${s.model_ok ? "No face in view." : "Enrol students to see matching distances."}</p>`;

  if (tab === "live" && !render.loading) {
    // preload so the stack never flashes empty
    render.loading = true;
    const url = "/explain.jpg?t=" + Date.now(), im = new Image();
    im.onload = im.onerror = () => { $$(".layer .img").forEach((d) => (d.style.backgroundImage = `url(${url})`)); render.loading = false; };
    im.src = url;
  }

  // enrolment
  const reg = s.register, active = s.mode === "register" && reg;
  $("#btnReg").disabled = active || s.training;
  $("#btnRegCancel").hidden = !active;
  if (active) {
    $("#ringTxt").textContent = `${reg.saved}/${reg.target}`;
    $("#ringProg").style.strokeDashoffset = 97.4 * (1 - reg.saved / reg.target);
    $("#regHint").textContent = reg.hint;
  } else if (render.wasReg) {
    $("#ringProg").style.strokeDashoffset = 0;
    $("#regHint").textContent = s.training ? "Captured. Training…" : "Done. Enrol the next student, or go to Live.";
  }
  render.wasReg = active;
  if (s.message && s.message !== lastMsg) { lastMsg = s.message; toast(s.message); }
}

$$("#stackLegend li").forEach((li) => {
  const layer = $$(".layer")[+li.dataset.l];
  li.onmouseenter = () => layer.classList.add("lift");
  li.onmouseleave = () => layer.classList.remove("lift");
});
$("#btnFlat").onclick = () => { const f = $("#stack").classList.toggle("flat"); $("#btnFlat").textContent = f ? "Explode" : "Lay flat"; };
$("#btnStart").onclick = async () => { try { await api("/api/attendance/start", { json: {} }); } catch (e) { toast(e.message); } };
$("#btnStop").onclick = () => api("/api/attendance/stop", { json: {} });

// ------------------------------------------------------------------ enrolment
$$(".seg button").forEach((b) => (b.onclick = () => {
  $$(".seg button").forEach((x) => x.classList.toggle("on", x === b));
  $("#srcCam").hidden = b.dataset.src !== "cam";
  $("#srcUpload").hidden = b.dataset.src !== "upload";
}));
async function startReg(overwrite = false) {
  try { await api("/api/register/start", { json: { roll: $("#rRoll").value, name: $("#rName").value, overwrite } }); }
  catch (e) {
    if (e.status === 409 && confirm(e.message + " Replace their images?")) return startReg(true);
    toast(e.message);
  }
}
$("#btnReg").onclick = () => startReg();
$("#btnRegCancel").onclick = () => api("/api/register/cancel", { json: {} });
$("#btnUpload").onclick = async () => {
  const fd = new FormData();
  fd.append("roll", $("#rRoll").value); fd.append("name", $("#rName").value);
  for (const f of $("#rFiles").files) fd.append("photos", f);
  $("#btnUpload").disabled = true;
  try { const r = await api("/api/register/upload", { method: "POST", body: fd }); toast(`Imported ${r.saved} photographs. Training…`); }
  catch (e) { toast(e.message); }
  $("#btnUpload").disabled = false;
};

// ------------------------------------------------------------------ students (flip to see captures)
async function loadStudents() {
  const list = await api("/api/students");
  const ph = (r, i) => `/api/students/${encodeURIComponent(r)}/photo?i=${i}`;
  $("#studentCards").innerHTML = list.length ? list.map((s) => `
    <div class="flip" tabindex="0">
      <div class="front"><img src="${ph(s.roll, 0)}" alt="">
        <div class="body"><b>${esc(s.name)}</b><span>${esc(s.roll)} · ${s.images} img${s.trained ? "" : " · untrained"}</span></div></div>
      <div class="back"><div class="thumbs">${[3, 7, 11, 15, 19, 23, 26, 28, 29].filter((i) => i < s.images).map((i) => `<img loading="lazy" src="${ph(s.roll, i)}" alt="">`).join("")}</div>
        <div class="row"><span>${esc(s.roll)}</span><button data-del="${esc(s.roll)}">Remove</button></div></div>
    </div>`).join("") : `<div class="panel empty">Nobody enrolled yet.</div>`;
  $$(".flip").forEach((c) => (c.onclick = (e) => { if (!e.target.dataset.del) c.classList.toggle("turned"); }));
  $$("[data-del]").forEach((b) => (b.onclick = async () => {
    if (!confirm(`Remove ${b.dataset.del} and all their face images?`)) return;
    await api("/api/students/" + encodeURIComponent(b.dataset.del), { method: "DELETE" });
    loadStudents();
  }));
}
$("#btnTrain").onclick = () => api("/api/train", { json: {} }).then(() => toast("Training…"));

// ------------------------------------------------------------------ records
async function loadRecords() {
  const recs = await api("/api/records");
  $("#recordList").innerHTML = recs.length ? recs.map((r, i) => `
    <details class="panel reg" ${i === 0 ? "open" : ""}><summary><b>${esc(r.date)}</b>
      <span class="frac-big">${r.present} / ${r.total} present</span><span class="grow"></span>
      <a href="/api/records/${esc(r.date)}.csv"><button>CSV</button></a></summary>
      <table><thead><tr><th>Roll</th><th>Name</th><th>Status</th><th>Time</th></tr></thead><tbody>
      ${r.rows.map((x) => `<tr><td class="m">${esc(x.roll)}</td><td>${esc(x.name)}</td><td><span class="st ${esc(x.status)}">${esc(x.status)}</span></td><td class="m">${esc(x.time)}</td></tr>`).join("")}
      </tbody></table></details>`).join("") : `<div class="panel empty">No sessions recorded yet.</div>`;
}

// ------------------------------------------------------------------ evidence
async function loadBench() {
  const b = await api("/api/benchmarks");
  const pct = (x) => (100 * x).toFixed(1) + "%";
  const k = [];
  let det, lfw, v, y, y4;
  if (b.detection) {
    det = b.detection.mosaic;
    k.push(["false boxes / frame", det.consensus.false_per_frame.toFixed(2), `Haar alone ${det.haar.false_per_frame.toFixed(2)} &middot; recall ${pct(det.consensus.recall)}`]);
  }
  if (b.lfw_all62_20) {
    lfw = b.lfw_all62_20; v = lfw.variants["uLBP(8,1)+(8,2) weighted"];
    k.push(["LFW rank-1, 62 people", pct(v.accuracy), `OpenCV LBPH ${pct(lfw.baseline_lbph.accuracy)}`]);
    k.push(["students missed @ FAR 1%", pct(v["frr@far1 distance+ratio multi-frame"]), `${lfw.fusion_frames}-frame fusion &middot; LBPH ${pct(lfw.baseline_lbph["frr@far1"].distance)}`]);
  }
  if (b.yale) {
    y = b.yale.variants; y4 = y["LTP(8,1)+(8,2), Tan-Triggs"]["4"];
    k.push(["Yale B side light (S4)", pct(y4), `raw LBP ${pct(y["LBP 256-bin, no normalisation (LBPH-like)"]["4"])}`]);
  }
  $("#benchKpis").innerHTML = k.map(([a, val, sub]) => `<div class="ro"><span>${a}</span><b>${val}</b><p>${sub}</p></div>`).join("")
    || `<div class="ro"><span>no results</span><p>Run <code>python benchmark.py</code>.</p></div>`;

  const figs = [
    ["pipeline_stages.png", "The chain, stage by stage", "Fig. 1", true,
      `<p>An LFW photograph through every stage: Gaussian smoothing, luma conversion, CLAHE, the two cascades (Haar in blue, LBP in orange), the normalised chip, raw LBP codes, the 59-bin uniform mapping and the histogram of a single cell.</p>
       <p>A uniform pattern has at most two 0/1 transitions around the circle. These 58 patterns are spots, edges, corners and line ends, and they cover most of a face's texture. Every other pattern shares one bin, which shrinks each cell from 256 bins to 59.</p>`],
    ["detection_modes.png", "Two cascades, one vote", "Fig. 2", true, det ? `
       <div class="num">${det.haar.false_per_frame.toFixed(2)} &rarr; ${det.consensus.false_per_frame.toFixed(2)}</div>
       <p>False boxes per 640&times;500 frame on 250 mosaics of four LFW faces.</p>
       <p>Haar rectangle features (sums over regions) and LBP features (sign comparisons) fail on <b>different</b> backgrounds. A box is accepted when both cascades agree, or when one of them is backed by an eye pair or by &ge;20 Haar neighbours. Recall goes from ${pct(det.haar.recall)} to ${pct(det.consensus.recall)}.</p>` : ""],
    ["far_frr_all62_20.png", "Strangers versus students", "Fig. 3", false, lfw ? `
       <div class="num">${pct(lfw.baseline_lbph["frr@far1"].distance)} &rarr; ${pct(v["frr@far1 distance+ratio"])} &rarr; ${pct(v["frr@far1 distance+ratio multi-frame"])}</div>
       <p>Students missed when at most 1% of strangers may get in (LFW, 62 people, leave-one-person-out impostors). In stage 1 the absolute distance threshold alone could not separate the two groups.</p>
       <p><b>Ratio test:</b> a stranger is about equally far from everyone, while a student is much closer to their own templates than to the runner-up. So the system accepts only when d<sub>1</sub>/d<sub>2</sub> &le; &tau;<sub>r</sub>. <b>Fusion</b> averages per-identity distances over ${lfw.fusion_frames} frames, which is what the tracker supplies in the live system.</p>` : ""],
    ["yale_recognition.png", "Lighting from the side", "Fig. 4", false, y ? `
       <div class="num">${pct(y["LBP 256-bin, no normalisation (LBPH-like)"]["5"])} &rarr; ${pct(y["LTP(8,1)+(8,2), Tan-Triggs"]["5"])}</div>
       <p>Extended Yale B, worst lighting subset (light more than 77&deg; off-axis). Gallery: subset 1 only.</p>
       <p>Tan&ndash;Triggs applies gamma 0.2, then a difference of Gaussians (&sigma; = 1, 2) that removes slow shading, then robust contrast equalisation. Local Ternary Patterns add a dead zone of &plusmn;5 grey levels, so noise in flat, dark regions stops flipping bits.</p>` : ""],
    ["cell_weights.png", "Which parts of the face matter", "Fig. 5", false, `
       <p>Each of the 128 cells (8&times;8 grid &times; two radii) is weighted by its Fisher ratio (&mu;<sub>b</sub>&minus;&mu;<sub>w</sub>)&sup2;/(&sigma;<sub>b</sub>&sup2;+&sigma;<sub>w</sub>&sup2;), measured on χ² distances between pairs of the same person and pairs of different people.</p>
       <p>The weights were learned on 96 LFW identities (937 images) that are <b>never used in testing</b>. They recover Ahonen's hand-set map: eyes and brows weigh ~3&times;, while corners, hair and background fall to ~0.</p>`],
    ["detection_lighting.png", "Detection as the lamp moves", "Fig. 6", false, `
       <p>Faces found per Yale B lighting subset. The Yale crops are tight, which suits the LBP cascade. That is why consensus with eye rescue beats Haar alone here.</p>
       <p>Past 50&deg; neither detector copes: half the face is in shadow. This is the practical limit of a frontal classical detector.</p>`],
  ];
  $("#figs").innerHTML = figs.map(([src, title, n, span, back]) => `
    <div class="fig${span ? " span" : ""}" tabindex="0">
      <div class="front"><h3>${title}<small>${n} &middot; turn over</small></h3><img src="/figures/${src}" alt="${esc(title)}"></div>
      <div class="back"><h3>${title}</h3>${back}</div>
    </div>`).join("");
  $$(".fig").forEach((f) => (f.onclick = () => f.classList.toggle("turned")));
}

// ------------------------------------------------------------------ controls
const KEYS = ["DETECTOR", "ILLUMINATION", "LBP_TERNARY", "DIST_THRESHOLD", "RATIO_THRESHOLD", "CONFIRM_FRAMES", "LIVENESS", "MIN_NEIGHBORS"];
async function loadSettings() {
  const s = await api("/api/settings");
  for (const k of KEYS) {
    const el = $("#s" + k); if (!el) continue;
    if (el.type === "checkbox") el.checked = s[k]; else el.value = s[k];
    const o = $("#o" + k); if (o) { o.textContent = el.value; el.oninput = () => (o.textContent = el.value); }
  }
  $("#camOn").checked = state?.camera_on ?? true;
}
$("#btnSave").onclick = async () => {
  const body = {};
  for (const k of KEYS) { const el = $("#s" + k); body[k] = el.type === "checkbox" ? el.checked : (el.type === "range" ? Number(el.value) : el.value); }
  await api("/api/settings", { json: body });
  toast("Applied.");
};
let suggestion = null;
$("#btnCal").onclick = async () => {
  $("#btnCal").disabled = true; $("#calLog").hidden = false; $("#calLog").textContent = "Running…";
  try { const r = await api("/api/calibrate", { json: {} }); $("#calLog").textContent = r.log; suggestion = r.result; $("#btnCalApply").hidden = !suggestion; }
  catch (e) { $("#calLog").textContent = e.message; }
  $("#btnCal").disabled = false;
};
$("#btnCalApply").onclick = async () => {
  await api("/api/settings", { json: { DIST_THRESHOLD: +suggestion.dist.toFixed(3), RATIO_THRESHOLD: +suggestion.ratio.toFixed(2) } });
  loadSettings(); toast("Thresholds applied.");
};
$("#camOn").onchange = (e) => api("/api/camera", { json: { on: e.target.checked } });

let first = "live";
try { first = localStorage.getItem("tab") || "live"; } catch {}
$("#tab-live").classList.remove("on");
show(TABS.includes(first) ? first : "live");
poll();
