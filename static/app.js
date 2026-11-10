// FaceRoll dashboard. Polls /api/status; plain JS, no build step.
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

// ------------------------------------------------------------------ tabs turn like the faces of a cube
function show(name) {
  if (name === tab) return;
  const prev = tab ? $("#tab-" + tab) : null, next = $("#tab-" + name);
  const fwd = !tab || TABS.indexOf(name) > TABS.indexOf(tab);
  tab = name;
  $$(".tabs a").forEach((a) => a.classList.toggle("on", a.dataset.tab === name));
  $$(".face").forEach((f) => f.classList.remove("in-r", "in-l", "out-r", "out-l"));
  if (prev) {
    prev.classList.add(fwd ? "out-r" : "out-l");
    const done = () => prev.classList.remove("on", "out-r", "out-l");
    prev.addEventListener("animationend", done, { once: true });
    setTimeout(done, 800);
  }
  next.classList.add("on", fwd ? "in-r" : "in-l");
  next.addEventListener("animationend", () => next.classList.remove("in-r", "in-l"), { once: true });
  $("#video").src = name === "live" ? "/stream.mjpg" : "";        // one camera stream at a time
  $("#regVideo").src = name === "register" ? "/stream.mjpg" : "";
  ({ students: loadStudents, records: loadRecords, insights: loadBench, settings: loadSettings })[name]?.();
  try { localStorage.setItem("tab", name); } catch {}
}
$$(".tabs a").forEach((a) => (a.onclick = () => show(a.dataset.tab)));

// ------------------------------------------------------------------ "what the computer sees"
const VIEW_NOTES = {
  camera: "The normal camera picture, with the system's decision drawn on top.",
  contrast: "Slightly blurred to remove grain, turned grey, then contrast evened out tile by tile (CLAHE). This is the picture the face finders actually search.",
  texture: "Every pixel replaced by its LBP code: 8 \"is my neighbour brighter?\" answers. Turn a lamp on or off and the photo changes a lot, but these codes barely change. That is why recognition survives a dim room.",
  votes: "What each face finder says on its own, before they vote. Haar (blue) looks for light-and-dark blocks, such as eyes being darker than cheeks. LBP (orange) looks for texture. A face is kept when they agree.",
};
function setView(v, send = true) {
  $$("#viewSeg button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
  $("#viewNote").textContent = VIEW_NOTES[v];
  if (send) api("/api/view", { json: { view: v } });
}
$$("#viewSeg button").forEach((b) => (b.onclick = () => setView(b.dataset.view)));

// ------------------------------------------------------------------ plain-language narration
function narrate(s) {
  const st = s.settings, faces = s.faces;
  if (!s.camera_ok) return ["no", "The camera isn't available.", "Close any other app using the webcam, or turn the camera on in Settings."];
  if (s.mode === "register") return ["wait", "Enrolling…", "Follow the instructions on the Enrol tab."];
  if (!faces.length) return ["", "I don't see anyone.", "Face the camera from about an arm's length away, with light on your face."];
  const f = faces.reduce((a, b) => (a.box[2] * a.box[3] > b.box[2] * b.box[3] ? a : b));
  const more = faces.length > 1 ? `I can see ${faces.length} faces; this is about the biggest one. ` : "";
  const rec = s.mode === "attendance";
  if (f.state === "face" || !s.model_ok) return ["wait", "I can see a face, but nobody is enrolled yet.", more + "Go to Enrol and add yourself; then I'll know who you are."];
  if (f.state === "unknown") {
    if (f.dist > st.DIST_THRESHOLD)
      return ["no", "I can see a face, but I don't recognise it.",
        more + `The closest enrolled student is ${f.nearest}, but not close enough: a difference of ${f.dist.toFixed(2)} against a limit of ${st.DIST_THRESHOLD}.`];
    if (f.runner_up === "crowd")
      return ["no", "I can see a face, but I don't recognise it.",
        more + `It's no closer to ${f.nearest} than to faces of people who aren't in this class (ratio ${f.ratio.toFixed(2)}, limit ${st.RATIO_THRESHOLD}), so I won't guess.`];
    return ["no", "I can see a face, but I'm not sure who it is.",
      more + `It's close to ${f.nearest}, but almost as close to ${f.runner_up} (ratio ${f.ratio.toFixed(2)}, limit ${st.RATIO_THRESHOLD}), so I won't guess.`];
  }
  if (f.state === "verifying")
    return ["wait", `Recognising ${f.name}…`, more + `Matched in ${Math.min(f.streak, st.CONFIRM_FRAMES)} of the ${st.CONFIRM_FRAMES} frames needed.` + (rec ? "" : " Press Start attendance to record it.")];
  if (f.state === "blink") return ["wait", `${f.name}, please blink.`, more + "This checks that it's a real person and not a photo."];
  if (f.state === "marked") {
    const row = s.roster.find((r) => r.roll === f.roll);
    return ["ok", `${f.name} is marked present.`, more + (row?.time ? `Recorded at ${row.time}.` : "")];
  }
  return ["", "", ""];
}

function meter(title, v, thr, max, note) {
  const ok = v <= thr;
  return `<div class="meter">${title}
    <div class="bar"><em style="width:${Math.min(100, (v / max) * 100)}%;background:var(${ok ? "--green" : "--red"})"></em><i style="left:${Math.min(100, (thr / max) * 100)}%"></i></div>
    <div class="val"><span>${note}</span><span>${v.toFixed(2)} / limit ${thr}</span></div></div>`;
}

// ------------------------------------------------------------------ polling
async function poll() {
  try { state = await api("/api/status"); render(state); } catch (e) { /* server restarting */ }
  setTimeout(poll, 400);
}

function render(s) {
  const rec = s.mode === "attendance";
  $("#sessionText").textContent = rec ? "Recording attendance" : (s.training ? "Learning faces…" : "Not recording");
  $("#sessionText").className = rec ? "rec" : "";
  $("#btnStart").hidden = rec;
  $("#btnStop").hidden = !rec;
  if (render.view !== s.view) { render.view = s.view; setView(s.view, false); }

  const [cls, line, sub] = narrate(s);
  $("#say").className = "say " + cls;
  $("#say").innerHTML = `${esc(line)}<small>${esc(sub)}</small>`;

  const present = s.roster.filter((r) => r.status === "Present").length;
  $("#kPresent").textContent = present;
  $("#kTotal").textContent = s.roster.length;
  // private mode: on a projected screen, don't show who is absent
  const names = showNames();
  $("#roster").hidden = !names; $("#feedCard").hidden = !names; $("#privNote").hidden = names;
  const last = s.events[0];
  if (!names && last && last.time !== render.lastWelcome) {
    render.lastWelcome = last.time;
    $("#welcome").textContent = `Welcome, ${last.name}: marked present.`;
    $("#welcome").hidden = false;
    clearTimeout(render.wt); render.wt = setTimeout(() => ($("#welcome").hidden = true), 8000);
  }
  if (names) $("#welcome").hidden = true;
  const rk = JSON.stringify(s.roster);
  if (rk !== render.rk) {
    render.rk = rk;
    $("#roster").innerHTML = s.roster.length ? s.roster.map((r) => `
      <li class="${r.status}"><img class="avatar" src="/api/students/${encodeURIComponent(r.roll)}/photo" alt="" onerror="this.style.visibility='hidden'">
      <div><b>${esc(r.name)}</b><span class="r">${esc(r.roll)}${r.time ? " · " + esc(r.time) : ""}</span></div>
      <span class="tag ${r.status}">${r.status}</span></li>`).join("") : `<li class="empty-note">Nobody enrolled yet.</li>`;
  }
  const fk = JSON.stringify(s.events);
  if (fk !== render.fk) {
    render.fk = fk;
    $("#feed").innerHTML = s.events.length ? s.events.map((e) => `<li><time>${esc(e.time)}</time>${esc(e.name)}</li>`).join("")
      : `<li class="empty-note">${rec ? "Waiting for the first person…" : "Press Start attendance to begin."}</li>`;
  }

  const st = s.settings, faces = s.faces.filter((f) => f.dist !== null);
  $("#faceTable").innerHTML = faces.length ? faces.map((f) => `
    <div>
      <div class="fr-head"><b>${esc(f.name || "Not recognised")}</b><span>closest: ${esc(f.nearest || "–")}</span></div>
      ${meter("How different from the closest student", f.dist, st.DIST_THRESHOLD, Math.max(0.8, f.dist), "lower = more alike")}
      ${f.ratio > 0 ? meter(`Closest compared with the next face (${f.runner_up === "crowd" ? "someone outside the class" : esc(f.runner_up)})`, f.ratio, st.RATIO_THRESHOLD, Math.max(1.0, f.ratio), "lower = more certain")
        : `<p class="muted small">Nothing to compare against yet, so only the distance limit applies.</p>`}
    </div>`).join("") : `<p class="muted">${s.model_ok ? "When a face is in view, its numbers appear here." : "Enrol someone to see how matching works."}</p>`;

  if (tab === "live" && !render.loading) {
    render.loading = true;
    const url = "/explain.jpg?t=" + Date.now(), im = new Image();
    im.onload = im.onerror = () => { $$(".layer .img").forEach((d) => (d.style.backgroundImage = `url(${url})`)); render.loading = false; };
    im.src = url;
  }

  const reg = s.register, active = s.mode === "register" && reg;
  $("#btnReg").disabled = active || s.training;
  $("#btnRegCancel").hidden = !active;
  if (active) {
    $("#ringTxt").textContent = `${reg.saved}/${reg.target}`;
    $("#ringProg").style.strokeDashoffset = 97.4 * (1 - reg.saved / reg.target);
    $("#regHint").textContent = reg.hint;
  } else if (render.wasReg) {
    $("#ringProg").style.strokeDashoffset = 0;
    $("#regHint").textContent = s.training ? "Got it. Learning your face…" : "Done. Enrol the next person, or go to Live.";
  }
  render.wasReg = active;
  if (s.message && s.message !== lastMsg) { lastMsg = s.message; toast(s.message); }
}

function showNames() { try { return localStorage.getItem("showNames") === "1"; } catch { return false; } }
$("#showNames").checked = showNames();
$("#showNames").onchange = (e) => { try { localStorage.setItem("showNames", e.target.checked ? "1" : "0"); } catch {} };

$$("#steps li").forEach((li) => {
  const layer = $$(".layer")[+li.dataset.l];
  li.onmouseenter = () => layer.classList.add("lift");
  li.onmouseleave = () => layer.classList.remove("lift");
});
$("#btnFlat").onclick = () => { const f = $("#stack").classList.toggle("flat"); $("#btnFlat").textContent = f ? "Stack the layers in 3D" : "Lay the layers flat"; };
$("#btnStart").onclick = async () => { try { await api("/api/attendance/start", { json: {} }); show("live"); } catch (e) { toast(e.message); } };
$("#btnStop").onclick = () => api("/api/attendance/stop", { json: {} });

// ------------------------------------------------------------------ enrolment
$$("#tab-register .seg button").forEach((b) => (b.onclick = () => {
  $$("#tab-register .seg button").forEach((x) => x.classList.toggle("on", x === b));
  $("#srcCam").hidden = b.dataset.src !== "cam";
  $("#srcUpload").hidden = b.dataset.src !== "upload";
}));
async function startReg(overwrite = false) {
  try { await api("/api/register/start", { json: { roll: $("#rRoll").value, name: $("#rName").value, overwrite } }); }
  catch (e) {
    if (e.status === 409 && confirm(e.message + " Replace their photos?")) return startReg(true);
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
  try { const r = await api("/api/register/upload", { method: "POST", body: fd }); toast(`Imported ${r.saved} photos.`); }
  catch (e) { toast(e.message); }
  $("#btnUpload").disabled = false;
};

// ------------------------------------------------------------------ students
async function loadStudents() {
  const list = await api("/api/students");
  const ph = (r, i) => `/api/students/${encodeURIComponent(r)}/photo?i=${i}`;
  $("#studentCards").innerHTML = list.length ? list.map((s) => `
    <div class="flip" tabindex="0">
      <div class="front"><img src="${ph(s.roll, 0)}" alt="">
        <div class="body"><b>${esc(s.name)}</b><span>${esc(s.roll)} · ${s.images} photos${s.trained ? "" : " · not learned yet"}</span></div></div>
      <div class="back"><div class="thumbs">${[3, 7, 11, 15, 19, 23, 26, 28, 29].filter((i) => i < s.images).map((i) => `<img loading="lazy" src="${ph(s.roll, i)}" alt="">`).join("")}</div>
        <div class="row"><span>${esc(s.name)}</span><button data-del="${esc(s.roll)}">Remove</button></div></div>
    </div>`).join("") : `<div class="card empty">Nobody enrolled yet.</div>`;
  $$(".flip").forEach((c) => (c.onclick = (e) => { if (!e.target.dataset.del) c.classList.toggle("turned"); }));
  $$("[data-del]").forEach((b) => (b.onclick = async () => {
    if (!confirm(`Remove ${b.dataset.del} and their photos?`)) return;
    await api("/api/students/" + encodeURIComponent(b.dataset.del), { method: "DELETE" });
    loadStudents();
  }));
}
$("#btnTrain").onclick = () => api("/api/train", { json: {} }).then(() => toast("Learning faces…"));

// ------------------------------------------------------------------ records
async function loadRecords() {
  const recs = await api("/api/records");
  $("#recordList").innerHTML = recs.length ? recs.map((r, i) => `
    <details class="card reg" ${i === 0 ? "open" : ""}><summary><b>${esc(r.date)}</b>
      <span class="muted">${r.present} of ${r.total} present</span><span class="grow"></span>
      <a href="/api/records/${esc(r.date)}.csv"><button>Download CSV</button></a></summary>
      <table><thead><tr><th>Roll</th><th>Name</th><th>Status</th><th>Time</th></tr></thead><tbody>
      ${r.rows.map((x) => `<tr><td>${esc(x.roll)}</td><td>${esc(x.name)}</td><td><span class="tag ${esc(x.status)}">${esc(x.status)}</span></td><td>${esc(x.time)}</td></tr>`).join("")}
      </tbody></table></details>`).join("") : `<div class="card empty">No attendance taken yet.</div>`;
}

// ------------------------------------------------------------------ evidence
async function loadBench() {
  const b = await api("/api/benchmarks");
  const pct = (x) => Math.round(100 * x) + "%";
  const k = [];
  let det, lfw, v, y;
  if (b.detection) {
    det = b.detection.mosaic;
    k.push(["Wrong boxes per frame", det.consensus.false_per_frame.toFixed(2), `was ${det.haar.false_per_frame.toFixed(2)} with Haar alone`]);
  }
  if (b.lfw_all62_20) {
    lfw = b.lfw_all62_20; v = lfw.variants["uLBP(8,1)+(8,2) weighted"];
    k.push(["Right person picked (62 people)", pct(v.accuracy), `was ${pct(lfw.baseline_lbph.accuracy)} with OpenCV's LBPH`]);
    k.push(["Students missed, strangers kept out", pct(v["frr@far1 distance+ratio multi-frame"]), `was ${pct(lfw.baseline_lbph["frr@far1"].distance)}; lower is better`]);
  }
  if (b.yale) {
    y = b.yale.variants;
    k.push(["Recognised with light from the side", pct(y["LTP(8,1)+(8,2), Tan-Triggs"]["4"]), `was ${pct(y["LBP 256-bin, no normalisation (LBPH-like)"]["4"])}`]);
  }
  $("#benchKpis").innerHTML = k.map(([a, val, sub]) => `<div class="stat"><span>${a}</span><b>${val}</b><p>${sub}</p></div>`).join("");

  const figs = [
    ["pipeline_stages.png", "Every step on one photo", true,
      `<p>From left: the photo, grey and slightly blurred, contrast evened out, the two face finders (Haar blue, LBP orange), the cropped face, its texture codes, the same codes simplified to 59 kinds, and the count of each kind in one square of the face.</p>
       <p>Those counts, for all 64 squares at two sizes, are the face's "fingerprint".</p>`],
    ["detection_modes.png", "Two face finders voting", true, det ? `
       <div class="num">${det.haar.false_per_frame.toFixed(2)} → ${det.consensus.false_per_frame.toFixed(2)} wrong boxes per frame</div>
       <p>Haar adds up light and dark blocks; LBP compares neighbouring pixels. They are fooled by different things (curtains, shelves, patterned shirts), so when both agree it's almost always a real face.</p>
       <p>The cost: ${pct(det.haar.recall)} → ${pct(det.consensus.recall)} of faces found, which the 5-frame check makes up for.</p>` : ""],
    ["far_frr_all62_20.png", "Keeping strangers out", false, lfw ? `
       <div class="num">${pct(lfw.baseline_lbph["frr@far1"].distance)} → ${pct(v["frr@far1 distance+ratio multi-frame"])}</div>
       <p>Share of real students wrongly turned away, when at most 1 stranger in 100 is allowed in.</p>
       <p>The fix is the <b>runner-up test</b>: a real student is much closer to their own photos than to anyone else's, while a stranger is roughly equally far from everyone. Averaging over a couple of frames helps further.</p>` : ""],
    ["yale_recognition.png", "Light from the side", false, y ? `
       <div class="num">${pct(y["LBP 256-bin, no normalisation (LBPH-like)"]["4"])} → ${pct(y["LTP(8,1)+(8,2), Tan-Triggs"]["4"])}</div>
       <p>Faces lit from 50–77° to the side, matched against photos lit from the front.</p>
       <p>The "Remove shadows" setting (Tan–Triggs) brightens dark areas, filters out the slow shadow gradient and keeps the fine detail. LTP ignores tiny brightness differences that are just noise.</p>` : ""],
    ["cell_weights.png", "Which parts of the face matter", false, `
       <p>We measured, square by square, how well each part of the face tells people apart, using 96 people who are never used for testing.</p>
       <p>It found by itself what researchers set by hand in 2004: eyes and eyebrows matter about 3× more than average; corners (hair, background) hardly at all.</p>`],
    ["detection_lighting.png", "Finding faces as the lamp moves", false, `
       <p>Share of faces found as the light moves further to the side.</p>
       <p>Past about 50° half the face is in shadow and neither finder copes. That's the honest limit of a classical frontal detector.</p>`],
  ];
  $("#figs").innerHTML = figs.map(([src, title, span, back]) => `
    <div class="fig${span ? " span" : ""}" tabindex="0">
      <div class="front"><h3>${title}<small>click to flip</small></h3><img src="/figures/${src}" alt="${esc(title)}"></div>
      <div class="back"><h3>${title}</h3>${back}</div>
    </div>`).join("");
  $$(".fig").forEach((f) => (f.onclick = () => f.classList.toggle("turned")));
}

// ------------------------------------------------------------------ settings
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
  toast("Saved.");
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
  loadSettings(); toast("Applied.");
};
$("#camOn").onchange = (e) => api("/api/camera", { json: { on: e.target.checked } });

let first = "live";
try { first = localStorage.getItem("tab") || "live"; } catch {}
$("#tab-live").classList.remove("on");
show(TABS.includes(first) ? first : "live");
poll();
