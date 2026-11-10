// FaceRoll in the browser: camera + UI on this thread, the Python pipeline in worker.js.
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const TABS = ["live", "register", "students", "records", "settings"];
const DEFAULTS = { DETECTOR: "consensus", DIST_THRESHOLD: 0.5, RATIO_THRESHOLD: 0.84, CONFIRM_FRAMES: 5, LIVENESS: false };
const COLORS = { marked: "#2f8a3c", verifying: "#e0a000", blink: "#2a66c9", unknown: "#d0342c", face: "#eeeeee" };
const ls = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
const today = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; };
const clock = () => new Date().toTimeString().slice(0, 8);
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove("show"), 3800); }

// ------------------------------------------------------------------ storage (IndexedDB for photos, localStorage for the rest)
const db = new Promise((res, rej) => {
  const r = indexedDB.open("faceroll", 1);
  r.onupgradeneeded = () => r.result.createObjectStore("students", { keyPath: "id" });
  r.onsuccess = () => res(r.result);
  r.onerror = () => rej(r.error);
});
async function store(mode, fn) {
  const d = await db;
  return new Promise((res, rej) => {
    const tx = d.transaction("students", mode), req = fn(tx.objectStore("students"));
    tx.oncomplete = () => res(req && req.result);
    tx.onerror = () => rej(tx.error);
  });
}
const allStudents = () => store("readonly", (s) => s.getAll());
let students = [];                       // [{id, roll, name, chips, thumb(dataURL)}]
let settings = { ...DEFAULTS, ...ls.get("faceroll-settings", {}) };
let records = ls.get("faceroll-records", {});        // {date: {roll: {name, time}}}

// ------------------------------------------------------------------ worker
const worker = new Worker("worker.js");
let ready = false, busy = false, last = { faces: [] }, mode = "preview", view = "camera";
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "progress") { $("#bootText").textContent = m.text; $("#bootBar").style.width = m.pct + "%"; }
  else if (m.type === "ready") boot2();
  else if (m.type === "loaded") { ready = true; $("#boot").hidden = true; $("#btnStart").disabled = !students.length; $("#btnReg").disabled = false; render(); }
  else if (m.type === "result") onResult(m.res);
  else if (m.type === "error") { busy = false; console.error(m.text); toast("Processing error: " + m.text.slice(0, 120)); }
};
worker.postMessage({ type: "boot" });

async function boot2() {
  $("#bootText").textContent = "Loading your students…"; $("#bootBar").style.width = "95%";
  students = await allStudents();
  worker.postMessage({ type: "config", values: settings });
  worker.postMessage({ type: "load", students: students.map(({ id, roll, name, chips }) => ({ id, roll, name, chips })), present: presentToday() });
}
const presentToday = () => Object.fromEntries(Object.entries(records[today()] || {}).map(([r, v]) => [r, v.time]));

// ------------------------------------------------------------------ camera
const video = document.createElement("video");
video.muted = true; video.playsInline = true;
const grab = document.createElement("canvas"); grab.width = 640; grab.height = 480;
const gctx = grab.getContext("2d", { willReadFrequently: true });
const screen = $("#screen"), sctx = screen.getContext("2d");
let camOn = false, viewImg = null, frames = 0, fpsT = performance.now(), msAvg = 0;

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false });
    video.srcObject = stream; await video.play();
    camOn = true; $("#camOff").hidden = true;
  } catch (err) {
    $("#camErr").textContent = err.name === "NotAllowedError" ? "Camera permission was refused. Allow it in the browser's address bar, then press the button again."
      : "No camera found, or it is in use by another app. (" + err.name + ")";
  }
}
$("#btnCam").onclick = startCamera;
if (navigator.permissions) navigator.permissions.query({ name: "camera" }).then((p) => { if (p.state === "granted") startCamera(); }).catch(() => {});

function loop() {
  if (camOn && video.readyState >= 2) {
    // mirror like a selfie camera; process exactly what is shown
    const vw = video.videoWidth, vh = video.videoHeight, s = Math.max(640 / vw, 480 / vh);
    gctx.save(); gctx.translate(640, 0); gctx.scale(-1, 1);
    gctx.drawImage(video, (640 - vw * s) / 2, (480 - vh * s) / 2, vw * s, vh * s); gctx.restore();
    if (ready && !busy) {
      busy = true;
      const img = gctx.getImageData(0, 0, 640, 480);
      worker.postMessage({ type: "frame", rgba: img.data, w: 640, h: 480, now: clock() }, [img.data.buffer]);
    }
    const out = view === "camera" || !viewImg ? grab : viewImg;
    sctx.drawImage(out, 0, 0, 640, 480);
    if (view === "camera") drawBoxes(sctx, last.faces);
    if (mode === "enrol") { const r = $("#regScreen").getContext("2d"); r.drawImage(grab, 0, 0); drawBoxes(r, last.faces); }
  }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

function drawBoxes(ctx, faces) {
  ctx.lineWidth = 3; ctx.font = "600 17px system-ui, sans-serif";
  for (const f of faces || []) {
    const [x, y, w, h] = f.box, c = COLORS[f.state] || "#eee";
    ctx.strokeStyle = c; ctx.strokeRect(x, y, w, h);
    const label = f.label || ({ marked: `${f.name} - present`, verifying: `${f.name}?  ${Math.min(f.streak, settings.CONFIRM_FRAMES)}/${settings.CONFIRM_FRAMES}`,
      blink: `${f.name} - please blink`, unknown: "not recognised", face: "face found" })[f.state];
    const tw = ctx.measureText(label).width + 14, ty = y > 30 ? y - 28 : y + h;
    ctx.fillStyle = c; ctx.fillRect(x - 1.5, ty, tw, 26);
    ctx.fillStyle = f.state === "face" ? "#111" : "#fff"; ctx.fillText(label, x + 6, ty + 19);
  }
}

function onResult(res) {
  busy = false;
  last = res;
  frames++;
  msAvg = msAvg ? msAvg * 0.9 + res.ms * 0.1 : res.ms;
  const t = performance.now();
  if (t - fpsT > 1000) { $("#fps").textContent = `${(frames * 1000 / (t - fpsT)).toFixed(1)} frames/s analysed · ${msAvg.toFixed(0)} ms each`; frames = 0; fpsT = t; }
  if (res.view) {
    const c = viewImg || Object.assign(document.createElement("canvas"), { width: 320, height: 240 });
    c.getContext("2d").putImageData(new ImageData(res.view, 320, 240), 0, 0); viewImg = c;
  }
  if (res.explain) {
    const all = new ImageData(res.explain, 480, 120), tmp = document.createElement("canvas");
    tmp.width = 480; tmp.height = 120; tmp.getContext("2d").putImageData(all, 0, 0);
    $$(".layer canvas").forEach((cv, i) => cv.getContext("2d").drawImage(tmp, i * 120, 0, 120, 120, 0, 0, 120, 120));
  }
  if (res.present && mode === "attendance") {
    const day = (records[today()] ||= {});
    let changed = false;
    for (const [roll, time] of Object.entries(res.present)) {
      if (!day[roll]) { const s = students.find((x) => x.roll === roll); day[roll] = { name: s ? s.name : roll, time }; changed = true; }
    }
    if (changed) ls.set("faceroll-records", records);
  }
  if (res.enrol) {
    $("#ringTxt").textContent = `${res.enrol.saved}/${res.enrol.target}`;
    $("#ringProg").style.strokeDashoffset = 97.4 * (1 - res.enrol.saved / res.enrol.target);
    $("#regHint").textContent = res.enrol.hint;
  }
  if ("enrolled" in res) finishEnrol(res.enrolled, res.cancelled);
  render();
}

// ------------------------------------------------------------------ narration
function narrate(faces) {
  if (!camOn) return ["", "The camera is off.", "Press Turn on camera and allow access when your browser asks."];
  if (mode === "enrol") return ["wait", "Enrolling…", "Follow the instructions on the Enrol tab."];
  if (!faces.length) return ["", "I don't see anyone.", "Face the camera from about an arm's length away, with light on your face."];
  const f = faces.reduce((a, b) => (a.box[2] > b.box[2] ? a : b)), st = settings;
  const more = faces.length > 1 ? `I can see ${faces.length} faces; this is about the biggest one. ` : "";
  const rec = mode === "attendance";
  if (f.state === "face") return ["wait", "I can see a face, but nobody is enrolled yet.", more + "Go to Enrol and add yourself; then I'll know who you are."];
  if (f.state === "unknown") {
    if (f.dist > st.DIST_THRESHOLD) return ["no", "I can see a face, but I don't recognise it.", more + `The closest enrolled student is ${f.nearest}, but not close enough: a difference of ${f.dist.toFixed(2)} against a limit of ${st.DIST_THRESHOLD}.`];
    if (f.runner_up === "crowd") return ["no", "I can see a face, but I don't recognise it.", more + `It's no closer to ${f.nearest} than to faces of people who aren't in this class (ratio ${f.ratio.toFixed(2)}, limit ${st.RATIO_THRESHOLD}), so I won't guess.`];
    return ["no", "I can see a face, but I'm not sure who it is.", more + `It's close to ${f.nearest}, but almost as close to ${f.runner_up} (ratio ${f.ratio.toFixed(2)}, limit ${st.RATIO_THRESHOLD}), so I won't guess.`];
  }
  if (f.state === "verifying") return ["wait", `Recognising ${f.name}…`, more + `Matched in ${Math.min(f.streak, st.CONFIRM_FRAMES)} of the ${st.CONFIRM_FRAMES} frames needed.` + (rec ? "" : " Press Start attendance to record it.")];
  if (f.state === "blink") return ["wait", `${f.name}, please blink.`, more + "This checks that it's a real person and not a photo."];
  if (f.state === "marked") return ["ok", `${f.name} is marked present.`, more + `Recorded at ${(records[today()] || {})[f.roll]?.time || ""}.`];
  return ["", "", ""];
}

function meter(title, v, thr, max, note) {
  const ok = v <= thr;
  return `<div class="meter">${title}<div class="bar"><em style="width:${Math.min(100, (v / max) * 100)}%;background:var(${ok ? "--green" : "--red"})"></em><i style="left:${Math.min(100, (thr / max) * 100)}%"></i></div>
    <div class="val"><span>${note}</span><span>${v.toFixed(2)} / limit ${thr}</span></div></div>`;
}

function render() {
  const rec = mode === "attendance", day = records[today()] || {};
  $("#sessionText").textContent = rec ? "Recording attendance" : "Not recording";
  $("#sessionText").className = rec ? "rec" : "";
  $("#btnStart").hidden = rec; $("#btnStop").hidden = !rec;
  const [cls, line, sub] = narrate(last.faces || []);
  $("#say").className = "say " + cls; $("#say").innerHTML = `${esc(line)}<small>${esc(sub)}</small>`;

  const roster = students.map((s) => ({ ...s, status: day[s.roll] ? "Present" : "Absent", time: day[s.roll]?.time || "" }))
    .sort((a, b) => (a.status !== b.status ? (a.status === "Present" ? -1 : 1) : a.roll.localeCompare(b.roll)));
  $("#kPresent").textContent = roster.filter((r) => r.status === "Present").length;
  $("#kTotal").textContent = roster.length;
  const names = ls.get("faceroll-names", false);
  $("#showNames").checked = names;
  $("#roster").hidden = !names; $("#feedCard").hidden = !names; $("#privNote").hidden = names;
  const key = JSON.stringify(roster.map((r) => [r.roll, r.status, r.time]));
  if (key !== render.key) {
    render.key = key;
    $("#roster").innerHTML = roster.length ? roster.map((r) => `<li class="${r.status}"><img class="avatar" src="${r.thumb}" alt="">
      <div><b>${esc(r.name)}</b><span class="r">${esc(r.roll)}${r.time ? " · " + esc(r.time) : ""}</span></div><span class="tag ${r.status}">${r.status}</span></li>`).join("")
      : `<li class="empty-note">Nobody enrolled yet.</li>`;
  }
  const ev = last.events || [];
  if (ev[0] && ev[0].time !== render.welcome) {
    render.welcome = ev[0].time;
    if (!names) { $("#welcome").textContent = `Welcome, ${ev[0].name}: marked present.`; $("#welcome").hidden = false; clearTimeout(render.wt); render.wt = setTimeout(() => ($("#welcome").hidden = true), 8000); }
  }
  $("#feed").innerHTML = ev.length ? ev.map((e) => `<li><time>${esc(e.time)}</time>${esc(e.name)}</li>`).join("") : `<li class="empty-note">${rec ? "Waiting for the first person…" : "Press Start attendance to begin."}</li>`;

  const faces = (last.faces || []).filter((f) => f.dist !== undefined);
  $("#faceTable").innerHTML = faces.length ? faces.map((f) => `<div>
      <div class="fr-head"><b>${esc(f.name || "Not recognised")}</b><span>closest: ${esc(f.nearest || "–")}</span></div>
      ${meter("How different from the closest student", f.dist, settings.DIST_THRESHOLD, Math.max(0.8, f.dist), "lower = more alike")}
      ${f.ratio > 0 ? meter(`Closest compared with the next face (${f.runner_up === "crowd" ? "someone outside the class" : esc(f.runner_up)})`, f.ratio, settings.RATIO_THRESHOLD, Math.max(1, f.ratio), "lower = more certain") : ""}
    </div>`).join("") : `<p class="muted">${students.length ? "When a face is in view, its numbers appear here." : "Enrol someone to see how matching works."}</p>`;
}

// ------------------------------------------------------------------ tabs (3D cube turn)
let tab = null;
function show(name) {
  if (name === tab) return;
  const prev = tab ? $("#tab-" + tab) : null, next = $("#tab-" + name), fwd = !tab || TABS.indexOf(name) > TABS.indexOf(tab);
  tab = name;
  $$(".tabs a").forEach((a) => a.classList.toggle("on", a.dataset.tab === name));
  $$(".face").forEach((f) => f.classList.remove("in-r", "in-l", "out-r", "out-l"));
  if (prev) { prev.classList.add(fwd ? "out-r" : "out-l"); const done = () => prev.classList.remove("on", "out-r", "out-l"); prev.addEventListener("animationend", done, { once: true }); setTimeout(done, 800); }
  next.classList.add("on", fwd ? "in-r" : "in-l");
  next.addEventListener("animationend", () => next.classList.remove("in-r", "in-l"), { once: true });
  if (name === "students") loadStudents();
  if (name === "records") loadRecords();
  if (name === "settings") loadSettings();
}
$$(".tabs a").forEach((a) => (a.onclick = () => show(a.dataset.tab)));
$("#tab-live").classList.remove("on");
show("live");

// ------------------------------------------------------------------ live controls
const VIEW_NOTES = {
  camera: "The normal camera picture, with the system's decision drawn on top.",
  contrast: "Slightly blurred to remove grain, turned grey, then contrast evened out tile by tile (CLAHE). This is the picture the face finders search.",
  texture: "Every pixel replaced by its LBP code: 8 \"is my neighbour brighter?\" answers. Switch a lamp on or off: the photo changes a lot, these codes barely do.",
  votes: "What each face finder says on its own, before they vote. Haar (blue) looks for light-and-dark blocks; LBP (orange) looks for texture. A face is kept when they agree.",
};
function setView(v) { view = v; viewImg = null; $$("#viewSeg button").forEach((b) => b.classList.toggle("on", b.dataset.view === v)); $("#viewNote").textContent = VIEW_NOTES[v]; worker.postMessage({ type: "view", view: v }); }
$$("#viewSeg button").forEach((b) => (b.onclick = () => setView(b.dataset.view)));
setView("camera");
$("#showNames").onchange = (e) => { ls.set("faceroll-names", e.target.checked); render.key = null; render(); };
$$("#steps li").forEach((li) => { const l = $$(".layer")[+li.dataset.l]; li.onmouseenter = () => l.classList.add("lift"); li.onmouseleave = () => l.classList.remove("lift"); });
$("#btnFlat").onclick = () => { const f = $("#stack").classList.toggle("flat"); $("#btnFlat").textContent = f ? "Stack the layers in 3D" : "Lay the layers flat"; };
$("#btnStart").onclick = () => {
  if (!camOn) return toast("Turn on the camera first.");
  mode = "attendance"; worker.postMessage({ type: "present", present: presentToday() }); worker.postMessage({ type: "mode", mode });
  records[today()] ||= {}; ls.set("faceroll-records", records); render();
};
$("#btnStop").onclick = () => {
  mode = "preview"; worker.postMessage({ type: "mode", mode });
  const day = records[today()] || {};
  toast(`Saved: ${Object.keys(day).length} of ${students.length} present. See Records.`); render();
};

// ------------------------------------------------------------------ enrolment
$("#btnReg").onclick = () => {
  const roll = $("#rRoll").value.trim(), name = $("#rName").value.trim().replace(/\s+/g, " ");
  if (!/^[A-Za-z0-9-]+$/.test(roll)) return toast("Roll number may only contain letters, digits and '-'.");
  if (!/^[A-Za-z .'-]+$/.test(name)) return toast("Name may only contain letters, spaces, '.', ''' and '-'.");
  if (!camOn) return toast("Turn on the camera first (Live tab).");
  const existing = students.find((s) => s.roll === roll);
  if (existing && !confirm(`Roll ${roll} is already enrolled as ${existing.name}. Replace their photos?`)) return;
  const id = existing ? existing.id : Math.max(0, ...students.map((s) => s.id)) + 1;
  mode = "enrol"; worker.postMessage({ type: "enrol", id, roll, name });
  $("#btnReg").disabled = true; $("#btnRegCancel").hidden = false;
};
$("#btnRegCancel").onclick = () => worker.postMessage({ type: "cancelEnrol" });

async function finishEnrol(e, cancelled) {
  mode = "preview"; $("#btnReg").disabled = false; $("#btnRegCancel").hidden = true;
  if (!e) { $("#regHint").textContent = cancelled ? "Stopped. Fewer than 10 good photos, so nothing was saved." : "Not enough good photos. Try again with more light."; return; }
  const c = document.createElement("canvas"); c.width = c.height = 96;
  c.getContext("2d").putImageData(new ImageData(e.thumb, 96, 96), 0, 0);
  const rec = { id: e.id, roll: e.roll, name: e.name, chips: e.chips, thumb: c.toDataURL("image/jpeg", 0.85) };
  await store("readwrite", (s) => s.put(rec));
  students = await allStudents();
  $("#ringProg").style.strokeDashoffset = 0;
  $("#regHint").textContent = `Done: ${e.name} is enrolled. Enrol the next person, or go to Live.`;
  $("#btnStart").disabled = false;
  render.key = null; render();
}

// ------------------------------------------------------------------ students
async function loadStudents() {
  $("#studentCards").innerHTML = students.length ? students.map((s) => `
    <div class="card scard"><img src="${s.thumb}" alt="" style="width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px">
      <div class="body" style="padding-top:10px"><b>${esc(s.name)}</b><br><span class="muted small">${esc(s.roll)} · ${s.chips.length / 40000} photos</span>
      <div style="margin-top:8px"><button data-del="${s.id}">Remove</button></div></div></div>`).join("")
    : `<div class="card empty">Nobody enrolled yet. Use the Enrol tab.</div>`;
  $$("[data-del]").forEach((b) => (b.onclick = async () => {
    const s = students.find((x) => x.id === +b.dataset.del);
    if (!confirm(`Remove ${s.name} and their photos from this browser?`)) return;
    await store("readwrite", (st) => st.delete(s.id));
    worker.postMessage({ type: "remove", id: s.id });
    students = await allStudents(); render.key = null; loadStudents(); render();
  }));
}

// ------------------------------------------------------------------ records
function csvFor(date) {
  const day = records[date] || {}, rows = new Map(students.map((s) => [s.roll, s.name]));
  for (const [r, v] of Object.entries(day)) if (!rows.has(r)) rows.set(r, v.name);
  const q = (v) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
  return "roll,name,status,time\n" + [...rows].sort().map(([r, n]) => [r, n, day[r] ? "Present" : "Absent", day[r]?.time || ""].map(q).join(",")).join("\n") + "\n";
}
function loadRecords() {
  const dates = Object.keys(records).sort().reverse();
  $("#recordList").innerHTML = dates.length ? dates.map((d, i) => {
    const day = records[d], rows = csvFor(d).trim().split("\n").slice(1).map((l) => l.split(","));
    return `<details class="card reg" ${i === 0 ? "open" : ""}><summary><b>${esc(d)}</b><span class="muted">${Object.keys(day).length} present</span><span class="grow"></span>
      <button data-csv="${esc(d)}">Download CSV</button></summary>
      <table><thead><tr><th>Roll</th><th>Name</th><th>Status</th><th>Time</th></tr></thead><tbody>
      ${rows.map((x) => `<tr><td>${esc(x[0])}</td><td>${esc(x[1])}</td><td><span class="tag ${esc(x[2])}">${esc(x[2])}</span></td><td>${esc(x[3])}</td></tr>`).join("")}</tbody></table></details>`;
  }).join("") : `<div class="card empty">No attendance taken yet.</div>`;
  $$("[data-csv]").forEach((b) => (b.onclick = (ev) => {
    ev.preventDefault();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csvFor(b.dataset.csv)], { type: "text/csv" }));
    a.download = `attendance_${b.dataset.csv}.csv`; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }));
}

// ------------------------------------------------------------------ settings
const KEYS = Object.keys(DEFAULTS);
function loadSettings() {
  for (const k of KEYS) {
    const el = $("#s" + k); if (el.type === "checkbox") el.checked = settings[k]; else el.value = settings[k];
    const o = $("#o" + k); if (o) { o.textContent = el.value; el.oninput = () => (o.textContent = el.value); }
  }
}
function applySettings() { ls.set("faceroll-settings", settings); worker.postMessage({ type: "config", values: settings }); }
$("#btnSave").onclick = () => {
  for (const k of KEYS) { const el = $("#s" + k); settings[k] = el.type === "checkbox" ? el.checked : el.type === "range" ? Number(el.value) : el.value; }
  applySettings(); toast("Saved.");
};
$("#btnReset").onclick = () => { settings = { ...DEFAULTS }; applySettings(); loadSettings(); toast("Back to defaults."); };
$("#btnWipe").onclick = async () => {
  if (!confirm("Erase all students, photos and records stored by FaceRoll in this browser?")) return;
  for (const s of students) { await store("readwrite", (st) => st.delete(s.id)); worker.postMessage({ type: "remove", id: s.id }); }
  students = []; records = {}; ls.set("faceroll-records", records); render.key = null; render(); toast("Everything erased.");
};
