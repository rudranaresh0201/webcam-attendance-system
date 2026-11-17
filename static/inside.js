// "Inside" tab: the pipeline's own numbers for the biggest face in view, polled from /api/inside.
//   1. the face at every step (one image from /inside.jpg, sliced into tiles)
//   2. the Haar cascade re-run on the face window, stepping through each rectangle test
//   3. the χ² distance race against every student and the background crowd
//   4. the LBP descriptor as a 3D field of histograms (three.js, served from static/vendor)
(() => {
  const $ = (s) => document.querySelector(s), $$ = (s) => [...document.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const active = () => document.querySelector("#tab-inside")?.classList.contains("on");
  let data = null, polling = false, weights = null;

  // ---------------------------------------------------------------- 1. tiles
  const NAMES = ["Camera", "Grey + blur", "CLAHE", "Face chip 200×200", "LBP codes", "Uniform LBP (59 bins)"];
  $("#inTiles").innerHTML = NAMES.map((n, i) => `<div class="tile"><div class="img" style="background-position:${i * 20}% 0"></div><span><b>${i + 1}</b>${n}</span></div>`).join("");
  const tileImg = new Image();
  function loadTiles() {
    return new Promise((res) => {
      const url = "/inside.jpg?t=" + Date.now(), im = new Image();
      im.onload = () => { $$("#inTiles .img").forEach((d) => (d.style.backgroundImage = `url(${url})`)); tileImg.src = url; res(im); };
      im.onerror = () => res(null);
      im.src = url;
    });
  }

  // ---------------------------------------------------------------- 2. Haar stepper
  const cv = $("#haarCv"), ctx = cv.getContext("2d");
  let hStage = 0, hTest = 0, playing = !reduce;
  $$("#haarStageSeg button").forEach((b) => (b.onclick = () => {
    hStage = +b.dataset.s; hTest = 0;
    $$("#haarStageSeg button").forEach((x) => x.classList.toggle("on", x === b)); drawHaar();
  }));
  $("#haarPlay").onclick = () => { playing = !playing; $("#haarPlay").textContent = playing ? "Pause" : "Play"; };
  cv.onclick = () => { if (!data) return; hTest++; drawHaar(); };
  setInterval(() => { if (playing && active() && data) { hTest++; drawHaar(); } }, 900);

  function drawHaar() {
    if (!data) return;
    const h = data.haar, win = data.window, N = win.length, S = cv.width / N;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const v = win[y][x]; ctx.fillStyle = `rgb(${v},${v},${v})`; ctx.fillRect(x * S, y * S, S + 0.5, S + 0.5); }
    // stage bar
    $("#stageBar").innerHTML = Array.from({ length: h.total }, (_, i) => {
      const st = h.stages[i];
      return `<i class="${st ? (st.pass ? "ok" : "no") : ""}${i === hStage ? " sel" : ""}" title="stage ${i + 1}${st ? `: sum ${st.sum} vs ${st.threshold}` : ": not reached"}"></i>`;
    }).join("");
    const face = h.passed === h.total;
    const votes = h.votes ? ` This is one of <b>${h.votes}</b> overlapping windows OpenCV accepted around your face; that count is the vote checked by minNeighbors.` : "";
    $("#haarVerdict").innerHTML = (face ? `<span style="color:var(--green)">Passed all ${h.total} stages</span> · ${h.features_evaluated.toLocaleString()} tests → face.`
      : `<span style="color:var(--red)">This window stopped at stage ${h.passed + 1}</span> after ${h.features_evaluated} tests.`) + `<span class="muted small" style="font-weight:400">${votes}</span>`;
    const st = h.stages[hStage];
    if (!st || !st.detail.length) { $("#haarTest").innerHTML = `<span class="muted">Stage ${hStage + 1} wasn't reached: the cascade stopped earlier.</span>`; $("#haarSum").innerHTML = ""; return; }
    const k = hTest % st.detail.length, t = st.detail[k];
    // overlay: the rectangles of this test; negative weight darkens, positive weight brightens
    for (const [x, y, w, hh, wt] of t.rects) {
      ctx.fillStyle = wt < 0 ? "rgba(0,0,0,.55)" : "rgba(255,255,255,.6)";
      ctx.fillRect(x * S, y * S, w * S, hh * S);
      ctx.strokeStyle = wt < 0 ? "#2a66c9" : "#c27c00"; ctx.lineWidth = 2; ctx.strokeRect(x * S + 1, y * S + 1, w * S - 2, hh * S - 2);
    }
    const below = t.value < t.threshold;
    $("#haarTest").innerHTML = `Stage ${st.stage}, test <b>${k + 1}</b> of ${st.tests} (click the face for the next one)<br>
      feature value <b>${t.value.toFixed(3)}</b> ${below ? "&lt;" : "≥"} threshold <b>${t.threshold.toFixed(3)}</b>
      → vote <b style="color:var(${t.face_like ? "--green" : "--red"})">${t.vote > 0 ? "+" : ""}${t.vote.toFixed(2)}</b> ${t.face_like ? "(face-like)" : "(not face-like)"}`;
    const partial = st.detail.slice(0, k + 1).reduce((a, x) => a + x.vote, 0);
    const lo = Math.min(st.threshold, partial, 0) - 0.5, hi = Math.max(st.sum, partial, st.threshold) + 0.5, pos = (v) => ((v - lo) / (hi - lo)) * 100;
    $("#haarSum").innerHTML = `<div class="meter">Running sum of votes after ${k + 1} tests: <b>${partial.toFixed(2)}</b> (stage needs ≥ ${st.threshold.toFixed(2)}; full stage: ${st.sum.toFixed(2)} → ${st.pass ? "pass" : "reject"})
      <div class="bar"><em style="width:${Math.max(0, Math.min(100, pos(partial)))}%;background:var(${partial >= st.threshold ? "--green" : "--amber"})"></em><i style="left:${pos(st.threshold)}%"></i></div></div>`;
  }

  // ---------------------------------------------------------------- 3. race
  function drawRace() {
    const r = data.race, thr = data.thresholds;
    if (!r.length) { $("#race").innerHTML = `<p class="muted">Enrol students to see the race.</p>`; $("#raceVerdict").innerHTML = ""; return; }
    const rows = r.slice(0, 7).map((x, i) => ({ ...x, first: i === 0 }));
    if (data.crowd !== null) rows.push({ name: "Crowd (closest of 192)", d: data.crowd, crowd: true });
    const max = Math.max(0.8, ...rows.map((x) => x.d));
    const d1 = r[0].d, d2c = [r[1]?.d, data.crowd].filter((v) => v != null), d2 = d2c.length ? Math.min(...d2c) : null;
    const ratio = d2 ? d1 / d2 : 0, ok = d1 <= thr.dist && ratio <= thr.ratio;
    $("#race").innerHTML = rows.map((x) => `<div class="row ${x.first ? "first" : ""} ${x.first && ok ? "ok" : ""} ${x.crowd ? "crowd" : ""}">
      <span class="nm" title="${esc(x.name)}">${esc(x.name)}</span><span class="tr"><em style="width:${(x.d / max) * 100}%"></em>${x.first ? `<i style="left:${(thr.dist / max) * 100}%" title="limit ${thr.dist}"></i>` : ""}</span><span class="v">${x.d.toFixed(3)}</span></div>`).join("");
    const runner = d2 === data.crowd ? "a stranger from the crowd" : esc(r[1]?.name);
    $("#raceVerdict").innerHTML = `Closest: <b>${esc(r[0].name)}</b> at ${d1.toFixed(3)} (limit ${thr.dist}). ` +
      (d2 ? `Runner-up: ${runner} at ${d2.toFixed(3)}, ratio <b>${ratio.toFixed(2)}</b> (limit ${thr.ratio}). ` : "") +
      `This frame: <b class="${ok ? "ok" : "no"}">${ok ? "accepted as " + esc(r[0].name) : "not sure, counts as Unknown"}</b>. Five of eight such frames mark a student present.`;
  }

  // ---------------------------------------------------------------- 4. 3D fingerprint
  let fp = null, fpTried = false;
  function initFp() {
    const el = $("#fp3d");
    let gl = false;
    try { const c = document.createElement("canvas"); gl = !!(window.THREE && (c.getContext("webgl") || c.getContext("experimental-webgl"))); } catch (e) {}
    if (!gl) { $("#fpHint").textContent = "The 3D view needs WebGL, which this browser has turned off."; return null; }
    const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(34, 2, 0.1, 100);
    const r = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    r.setPixelRatio(Math.min(devicePixelRatio, 2)); el.prepend(r.domElement);
    const fit = () => { r.setSize(el.clientWidth, el.clientHeight, false); cam.aspect = el.clientWidth / el.clientHeight; cam.updateProjectionMatrix(); };
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dl = new THREE.DirectionalLight(0xffffff, 0.6); dl.position.set(3, 8, 5); scene.add(dl);
    const group = new THREE.Group(); scene.add(group);
    const G = 8, B = 59, SUB = 8, cell = 1, slot = cell / SUB;
    const texCanvas = document.createElement("canvas"); texCanvas.width = texCanvas.height = 160;
    const tex = new THREE.CanvasTexture(texCanvas);
    const base = new THREE.Mesh(new THREE.PlaneGeometry(G, G), new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.9 }));
    base.rotation.x = -Math.PI / 2; group.add(base);
    const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(slot * 0.6, 1, slot * 0.6), new THREE.MeshLambertMaterial(), G * G * B);
    group.add(mesh);
    const m4 = new THREE.Matrix4(), col = new THREE.Color(), lo = new THREE.Color("#2a66c9"), hi = new THREE.Color("#c27c00");
    const cur = new Float32Array(G * G * B), target = new Float32Array(G * G * B);
    let mode = "hist", rotY = -0.5, tilt = 0.95, drag = null, idle = 0, run = true;
    cam.position.set(0, 0, 13);
    $$("#fpMode button").forEach((b) => (b.onclick = () => { mode = b.dataset.m; $$("#fpMode button").forEach((x) => x.classList.toggle("on", x === b)); setTarget(); }));
    r.domElement.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY, rotY, tilt]; r.domElement.setPointerCapture(e.pointerId); });
    r.domElement.addEventListener("pointermove", (e) => { if (!drag) return; rotY = drag[2] + (e.clientX - drag[0]) * 0.01; tilt = Math.max(0.2, Math.min(1.45, drag[3] + (e.clientY - drag[1]) * 0.006)); idle = 0; });
    r.domElement.addEventListener("pointerup", () => (drag = null));
    addEventListener("resize", fit); fit();
    new IntersectionObserver((es) => es.forEach((e) => { run = e.isIntersecting; if (run) loop(); })).observe(el);
    function setTarget() {
      if (!data) return;
      const wmean = weights ? weights.slice(0, G * G).reduce((a, b) => a + b, 0) / (G * G) : 1;
      for (let c = 0; c < G * G; c++) {
        const w = mode === "w" && weights ? weights[c] / wmean : 1;
        for (let b = 0; b < B; b++) target[c * B + b] = (data.hist[c][b] || 0) * w;
      }
    }
    function updateTexture() {
      if (!tileImg.complete || !tileImg.naturalWidth) return;
      const c = texCanvas.getContext("2d"), t = tileImg.naturalHeight;    // tile 4 (index 3) is the chip
      c.drawImage(tileImg, 3 * t, 0, t, t, 0, 0, 160, 160); tex.needsUpdate = true;
    }
    function loop() {
      if (!run) return;
      if (!drag && !reduce) { idle++; if (idle > 120) rotY += 0.0025; }
      group.rotation.set(0, 0, 0); group.rotateX(tilt); group.rotateY(rotY);
      for (let c = 0; c < G * G; c++) {
        const cx = c % G, cy = Math.floor(c / G);
        for (let b = 0; b < B; b++) {
          const i = c * B + b; cur[i] += (target[i] - cur[i]) * 0.15;
          const hgt = Math.max(0.005, Math.min(cur[i] * 9, 4));
          const sx = b % SUB, sy = Math.floor(b / SUB);
          m4.makeScale(1, hgt, 1).setPosition(-G / 2 + cx * cell + (sx + 0.5) * slot, hgt / 2, -G / 2 + cy * cell + (sy + 0.5) * slot);
          mesh.setMatrixAt(i, m4);
          col.copy(lo).lerp(hi, Math.min(1, hgt / 1.6)); mesh.setColorAt(i, col);
        }
      }
      mesh.instanceMatrix.needsUpdate = true; if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      r.render(scene, cam); requestAnimationFrame(loop);
    }
    // tilt the whole field towards the camera: x-rotation is applied to the group, so the grid faces the viewer
    group.position.y = -0.6;
    loop();
    return { setTarget, updateTexture };
  }

  // ---------------------------------------------------------------- polling
  async function tick() {
    if (!active()) { polling = false; return; }
    try {
      const r = await fetch("/api/inside" + (weights ? "" : "?weights=1"));
      const d = await r.json();
      if (d.face) {
        data = d; if (d.weights) weights = d.weights;
        await loadTiles();
        $("#inStatus").textContent = `Biggest face in view · detector: ${d.sources} · box ${d.box[2]}×${d.box[3]} px`;
        $("#fpHint").textContent = "drag to turn";
        drawHaar(); drawRace();
        if (!fpTried) { fpTried = true; fp = initFp(); }
        fp?.setTarget(); fp?.updateTexture();
      } else {
        $("#inStatus").textContent = "No face in view. Look at the camera.";
      }
    } catch (e) { /* server busy or restarting */ }
    setTimeout(tick, 450);
  }
  window.startInside = () => { if (!polling) { polling = true; tick(); } };
  if (active()) window.startInside();
})();
