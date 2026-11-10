// Runs FaceRoll's Python pipeline (Pyodide + OpenCV) off the main thread.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js");

const FILES = [
  "py/config.py", "py/lbp.py", "py/detector.py", "py/recognizer.py", "py/tracker.py", "py/engines.py", "py/utils.py", "py/webcore.py",
  "py/cascades/haarcascade_frontalface_default.xml", "py/cascades/haarcascade_eye.xml", "py/cascades/lbpcascade_frontalface_improved.xml",
  "py/assets/cell_weights_ulbp_8.1-8.2_g8x8.npy", "py/assets/cohort_ulbp_8.1-8.2_g8x8_clahe.npz",
];
let py, core;

async function boot() {
  postMessage({ type: "progress", text: "Loading Python…", pct: 5 });
  py = await loadPyodide();
  postMessage({ type: "progress", text: "Loading OpenCV and NumPy (about 15 MB, cached after the first visit)…", pct: 25 });
  await py.loadPackage(["numpy", "opencv-python"]);
  postMessage({ type: "progress", text: "Loading the face detectors and the background crowd…", pct: 80 });
  py.FS.mkdirTree("/app/cascades");
  py.FS.mkdirTree("/app/assets");
  await Promise.all(FILES.map(async (f) => {
    const r = await fetch(f);
    if (!r.ok) throw new Error("missing " + f);
    py.FS.writeFile("/app/" + f.slice(3), new Uint8Array(await r.arrayBuffer()));
  }));
  py.runPython(`import sys, os; sys.path.insert(0, "/app"); os.chdir("/app")`);
  core = py.pyimport("webcore").core;
  postMessage({ type: "ready" });
}

const toJs = (o) => (o && o.toJs ? o.toJs({ dict_converter: Object.fromEntries }) : o);

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "boot") return await boot();
    if (!core) return;
    if (m.type === "frame") {
      const res = toJs(core.frame(m.rgba, m.w, m.h, m.now));
      const transfer = [];
      for (const k of ["view", "explain"]) if (res[k]) { res[k] = new Uint8ClampedArray(res[k]); transfer.push(res[k].buffer); }
      if (res.enrolled) { res.enrolled.chips = new Uint8Array(res.enrolled.chips); res.enrolled.thumb = new Uint8ClampedArray(res.enrolled.thumb); }
      postMessage({ type: "result", res }, transfer);
    } else if (m.type === "load") {
      core.load_students(py.toPy(m.students));
      core.set_present(py.toPy(m.present));
      postMessage({ type: "loaded" });
    } else if (m.type === "present") {
      core.set_present(py.toPy(m.present));
    } else if (m.type === "config") {
      core.configure(py.toPy(m.values));
    } else if (m.type === "mode") {
      core.mode = m.mode;
    } else if (m.type === "view") {
      core.view = m.view;
    } else if (m.type === "enrol") {
      core.start_enrol(m.id, m.roll, m.name);
    } else if (m.type === "cancelEnrol") {
      const r = toJs(core.cancel_enrol());
      if (r) { r.chips = new Uint8Array(r.chips); r.thumb = new Uint8ClampedArray(r.thumb); }
      postMessage({ type: "result", res: { enrolled: r, faces: [], cancelled: !r } });
    } else if (m.type === "remove") {
      core.remove(m.id);
    }
  } catch (err) {
    postMessage({ type: "error", text: String(err).slice(0, 500) });
  }
};
