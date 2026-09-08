// gis-agent workspace: map + conversational agent, no framework, no build step.
import { api, $, el, toast, fmt } from "/static/app.js";

const state = {
  job: localStorage.getItem("gisagent.job") || "",
  bbox: null,          // [w,s,e,n] of the area the user selected
  socket: null,
  busy: false,
  layers: {},
  map: null,
  rect: null,
};

/* ------------------------------------------------------------------ map */

function initMap() {
  state.map = L.map("map", { zoomControl: true, preferCanvas: true })
    .setView([42.78, -71.18], 14);
  state.layers.base = L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    { maxZoom: 19, opacity: 0.6, attribution: "&copy; OpenStreetMap" }
  );
  enableAreaPicking();
}

/** Drag a rectangle on the map without pulling in another library. */
function enableAreaPicking() {
  const map = state.map;
  let start = null, ghost = null, armed = false;

  const arm = (on) => {
    armed = on;
    document.body.classList.toggle("picking", on);
    $("#btn-pick").textContent = on ? "Cancel selection" : "Select area to rework";
    map.dragging[on ? "disable" : "enable"]();
  };

  $("#btn-pick").addEventListener("click", () => arm(!armed));
  $("#btn-clearpick").addEventListener("click", () => {
    if (state.rect) { map.removeLayer(state.rect); state.rect = null; }
    state.bbox = null;
    $("#btn-clearpick").style.display = "none";
    $("#mapnote").textContent = "Drag a box on the map, then tell the agent what is wrong there.";
    refreshSuggestions();
  });

  map.on("mousedown", (e) => {
    if (!armed) return;
    start = e.latlng;
    if (ghost) map.removeLayer(ghost);
    ghost = L.rectangle([start, start], {
      color: "#4c9aff", weight: 1.5, fillOpacity: 0.12, dashArray: "5,4",
    }).addTo(map);
  });

  map.on("mousemove", (e) => {
    if (armed && start && ghost) ghost.setBounds(L.latLngBounds(start, e.latlng));
  });

  map.on("mouseup", (e) => {
    if (!armed || !start) return;
    const b = L.latLngBounds(start, e.latlng);
    start = null;
    if (ghost) { map.removeLayer(ghost); ghost = null; }
    if (Math.abs(b.getEast() - b.getWest()) < 1e-5) { arm(false); return; }

    if (state.rect) map.removeLayer(state.rect);
    state.rect = L.rectangle(b, {
      color: "#4c9aff", weight: 2, fillOpacity: 0.1,
    }).addTo(map);
    state.bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
    $("#btn-clearpick").style.display = "";
    $("#mapnote").innerHTML =
      `Area selected (${(state.bbox[2] - state.bbox[0]).toFixed(4)}&deg; x ` +
      `${(state.bbox[3] - state.bbox[1]).toFixed(4)}&deg;). Tell the agent what to do with it.`;
    arm(false);
    refreshSuggestions();
    $("#msg").focus();
  });
}

/** Colour a centreline by how confident the model was about it. */
function roadStyle(feature) {
  const c = feature?.properties?.confidence ?? 0;
  // low confidence fades toward orange, high stays bright yellow
  const colour = c >= 0.7 ? "#ffd028" : c >= 0.45 ? "#ffa53b" : "#ff6b6b";
  return { color: colour, weight: c >= 0.7 ? 2.6 : 2, opacity: 0.45 + 0.5 * Math.min(c, 1) };
}

function clearOverlays() {
  for (const k of ["image", "mask", "truth", "vec"]) {
    if (state.layers[k]) { state.map.removeLayer(state.layers[k]); delete state.layers[k]; }
  }
}

function applyToggles() {
  const set = (key, on) => {
    const lyr = state.layers[key];
    if (!lyr) return;
    if (on && !state.map.hasLayer(lyr)) state.map.addLayer(lyr);
    if (!on && state.map.hasLayer(lyr)) state.map.removeLayer(lyr);
  };
  set("image", $("#l-image").checked);
  set("mask", $("#l-mask").checked);
  set("truth", $("#l-truth").checked);
  set("vec", $("#l-vec").checked);
  set("base", $("#l-base").checked);
  if (state.rect) state.rect.bringToFront();
}

/* --------------------------------------------------------------- loading */

async function loadJobs() {
  let jobs = [];
  try { ({ jobs } = await api.get("/api/jobs")); } catch (e) { return; }
  const box = $("#joblist");
  if (!jobs.length) {
    box.replaceChildren(el("p", { class: "empty" }, "no regions yet"));
    return;
  }
  if (!state.job || !jobs.some(j => j.job_id === state.job)) state.job = jobs[0].job_id;
  box.replaceChildren(...jobs.map(j =>
    el("div", {
      class: "job" + (j.job_id === state.job ? " sel" : ""),
      onclick: () => selectJob(j.job_id),
    },
      el("div", { class: "id" }, j.manifest?.name || j.job_id),
      el("div", { class: "sub" },
        el("span", {}, `${j.stage} - ${j.region?.n_tiles ?? "?"} tile(s)`),
        j.metrics
          ? el("span", { class: "score q-" + band(j.metrics.iou).key },
               fmt(j.metrics.iou))
          : el("span", { class: "score dimtext" }, "--")))));
}

async function selectJob(id) {
  state.job = id;
  localStorage.setItem("gisagent.job", id);
  await Promise.all([loadJobs(), loadJob(), loadConversation()]);
  connectSocket();
}

async function loadJob() {
  if (!state.job) return;
  clearOverlays();
  let status, prev;
  try {
    [status, prev] = await Promise.all([
      api.get(`/api/jobs/${state.job}`),
      api.get(`/api/jobs/${state.job}/previews`),
    ]);
  } catch (e) { return toast(e.message, true); }

  $("#stage").textContent = status.stage;

  const p = prev.previews || {};
  const op = Number($("#opacity").value) / 100;
  if (p.image) state.layers.image = L.imageOverlay(bust(p.image.url), p.image.bounds, { opacity: 1 });
  if (p.mask) state.layers.mask = L.imageOverlay(bust(p.mask.url), p.mask.bounds, { opacity: op });
  if (p.truth) state.layers.truth = L.imageOverlay(bust(p.truth.url), p.truth.bounds, { opacity: op });

  try {
    const gj = await api.get(`/api/jobs/${state.job}/roads.geojson?t=${Date.now()}`);
    state.layers.vec = L.geoJSON(gj, {
      style: roadStyle,
      onEachFeature: (f, lyr) => {
        const pr = f.properties || {};
        lyr.bindPopup(
          `<b>road centreline</b><br>length ${Number(pr.length_m || 0).toFixed(1)} m` +
          `<br>confidence <b>${pr.confidence_pct ?? "?"}%</b>`);
      },
    });
  } catch { /* not vectorized yet */ }

  applyToggles();
  if (prev.bounds && !state.bbox) {
    const [w, s, e, n] = prev.bounds;
    state.map.fitBounds([[s, w], [n, e]], { padding: [14, 14] });
  }

  renderMetrics(status.metrics);
  renderVectors(status.vector_stats, status.manifest?.refinements);
  refreshSuggestions();
}

const bust = (u) => `${u}?t=${Date.now()}`;

/* A bare decimal cannot tell an analyst that 0.098 means "do not ship this".
   Bands come from the measured runs: the trained model lands around 0.60 on
   suburban and 0.45 on dense urban, while zero-shot on a city grid collapsed
   to 0.098 -- a map missing ~90% of its streets. */
const BANDS = [
  { min: 0.55, key: "good",   label: "Good",            note: "review and export" },
  { min: 0.35, key: "usable", label: "Usable",          note: "check false positives before shipping" },
  { min: 0.15, key: "poor",   label: "Poor",            note: "refine before using" },
  { min: -1,   key: "failed", label: "Failed",          note: "most roads are missing - do not ship" },
];

function band(iou) {
  return BANDS.find(b => iou >= b.min) || BANDS[BANDS.length - 1];
}

function renderVerdict(m) {
  const box = $("#verdict");
  if (!box) return;
  if (!m) { box.replaceChildren(); return; }
  const b = band(m.iou);
  box.replaceChildren(el("div", { class: "verdict " + b.key },
    el("strong", {}, b.label), el("span", {}, b.note)));
}

function renderMetrics(m) {
  const box = $("#metrics");
  renderVerdict(m);
  if (!m) { box.replaceChildren(el("p", { class: "empty" }, "not evaluated yet"));
            $("#acc-note").textContent = ""; return; }
  $("#acc-note").textContent = `${m.slack_px}px slack`;
  const tile = (k, v, hero) =>
    el("div", { class: "metric" + (hero ? " hero" : "") },
      el("div", { class: "v" }, fmt(v)), el("div", { class: "k" }, k));
  box.replaceChildren(el("div", { class: "metrics" },
    tile("IoU", m.iou, true), tile("F1", m.f1), tile("relaxed F1", m.relaxed_f1),
    tile("precision", m.precision), tile("recall", m.recall),
    tile("rel. recall", m.relaxed_recall)));
}

function renderVectors(v, refinements) {
  const box = $("#vecstats");
  if (!v) {
    box.replaceChildren(el("dt", {}, "status"), el("dd", { class: "dimtext" }, "none yet"));
    $("#download").style.display = "none";
    return;
  }
  const rows = [
    ["segments", v.n_features],
    ["total length", `${fmt(v.total_length_km, 2)} km`],
    ["mean confidence", `${(100 * (v.mean_confidence || 0)).toFixed(0)}%`],
    ["low confidence", v.low_confidence_features ?? 0],
  ];
  if (refinements?.length) rows.push(["reworks", refinements.length]);
  box.replaceChildren(...rows.flatMap(([k, val]) =>
    [el("dt", {}, k), el("dd", {}, String(val))]));
  const dl = $("#download");
  dl.href = `/api/jobs/${state.job}/roads.geojson`;
  dl.download = `${state.job}-roads.geojson`;
  dl.style.display = "inline";
}

/* ------------------------------------------------------------------ chat */

function addMessage(role, text) {
  const log = $("#chatlog");
  if (log.querySelector(".empty")) log.replaceChildren();
  log.append(el("div", { class: `msg ${role}` },
    el("div", { class: "who" }, role === "user" ? "you" : "agent"),
    el("div", { class: "bubble" }, text)));
  log.scrollTop = log.scrollHeight;
}

function addActivity(ev) {
  const log = $("#chatlog");
  if (log.querySelector(".empty")) log.replaceChildren();

  if (ev.type === "message" || ev.type === "done") {
    if (ev.text) addMessage("agent", ev.text);
    return;
  }
  if (ev.type === "eof") { setBusy(false); loadJob(); return; }
  if (ev.type === "connected" || ev.type === "status") return;

  const line = el("div", { class: "line" });
  if (ev.type === "tool_call") {
    line.append(el("span", { class: "spinner" }), el("span", { class: "name" }, ev.tool));
  } else if (ev.type === "tool_result") {
    line.append(el("span", { style: "color:var(--good)" }, "✓"),
                el("span", { class: "name" }, ev.tool));
  } else if (ev.type === "thinking") {
    line.append(el("span", { class: "name", style: "color:var(--warn)" }, "reasoning"));
  } else if (ev.type === "error") {
    line.append(el("span", { style: "color:var(--bad)" }, "✕"),
                el("span", { class: "name" }, "error"));
  }
  if (ev.duration_s) line.append(el("span", { class: "time" }, `${ev.duration_s.toFixed(1)}s`));

  const node = el("div", { class: `act ${ev.type}` }, line);

  if (ev.type === "thinking" && ev.text) {
    node.append(el("div", { class: "sum" }, ev.text));
  }
  if (ev.type === "tool_call" && Object.keys(ev.args || {}).length) {
    const args = Object.entries(ev.args)
      .filter(([k]) => k !== "job_id")
      .map(([k, v]) => `${k}=${JSON.stringify(v)}`).join("  ");
    if (args) node.append(el("div", { class: "sum" }, args));
  }
  if (ev.type === "tool_result") {
    if (ev.summary) node.append(el("div", { class: "sum" }, ev.summary));
    const det = el("details", {}, el("summary", {}, "raw"),
      el("pre", {}, JSON.stringify(ev.result, null, 1).slice(0, 4000)));
    node.append(det);
    // a tool that changed the output means the map is stale
    if (["stitch_result", "vectorize_result", "refine_area"].includes(ev.tool)) {
      setTimeout(loadJob, 300);
    }
  }
  if (ev.type === "error" && ev.text) {
    node.append(el("div", { class: "sum" }, ev.text));
  }

  // replace the pending spinner for a call once its result lands
  if (ev.type === "tool_result") {
    const pending = [...log.querySelectorAll(".act.tool_call")].reverse()
      .find(n => n.querySelector(".name")?.textContent === ev.tool && n.dataset.done !== "1");
    if (pending) { pending.dataset.done = "1"; pending.querySelector(".spinner")?.remove(); }
  }

  log.append(node);
  log.scrollTop = log.scrollHeight;
}

function connectSocket() {
  if (state.socket) { state.socket.close(); state.socket = null; }
  if (!state.job) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const s = new WebSocket(`${proto}://${location.host}/ws/jobs/${state.job}`);
  s.onmessage = (m) => { try { addActivity(JSON.parse(m.data)); } catch {} };
  s.onclose = () => { if (state.socket === s) state.socket = null; };
  state.socket = s;
}

async function loadConversation() {
  const log = $("#chatlog");
  log.replaceChildren();
  if (!state.job) return;
  try {
    const c = await api.get(`/api/jobs/${state.job}/conversation`);
    for (const m of c.messages) addMessage(m.role === "user" ? "user" : "agent", m.content);
    if (!c.messages.length) {
      log.replaceChildren(el("p", { class: "empty" },
        "Ask the agent to annotate this region, or pick an area on the map and say what is wrong with it."));
    }
    setBusy(c.state === "running");
  } catch { /* new job */ }
}

function setBusy(on) {
  state.busy = on;
  $("#send").disabled = on;
  const pill = $("#agentstate");
  pill.className = "pill" + (on ? " ok" : "");
  pill.replaceChildren(
    on ? el("span", { class: "spinner" }) : el("span", { class: "dot" }),
    on ? "working" : "idle");
}

async function send(text) {
  if (!state.job) return toast("create or pick a region first", true);
  const message = (text ?? $("#msg").value).trim();
  if (!message) return;

  // attach the drawn area so the agent gets exact coordinates
  let payload = message;
  if (state.bbox) {
    payload += `\n\n[The user selected this area on the map: bbox_wgs84 = ` +
      `[${state.bbox.map(v => v.toFixed(6)).join(", ")}]. ` +
      `Use refine_area with this bbox_wgs84 for the rework.]`;
  }

  addMessage("user", message);
  $("#msg").value = "";
  $("#msg").style.height = "auto";
  setBusy(true);
  try {
    if (!state.socket || state.socket.readyState !== 1) connectSocket();
    await api.post(`/api/jobs/${state.job}/chat`, { message: payload });
  } catch (e) {
    setBusy(false);
    toast(e.message, true, 9000);
  }
}

function refreshSuggestions() {
  const box = $("#suggestions");
  const opts = state.bbox
    ? ["The roads in this area are missing - rework it",
       "This area has too many false roads",
       "Re-run this area with a higher upscale"]
    : ["Annotate the roads in this region",
       "Try to improve the score",
       "Which roads are least confident?"];
  box.replaceChildren(...opts.map(t =>
    el("button", { class: "sm", onclick: () => send(t), disabled: state.busy ? "" : null }, t)));
}

/* ------------------------------------------------------------- bootstrap */

async function loadHealth() {
  try {
    const h = await api.get("/api/health");
    const badge = (ok, label) =>
      el("span", { class: `pill ${ok ? "ok" : "no"}` }, el("span", { class: "dot" }), label);
    $("#health").replaceChildren(
      badge(h.cuda, h.gpu ? h.gpu.replace(/NVIDIA GeForce /, "") : "no GPU"),
      badge(h.qgis, "QGIS"),
      badge(h.sam_token, "SAM 3"),
      badge(h.llm_configured, (h.llm_model || "LLM").split("/").pop()));
  } catch {}
}

// layer + opacity controls
for (const id of ["l-image", "l-mask", "l-truth", "l-vec", "l-base"]) {
  $("#" + id).addEventListener("change", applyToggles);
}
$("#opacity").addEventListener("input", (e) => {
  const v = Number(e.target.value) / 100;
  $("#opval").textContent = e.target.value + "%";
  state.layers.mask?.setOpacity(v);
  state.layers.truth?.setOpacity(v);
});

// chat controls
$("#send").addEventListener("click", () => send());
$("#msg").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
$("#msg").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(e.target.scrollHeight, 130) + "px";
});
$("#btn-reset").addEventListener("click", async () => {
  if (!state.job) return;
  await fetch(`/api/jobs/${state.job}/conversation`, { method: "DELETE" });
  loadConversation();
});

// new region
$("#btn-new").addEventListener("click", () => $("#dlg-new").showModal());
$("#do-new").addEventListener("click", async () => {
  const btn = $("#do-new");
  btn.disabled = true;
  $("#newmsg").innerHTML = '<span class="spinner"></span> screening tiles and building the mosaic...';
  try {
    const res = await api.post("/api/jobs", {
      split: $("#split").value,
      block: Number($("#block").value),
      name: $("#jobname").value.trim(),
      screen: $("#screen").checked,
    });
    $("#dlg-new").close();
    $("#newmsg").textContent = "";
    toast(`Region ready: ${res.region.width}x${res.region.height} px`);
    await selectJob(res.job_id);
  } catch (e) {
    $("#newmsg").textContent = e.message;
  } finally { btn.disabled = false; }
});

// upload
$("#btn-upload").addEventListener("click", () => $("#dlg-upload").showModal());
$("#drop").addEventListener("click", () => $("#file").click());
$("#drop").addEventListener("dragover", (e) => { e.preventDefault(); $("#drop").classList.add("over"); });
$("#drop").addEventListener("dragleave", () => $("#drop").classList.remove("over"));
$("#drop").addEventListener("drop", (e) => {
  e.preventDefault(); $("#drop").classList.remove("over");
  if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
});
$("#file").addEventListener("change", (e) => { if (e.target.files[0]) upload(e.target.files[0]); });

async function upload(file) {
  $("#upmsg").innerHTML = `<span class="spinner"></span> uploading ${file.name}...`;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await fetch("/api/jobs/upload", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.text()).slice(0, 300));
    const res = await r.json();
    $("#dlg-upload").close();
    $("#upmsg").textContent = "";
    toast("Uploaded. Ask the agent to annotate it.");
    await selectJob(res.job_id);
  } catch (e) { $("#upmsg").textContent = e.message; }
}

/* Work shows the agent and its tools; Review hides them so the output can be
   checked without the machinery in the way. Diagnostic swaps the layer list
   for the correct / missed / false-positive view, which is a different
   question from "which layers are on". */
function wireTabs() {
  const modes = $("#modeswitch");
  if (modes) {
    modes.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-mode]");
      if (!btn) return;
      [...modes.children].forEach(b =>
        b.setAttribute("aria-selected", String(b === btn)));
      document.body.dataset.mode = btn.dataset.mode;
      const right = document.querySelector(".col.right");
      if (right) right.style.display = btn.dataset.mode === "review" ? "none" : "";
      const ws = document.querySelector(".workspace");
      if (ws) ws.style.gridTemplateColumns =
        btn.dataset.mode === "review" ? "300px minmax(0, 1fr)" : "";
      setTimeout(() => state.map && state.map.invalidateSize(), 60);
    });
  }

  const views = $("#viewtabs");
  if (views) {
    views.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-view]");
      if (!btn) return;
      [...views.children].forEach(b =>
        b.setAttribute("aria-selected", String(b === btn)));
      const diag = btn.dataset.view === "diagnostic";
      document.body.dataset.view = btn.dataset.view;
      // diagnostic = predicted vs truth together; layers = user's own choice
      if (diag) {
        $("#l-mask").checked = true;
        $("#l-truth").checked = true;
        $("#l-vec").checked = false;
      } else {
        $("#l-mask").checked = false;
        $("#l-truth").checked = false;
        $("#l-vec").checked = true;
      }
      ["l-mask", "l-truth", "l-vec"].forEach(id =>
        $("#" + id).dispatchEvent(new Event("change")));
    });
  }
}

initMap();
wireTabs();
loadHealth();
await loadJobs();
await loadJob();
await loadConversation();
connectSocket();
refreshSuggestions();
