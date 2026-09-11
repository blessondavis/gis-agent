// gis-agent workspace: map + conversational agent, no framework, no build step.
import { api, $, el, toast, fmt } from "/static/app.js";
import { createEditor } from "/static/editor.js";

let editor = null;

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
  // One canvas renderer for every vector layer. Leaflet only routes a click to
  // paths on the canvas that received it, so a second canvas stacked on top
  // (as preferCanvas creates for later layers) silently eats clicks on roads.
  // The tolerance makes 2 px centrelines clickable.
  state.map = L.map("map", {
    zoomControl: true,
    renderer: L.canvas({ padding: 0.5, tolerance: 7 }),
  }).setView([42.78, -71.18], 14);
  state.layers.base = L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    { maxZoom: 19, opacity: 0.6, attribution: "&copy; OpenStreetMap" }
  );
  enableAreaPicking();
  editor = createEditor({
    map: state.map,
    getJob: () => state.job,
    onNetwork: (fc, stats) => { showNetwork(fc); renderCompletion(stats); },
  });
}

/** Swap in a new network layer, keeping the user's layer toggle. */
function showNetwork(fc) {
  if (state.layers.vec) state.map.removeLayer(state.layers.vec);
  state.layers.vec = editor.render(fc);
  applyToggles();
}

/** Drag a rectangle on the map without pulling in another library. */
function enableAreaPicking() {
  const map = state.map;
  let start = null, ghost = null, armed = false;

  const arm = (on) => {
    if (on) editor?.reset();
    armed = on;
    document.body.classList.toggle("picking", on);
    $("#btn-pick").textContent = on ? "Cancel selection" : "Rework area with agent";
    $("#mapnote").textContent = on ? "Drag a box over the area the agent should redo." : "";
    map.dragging[on ? "disable" : "enable"]();
  };

  $("#btn-pick").addEventListener("click", () => arm(!armed));
  $("#btn-clearpick").addEventListener("click", () => {
    if (state.rect) { map.removeLayer(state.rect); state.rect = null; }
    state.bbox = null;
    $("#btn-clearpick").style.display = "none";
    $("#mapnote").textContent = "";
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

  let network = null;
  try {
    network = await api.get(`/api/jobs/${state.job}/network.geojson?t=${Date.now()}`);
    state.layers.vec = editor.render(network);
  } catch { /* not vectorized yet */ }

  applyToggles();
  if (prev.bounds && !state.bbox) {
    const [w, s, e, n] = prev.bounds;
    state.map.fitBounds([[s, w], [n, e]], { padding: [14, 14] });
  }

  renderMetrics(status.metrics);
  state.lastIoU = status.metrics?.iou;
  renderVectors(status.vector_stats, status.manifest?.refinements);
  state.hasTruth = Boolean(status.region?.truth);
  renderCompletion(network?.stats);
  refreshSuggestions();
}

/* How finished the network is, and who did the finishing. With ground truth
   the question "how much of the road network is there" has a direct answer
   (length-based completeness); without it, the split of work is still useful. */
let scoreTimer = null;

function renderCompletion(stats) {
  const box = $("#completion");
  if (!stats) {
    box.replaceChildren(el("p", { class: "empty" }, "no network yet"));
    $("#comp-note").textContent = "";
    return;
  }
  const km = (m) => (m / 1000).toFixed(m < 10000 ? 2 : 1);
  const share = stats.human_share || 0;
  $("#comp-note").textContent = stats.n_ops ? `${stats.n_ops} edit${stats.n_ops === 1 ? "" : "s"}` : "";
  box.replaceChildren(
    el("div", { class: "splitbar", title: "share of network length by source" },
      el("span", { class: "model", style: `flex:${Math.max(1 - share, 0.001)}` }),
      el("span", { class: "human", style: `flex:${Math.max(share, 0.001)}` })),
    el("dl", { class: "kv" },
      el("dt", {}, "model"), el("dd", {}, `${km(stats.model_length_m)} km · ${stats.n_model}`),
      el("dt", {}, "drawn by you"), el("dd", {}, `${km(stats.human_length_m)} km · ${stats.n_human}`),
      el("dt", {}, "removed"), el("dd", {}, String(stats.n_deleted + stats.n_superseded))),
    el("div", { id: "comp-score" }));
  $("#btn-undo").disabled = !stats.can_undo;
  $("#btn-redo").disabled = !stats.can_redo;
  if (state.hasTruth) {
    clearTimeout(scoreTimer);
    scoreTimer = setTimeout(loadNetworkScore, 400);
  }
}

async function loadNetworkScore() {
  const box = $("#comp-score");
  if (!box || !state.job) return;
  let s;
  try { s = await api.get(`/api/jobs/${state.job}/network/score`); } catch { return; }
  const pct = (v) => `${(100 * v).toFixed(1)}%`;
  const gain = (v) => v ? el("span", { class: v > 0 ? "up" : "down" },
    ` ${v > 0 ? "+" : ""}${(100 * v).toFixed(1)}`) : null;
  box.replaceChildren(
    el("div", { class: "comp" },
      el("div", { class: "v" }, pct(s.overall.completeness), gain(s.edit_gain.completeness)),
      el("div", { class: "k" }, `of real roads found (model alone ${pct(s.model_only.completeness)})`)),
    el("div", { class: "comp" },
      el("div", { class: "v" }, pct(s.overall.correctness), gain(s.edit_gain.correctness)),
      el("div", { class: "k" }, `of drawn roads are real (model alone ${pct(s.model_only.correctness)})`)),
    el("p", { class: "hint" }, `Length-based, within ${s.tolerance_m} m of the reference centreline.`));
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
  dl.href = `/api/jobs/${state.job}/network.geojson`;
  dl.download = `${state.job}-network.geojson`;
  dl.style.display = "inline";
}

/* ------------------------------------------------------------------ chat */

/* The agent panel renders typed events from the harness: each kind of thing
   is shown as what it is -- a plan checklist, a tool call that resolves in
   place, a measured change, a verification -- instead of one scrolling log. */

const calls = new Map();       // call_id -> step element, so results land on their call
const CHANGES_MAP = new Set(["stitch_result", "vectorize_result", "refine_area",
                             "apply_candidate", "repair_geometry"]);

function chatlog() {
  const log = $("#chatlog");
  if (log.querySelector(".empty")) log.replaceChildren();
  return log;
}

function scrollDown() { const l = $("#chatlog"); l.scrollTop = l.scrollHeight; }

/* The small subset of Markdown agents actually write -- bold, inline code,
   bullets, headings -- built as DOM nodes, never innerHTML, so model output
   cannot inject markup. */
function inlineMd(text) {
  const out = [];
  const re = /(\*\*[^*\n]+\*\*|`[^`\n]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0];
    out.push(t.startsWith("**") ? el("strong", {}, t.slice(2, -2)) : el("code", {}, t.slice(1, -1)));
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function markdown(text) {
  const blocks = [];
  let list = null;
  for (const raw of String(text).split("\n")) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-*•]\s+(.*)/);
    if (bullet) {
      if (!list) { list = el("ul"); blocks.push(list); }
      list.append(el("li", {}, inlineMd(bullet[1])));
      continue;
    }
    list = null;
    const head = line.match(/^#{1,4}\s+(.*)/);
    if (head) blocks.push(el("div", { class: "md-h" }, inlineMd(head[1])));
    else if (!line.trim()) blocks.push(el("div", { class: "md-gap" }));
    else blocks.push(el("div", {}, inlineMd(line)));
  }
  return blocks;
}

function addMessage(role, text, meta = {}) {
  const node = role === "user"
    ? el("div", { class: "msg user" }, text)
    : el("div", { class: "msg bot md" }, markdown(text));
  if (role === "user" && meta.checkpoint !== undefined) attachRewind(node, meta.index);
  chatlog().append(node);
  scrollDown();
  return node;
}

function attachRewind(node, index) {
  node.dataset.index = index;
  node.append(el("button", {
    class: "rewind", title: "Rewind: put the map, results and your edits back " +
      "to how they were just before this message, and drop everything after it",
    onclick: () => rewind(index),
  }, "↺"));
}

async function rewind(index) {
  if (state.busy) return toast("the agent is working; wait or stop it first", true);
  if (!confirm("Rewind to before this message? Results, map and your edits go " +
               "back to that point; later messages are removed.")) return;
  try {
    const r = await api.post(`/api/jobs/${state.job}/rewind`, { message_index: index });
    toast(`Rewound to before “${r.label.slice(0, 40)}”`);
    await Promise.all([loadJob(), loadConversation()]);
  } catch (e) { toast(e.message, true); }
}

function argText(args) {
  return Object.entries(args || {}).filter(([k]) => k !== "job_id")
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`).join("  ");
}

function renderPlan(plan) {
  const box = $("#planbox");
  if (!plan?.steps?.length) { box.hidden = true; return; }
  const icon = { pending: "○", in_progress: "◐", done: "●", skipped: "–" };
  const done = plan.steps.filter(s => s.status === "done").length;
  box.hidden = false;
  box.replaceChildren(
    el("div", { class: "plan-head" }, el("strong", {}, "Plan"),
      el("span", { class: "dimtext" }, `${done}/${plan.steps.length}`)),
    el("ol", {}, plan.steps.map(s =>
      el("li", { class: s.status }, el("span", { class: "ic" }, icon[s.status] || "○"), s.title))));
}

function metricDelta(ev) {
  // a measured change is the most interesting event in the stream
  const r = ev.result || {};
  const now = r.iou ?? r.metrics?.iou;
  if (now === undefined) return null;
  const before = state.lastIoU;
  state.lastIoU = now;
  if (before === undefined || before === null || Math.abs(now - before) < 0.0005) return null;
  return el("div", { class: "delta" }, "IoU ", el("span", {}, before.toFixed(3)),
    el("span", { class: "arrow" }, "→"),
    el("span", { class: "to " + (now > before ? "up" : "down") }, now.toFixed(3)));
}

function addActivity(ev) {
  if (ev.type === "connected" || ev.type === "status") return;
  if (ev.type === "eof") { setBusy(false); loadJob(); return; }
  if (ev.type === "checkpoint") {
    const last = [...$("#chatlog").querySelectorAll(".msg.user")].pop();
    if (last && !last.querySelector(".rewind")) attachRewind(last, ev.message_index);
    return;
  }
  if (ev.type === "plan") return renderPlan(ev.result);
  if (ev.type === "message" || ev.type === "done") {
    // "done" repeats the final message; show it once
    if (ev.text && ev.type === "message") addMessage("agent", ev.text);
    return;
  }
  const log = chatlog();

  if (ev.type === "plan_ready") {
    log.append(el("div", { class: "approve" },
      el("span", {}, "Plan ready. Nothing has been changed yet."),
      el("button", { class: "sm", onclick: (e) => {
        e.target.closest(".approve").remove();
        send("Approved. Carry out the plan.", { mode: "work" });
      } }, "Approve & run"),
      el("button", { class: "sm ghost", onclick: (e) => {
        e.target.closest(".approve").remove(); $("#msg").focus();
      } }, "Revise")));
    return scrollDown();
  }
  if (ev.type === "thinking") {
    // Reasoning models put their whole chain of thought in the content --
    // thousands of characters per step. Keep the first sentence in view and
    // fold the rest away.
    const text = ev.text || "";
    if (text.length <= 280) {
      log.append(el("div", { class: "narration" }, text));
    } else {
      const first = (text.match(/^[\s\S]{20,240}?[.!?](\s|$)/) || [text.slice(0, 200) + "…"])[0].trim();
      log.append(el("details", { class: "step", "data-kind": "think" },
        el("summary", {}, el("span", { class: "sumtext" }, first),
          el("span", { class: "count" }, `${text.split(/\s+/).length} words`)),
        el("div", { class: "body" }, text)));
    }
    return scrollDown();
  }
  if (ev.type === "edit") {
    const accepted = ev.op === "add" && (ev.note || "").startsWith("accepted suggestion");
    const what = accepted ? "accepted a suggestion" :
      ({ add: "drew a road", delete: "deleted a road", replace: "reshaped a road",
         dismiss: "dismissed a suggestion" }[ev.op] || ev.op);
    const verb = ev.action === "undo" ? `undid: ${what}` : ev.action === "redo" ? `redid: ${what}` : what;
    log.append(el("details", { class: "step", "data-kind": "edit" },
      el("summary", {}, `you ${verb}`,
        el("span", { class: "count" }, `${(ev.stats.human_length_m / 1000).toFixed(2)} km yours`))));
    return scrollDown();
  }
  if (ev.type === "verify") {
    log.append(el("details", { class: "step" + (ev.ok ? "" : " fail"), "data-kind": ev.ok ? "ok" : "fail" },
      el("summary", {}, el("span", { class: "tag " + (ev.ok ? "ok" : "fail") }, ev.ok ? "verified" : "check"),
        ev.ok ? "result measured; quoted scores match tool output" : "harness sent the agent back"),
      el("div", { class: "body" }, ev.text)));
    return scrollDown();
  }
  if (ev.type === "error") {
    log.append(el("details", { class: "step fail", "data-kind": "fail", open: "" },
      el("summary", {}, el("span", { class: "tag fail" }, "error"), "stopped"),
      el("div", { class: "body" }, ev.text || "")));
    return scrollDown();
  }
  if (ev.type === "tool_call") {
    const node = el("details", { class: "step", "data-kind": "work" },
      el("summary", {}, el("span", { class: "spinner" }), el("b", {}, ev.tool),
        el("span", { class: "dimtext argline" }, argText(ev.args))),
      el("div", { class: "body" }, argText(ev.args) || "(no arguments)"));
    calls.set(ev.call_id, node);
    log.append(node);
    return scrollDown();
  }
  if (ev.type === "tool_result") {
    let node = calls.get(ev.call_id);
    if (!node) { node = el("details", { class: "step" }, el("summary", {})); log.append(node); }
    node.dataset.kind = ev.ok ? "ok" : "fail";
    if (!ev.ok) node.classList.add("fail");
    node.querySelector("summary").replaceChildren(
      el("b", {}, ev.tool), el("span", { class: "sumtext" }, ev.summary || (ev.ok ? "done" : "failed")),
      el("span", { class: "count" }, `${(ev.duration_s || 0).toFixed(1)}s`));
    node.querySelector(".body")?.append(
      "\n\n", JSON.stringify(ev.result, null, 1).slice(0, 3000));
    const d = metricDelta(ev);
    if (d) node.after(d);
    if (CHANGES_MAP.has(ev.tool) && ev.ok && !state.replaying) setTimeout(loadJob, 300);
    return scrollDown();
  }
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
  calls.clear();
  renderPlan(null);
  try {
    const c = await api.get(`/api/jobs/${state.job}/conversation`);
    const replayable = c.events.some(e => e.type === "user");
    if (replayable) {
      // session replay: rebuild the whole panel -- plan, tool cards, checks,
      // edits -- from the job's event log, exactly as it streamed
      state.replaying = true;
      state.lastIoU = undefined;           // deltas are relative to the log's own history
      for (const ev of c.events) {
        if (ev.type === "user") {
          addMessage("user", ev.text, ev.rewindable ? { checkpoint: true, index: ev.index } : {});
        } else if (ev.type !== "eof" && ev.type !== "checkpoint") {
          addActivity(ev);
        }
      }
      state.replaying = false;
      document.querySelectorAll(".approve").forEach((n, i, all) => {
        if (i < all.length - 1) n.remove();   // only the latest plan is still pending
      });
    } else {
      for (const m of c.messages) {
        addMessage(m.role === "user" ? "user" : "agent", m.content,
                   m.checkpoint ? { checkpoint: m.checkpoint, index: m.index } : {});
      }
    }
    renderPlan(c.plan);
    if (!c.messages.length) {
      log.replaceChildren(el("p", { class: "empty" },
        "Ask the agent to annotate this region, or pick an area on the map and say what is wrong with it."));
    }
    setBusy(c.state === "running");
  } catch { /* new job */ }
}

function setBusy(on) {
  state.busy = on;
  const btn = $("#send");
  btn.textContent = on ? "Stop" : "Send";
  btn.classList.toggle("stop", on);
  const pill = $("#agentstate");
  pill.className = "pill" + (on ? " busy" : "");
  pill.replaceChildren(el("span", { class: "dot" }), on ? "working" : "idle");
  refreshSuggestions();
}

async function stop() {
  try { await api.post(`/api/jobs/${state.job}/chat/cancel`); } catch {}
}

async function send(text, opts = {}) {
  if (!state.job) return toast("create or pick a region first", true);
  if (state.busy) return;
  const message = (text ?? $("#msg").value).trim();
  if (!message) return;
  const mode = opts.mode || ($("#planfirst").checked ? "plan" : "work");

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
    await api.post(`/api/jobs/${state.job}/chat`, { message: payload, mode });
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

// edit tools
for (const b of document.querySelectorAll("[data-tool]")) {
  b.addEventListener("click", () => {
    if (!state.job) return toast("create or pick a region first", true);
    editor.setMode(b.dataset.tool);
  });
}
$("#btn-undo").addEventListener("click", () => editor.history("undo"));
$("#btn-redo").addEventListener("click", () => editor.history("redo"));

// chat controls
$("#send").addEventListener("click", () => (state.busy ? stop() : send()));
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
// handle for the browser console and for UI automation
window.gisWorkspace = { map: state.map, editor, state };
wireTabs();
loadHealth();
await loadJobs();
await loadJob();
await loadConversation();
connectSocket();
refreshSuggestions();
