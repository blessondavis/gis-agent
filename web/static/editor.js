// Manual road editing: draw, reshape and delete centrelines, and review the
// model's suggestions. The model gets most of a network; the last stretch is
// faster to draw than to prompt for, so the person gets real tools here, not
// only a chat box.
//
// Every edit goes to the server as one operation in an undoable log. The
// server merges that log over the machine output, so an edit survives the
// agent re-running the model underneath it.
//
// No drawing library: snapping, handles and the review queue are small enough
// to own, and owning them keeps the no-build, vendored-Leaflet setup intact.

import { api, $, el, toast } from "/static/app.js";

const SNAP_VERTEX_PX = 12;     // prefer an existing junction or road end...
const SNAP_EDGE_PX = 10;       // ...else a point along a road
const COLOURS = {
  human: "#b9a8ff",            // the person's roads, distinct from any model colour
  selected: "#ffffff",
  gap: "#4fe3e8",
  missed: "#ff8fd8",
};

export function createEditor({ map, getJob, onNetwork }) {
  // the map's own renderer: see initMap for why there must be only one
  const renderer = map.options.renderer;
  const st = {
    mode: "view",              // view | draw | select | delete | review
    network: null,
    segs: [],                  // flat segment index for snapping
    draw: null,                // { pts: LatLng[], line, rubber, dots }
    sel: null,                 // { feature, orig, pts, line, handles }
    snapMark: null,
    sugg: null,                // { features, layer, i }
    busy: false,
  };

  /* ------------------------------------------------------------ rendering */

  function modelStyle(f) {
    const c = f.properties?.confidence ?? 0;
    const colour = c >= 0.7 ? "#ffd028" : c >= 0.45 ? "#ffa53b" : "#ff6b6b";
    return { color: colour, weight: c >= 0.7 ? 2.6 : 2,
             opacity: 0.45 + 0.5 * Math.min(c, 1), renderer };
  }

  function style(f) {
    if (f.properties?.source === "human") {
      return { color: COLOURS.human, weight: 3, opacity: 0.95, renderer };
    }
    return modelStyle(f);
  }

  /** Build the network layer. The caller owns adding it to the map. */
  function render(fc) {
    st.network = fc;
    indexSegments(fc);
    return L.geoJSON(fc, {
      style,
      renderer,
      onEachFeature: (f, lyr) => lyr.on("click", (e) => featureClick(f, lyr, e)),
    });
  }

  function indexSegments(fc) {
    st.segs = [];
    for (const f of fc?.features || []) {
      const cs = f.geometry?.coordinates || [];
      for (let i = 0; i + 1 < cs.length; i++) {
        const a = [cs[i][1], cs[i][0]], b = [cs[i + 1][1], cs[i + 1][0]];
        st.segs.push({
          fid: f.properties?.id, a, b, first: i === 0, last: i + 2 === cs.length,
          minLat: Math.min(a[0], b[0]), maxLat: Math.max(a[0], b[0]),
          minLng: Math.min(a[1], b[1]), maxLng: Math.max(a[1], b[1]),
        });
      }
    }
  }

  /* -------------------------------------------------------------- snapping */

  /** Nearest junction/vertex within a few pixels, else nearest point on a road. */
  function snap(latlng, excludeFid) {
    const p = map.latLngToContainerPoint(latlng);
    const q = map.containerPointToLatLng([p.x + SNAP_VERTEX_PX, p.y + SNAP_VERTEX_PX]);
    const dLat = Math.abs(q.lat - latlng.lat), dLng = Math.abs(q.lng - latlng.lng);
    const near = st.segs.filter(s => s.fid !== excludeFid
      && s.maxLat >= latlng.lat - dLat && s.minLat <= latlng.lat + dLat
      && s.maxLng >= latlng.lng - dLng && s.minLng <= latlng.lng + dLng);

    let best = null, bestD = SNAP_VERTEX_PX;
    for (const s of near) {
      for (const v of [s.a, s.b]) {
        const d = map.latLngToContainerPoint(v).distanceTo(p);
        if (d < bestD) { bestD = d; best = { latlng: L.latLng(v), kind: "vertex" }; }
      }
    }
    if (best) return best;

    bestD = SNAP_EDGE_PX;
    for (const s of near) {
      const a = map.latLngToContainerPoint(s.a), b = map.latLngToContainerPoint(s.b);
      const abx = b.x - a.x, aby = b.y - a.y;
      const len2 = abx * abx + aby * aby || 1;
      const t = Math.max(0, Math.min(1, ((p.x - a.x) * abx + (p.y - a.y) * aby) / len2));
      const c = L.point(a.x + t * abx, a.y + t * aby);
      const d = c.distanceTo(p);
      if (d < bestD) { bestD = d; best = { latlng: map.containerPointToLatLng(c), kind: "edge" }; }
    }
    return best;
  }

  function showSnap(hit) {
    if (!hit) { if (st.snapMark) { map.removeLayer(st.snapMark); st.snapMark = null; } return; }
    const opts = { radius: hit.kind === "vertex" ? 7 : 5, color: "#fff", weight: 2,
                   fillColor: hit.kind === "vertex" ? COLOURS.human : "transparent",
                   fillOpacity: 0.9, interactive: false };
    if (!st.snapMark) st.snapMark = L.circleMarker(hit.latlng, opts).addTo(map);
    else st.snapMark.setLatLng(hit.latlng).setStyle(opts);
  }

  /* ------------------------------------------------------------------ modes */

  function setMode(mode) {
    if (mode === st.mode) mode = "view";
    cancelDraw(); clearSelection(); showSnap(null);
    if (mode !== "review") closeReview();
    st.mode = mode;
    document.body.dataset.edit = mode;
    for (const b of document.querySelectorAll("[data-tool]")) {
      b.setAttribute("aria-pressed", String(b.dataset.tool === mode));
    }
    map.doubleClickZoom[mode === "draw" ? "disable" : "enable"]();
    if (mode === "review") openReview();
    hint();
  }

  const HINTS = {
    view: "",
    draw: "Click to place vertices, snapping to nearby roads. Double-click or Enter to finish. Backspace removes the last vertex; Esc cancels.",
    select: "Click a road to reshape it. Drag vertices, drag a midpoint to add one, right-click a vertex to remove it.",
    delete: "Click a road to delete it. Ctrl+Z undoes.",
    review: "",
  };

  function hint(text) {
    const bar = $("#editbar");
    if (!bar) return;
    if (st.mode === "review" && st.sugg) return renderReview();
    const msg = text ?? HINTS[st.mode];
    if (!msg && !st.sel) { bar.hidden = true; return; }
    bar.hidden = false;
    bar.replaceChildren(el("span", { class: "note" }, msg || ""));
    if (st.sel) {
      bar.append(
        el("button", { class: "sm", onclick: saveShape }, "Save shape ⏎"),
        el("button", { class: "sm ghost", onclick: deleteSelected }, "Delete road ⌦"),
        el("button", { class: "sm ghost", onclick: () => { clearSelection(); hint(); } }, "Cancel"));
    }
  }

  /* ---------------------------------------------------------- map events */

  map.on("click", (e) => {
    if (st.mode !== "draw") return;
    const hit = snap(e.latlng);
    const ll = hit ? hit.latlng : e.latlng;
    if (!st.draw) startDraw();
    const last = st.draw.pts[st.draw.pts.length - 1];
    // the two clicks of a double-click must not become two vertices
    if (last && map.latLngToContainerPoint(last).distanceTo(map.latLngToContainerPoint(ll)) < 4) return;
    st.draw.pts.push(ll);
    st.draw.dots.push(L.circleMarker(ll, { radius: 4, color: COLOURS.human, weight: 2,
      fillColor: "#0f1114", fillOpacity: 1, interactive: false }).addTo(map));
    st.draw.line.setLatLngs(st.draw.pts);
  });

  map.on("dblclick", () => { if (st.mode === "draw") finishDraw(); });

  map.on("mousemove", (e) => {
    if (st.mode !== "draw") return;
    const hit = snap(e.latlng);
    showSnap(hit);
    if (st.draw?.pts.length) {
      st.draw.rubber.setLatLngs([st.draw.pts[st.draw.pts.length - 1], hit ? hit.latlng : e.latlng]);
    }
  });

  function featureClick(f, lyr, e) {
    if (st.mode === "draw") return;               // bubbles to the map click
    L.DomEvent.stopPropagation(e);
    if (st.mode === "select") return select(f);
    if (st.mode === "delete") return commit({ op: "delete", target: targetOf(f) },
                                            "Road deleted. Ctrl+Z to undo.");
    if (st.mode === "view") {
      const pr = f.properties || {};
      const who = pr.source === "human" ? "drawn by you" :
        `model, confidence <b>${pr.confidence_pct ?? "?"}%</b>`;
      L.popup().setLatLng(e.latlng).setContent(
        `<b>road centreline</b><br>${Number(pr.length_m || 0).toFixed(1)} m · ${who}`).openOn(map);
    }
  }

  /* ----------------------------------------------------------------- draw */

  function startDraw() {
    st.draw = {
      pts: [], dots: [],
      line: L.polyline([], { color: COLOURS.human, weight: 3, interactive: false }).addTo(map),
      rubber: L.polyline([], { color: COLOURS.human, weight: 2, dashArray: "5,5",
                               interactive: false }).addTo(map),
    };
  }

  function cancelDraw() {
    if (!st.draw) return;
    for (const lyr of [st.draw.line, st.draw.rubber, ...st.draw.dots]) map.removeLayer(lyr);
    st.draw = null;
  }

  function undoVertex() {
    if (!st.draw?.pts.length) return;
    st.draw.pts.pop();
    map.removeLayer(st.draw.dots.pop());
    st.draw.line.setLatLngs(st.draw.pts);
    st.draw.rubber.setLatLngs([]);
  }

  async function finishDraw() {
    if (!st.draw) return;
    const pts = st.draw.pts;
    cancelDraw();
    if (pts.length < 2) return;
    await commit({ op: "add", geometry: pts.map(ll => [ll.lng, ll.lat]), note: "drawn" });
  }

  /* --------------------------------------------------------------- reshape */

  function targetOf(f) {
    return { source: f.properties?.source || "model", id: f.properties?.id,
             geometry: f.geometry.coordinates };
  }

  function select(f) {
    clearSelection();
    const pts = f.geometry.coordinates.map(([x, y]) => L.latLng(y, x));
    st.sel = {
      feature: f, pts,
      halo: L.polyline(pts, { color: COLOURS.selected, weight: 8, opacity: 0.25,
                              interactive: false }).addTo(map),
      line: L.polyline(pts, { color: COLOURS.selected, weight: 2.5, interactive: false }).addTo(map),
      handles: [],
    };
    drawHandles();
    hint(`Reshaping a ${f.properties?.source === "human" ? "road you drew" : "model road"} ` +
         `(${Number(f.properties?.length_m || 0).toFixed(0)} m).`);
  }

  function drawHandles() {
    const s = st.sel;
    for (const h of s.handles) map.removeLayer(h);
    s.handles = [];
    s.pts.forEach((ll, i) => {
      const h = L.marker(ll, { draggable: true, keyboard: false,
        icon: L.divIcon({ className: "vx", iconSize: [12, 12] }) }).addTo(map);
      h.on("drag", (e) => {
        const hit = snap(e.latlng, s.feature.properties?.id);
        showSnap(hit);
        s.pts[i] = hit ? hit.latlng : e.latlng;
        if (hit) h.setLatLng(hit.latlng);
        s.line.setLatLngs(s.pts); s.halo.setLatLngs(s.pts);
      });
      h.on("dragend", () => { showSnap(null); drawHandles(); });
      h.on("contextmenu", (e) => {
        L.DomEvent.preventDefault(e.originalEvent);
        if (s.pts.length <= 2) return toast("a road needs at least two vertices", true);
        s.pts.splice(i, 1);
        s.line.setLatLngs(s.pts); s.halo.setLatLngs(s.pts);
        drawHandles();
      });
      s.handles.push(h);
    });
    // midpoints: drag one to insert a vertex there
    for (let i = 0; i + 1 < s.pts.length; i++) {
      const a = s.pts[i], b = s.pts[i + 1];
      const mid = L.latLng((a.lat + b.lat) / 2, (a.lng + b.lng) / 2);
      const h = L.marker(mid, { draggable: true, keyboard: false,
        icon: L.divIcon({ className: "vx mid", iconSize: [9, 9] }) }).addTo(map);
      let inserted = false;
      h.on("drag", (e) => {
        if (!inserted) { s.pts.splice(i + 1, 0, e.latlng); inserted = true; }
        s.pts[i + 1] = e.latlng;
        s.line.setLatLngs(s.pts); s.halo.setLatLngs(s.pts);
      });
      h.on("dragend", drawHandles);
      s.handles.push(h);
    }
  }

  function clearSelection() {
    if (!st.sel) return;
    for (const lyr of [st.sel.line, st.sel.halo, ...st.sel.handles]) map.removeLayer(lyr);
    st.sel = null;
    showSnap(null);
  }

  async function saveShape() {
    if (!st.sel) return;
    const { feature, pts } = st.sel;
    clearSelection();
    await commit({ op: "replace", target: targetOf(feature),
                   geometry: pts.map(ll => [ll.lng, ll.lat]), note: "reshaped" },
                 "Road reshaped.");
  }

  async function deleteSelected() {
    if (!st.sel) return;
    const f = st.sel.feature;
    clearSelection();
    await commit({ op: "delete", target: targetOf(f) }, "Road deleted. Ctrl+Z to undo.");
  }

  /* ------------------------------------------------------ server round trip */

  async function commit(body, message) {
    const job = getJob();
    if (!job || st.busy) return;
    st.busy = true;
    try {
      const res = await api.post(`/api/jobs/${job}/edits`, body);
      onNetwork(res.network, res.stats);
      if (message) toast(message, false, 2500);
    } catch (e) {
      toast(e.message, true);
    } finally {
      st.busy = false;
      hint();
    }
  }

  async function history(action) {
    const job = getJob();
    if (!job || st.busy) return;
    st.busy = true;
    try {
      const res = await api.post(`/api/jobs/${job}/edits/${action}`);
      onNetwork(res.network, res.stats);
      toast(`${action === "undo" ? "Undid" : "Redid"}: ${res.op.op}`, false, 1800);
    } catch (e) {
      if (!String(e.message).startsWith("409")) toast(e.message, true);
      else toast(`nothing to ${action}`, false, 1500);
    } finally { st.busy = false; }
  }

  /* ------------------------------------------------------ suggestion review */

  async function openReview() {
    const job = getJob();
    if (!job) return;
    hint("Looking for roads the model probably missed...");
    try {
      const fc = await api.get(`/api/jobs/${job}/suggestions`);
      if (st.mode !== "review") return;
      const layer = L.geoJSON(fc, {
        renderer,
        style: (f) => ({ color: COLOURS[f.properties.kind] || COLOURS.gap, weight: 3,
                         dashArray: "6,5", opacity: 0.9, renderer }),
        onEachFeature: (f, lyr) => lyr.on("click", (e) => {
          L.DomEvent.stopPropagation(e);
          st.sugg.i = st.sugg.features.indexOf(f);
          focusSuggestion();
        }),
      }).addTo(map);
      st.sugg = { features: fc.features, layer, i: 0, accepted: 0, dismissed: 0 };
      focusSuggestion();
    } catch (e) {
      hint(`No suggestions: ${e.message.replace(/^\d+ /, "")}`);
    }
  }

  function closeReview() {
    if (!st.sugg) return;
    map.removeLayer(st.sugg.layer);
    if (st.sugg.focus) map.removeLayer(st.sugg.focus);
    st.sugg = null;
  }

  function focusSuggestion() {
    const s = st.sugg;
    if (!s) return;
    if (s.focus) { map.removeLayer(s.focus); s.focus = null; }
    const f = s.features[s.i];
    if (f) {
      const lls = f.geometry.coordinates.map(([x, y]) => [y, x]);
      s.focus = L.polyline(lls, { color: "#fff", weight: 9, opacity: 0.22,
                                  interactive: false }).addTo(map);
      map.fitBounds(L.latLngBounds(lls).pad(1.5), { maxZoom: 18, animate: true });
    }
    renderReview();
  }

  function renderReview() {
    const bar = $("#editbar");
    const s = st.sugg;
    if (!bar || !s) return;
    bar.hidden = false;
    const f = s.features[s.i];
    if (!f) {
      bar.replaceChildren(el("span", { class: "note" },
        `Review done: ${s.accepted} accepted, ${s.dismissed} dismissed. `),
        el("button", { class: "sm ghost", onclick: () => setMode("view") }, "Close"));
      return;
    }
    const p = f.properties;
    const kind = p.kind === "gap"
      ? el("span", { class: "tag gap", title: "two dead ends that nearly meet; usually a real connection" }, "gap · usually right")
      : el("span", { class: "tag missed", title: "the model was unsure here; about half of these are real roads" }, "possible road · check it");
    bar.replaceChildren(
      el("span", { class: "note" }, `Suggestion ${s.i + 1} of ${s.features.length}`),
      kind,
      el("span", { class: "note mono" }, `${p.length_m.toFixed(0)} m · conf ${p.confidence_pct}%`),
      el("button", { class: "sm", onclick: acceptSuggestion }, "Accept A"),
      el("button", { class: "sm ghost", onclick: dismissSuggestion }, "Dismiss X"),
      el("button", { class: "sm ghost", onclick: () => step(1) }, "Skip N"));
  }

  function step(d) {
    const s = st.sugg;
    if (!s) return;
    s.i = Math.max(0, Math.min(s.features.length, s.i + d));
    focusSuggestion();
  }

  function dropCurrent() {
    const s = st.sugg;
    const f = s.features[s.i];
    s.features.splice(s.i, 1);
    s.layer.eachLayer(l => { if (l.feature === f) s.layer.removeLayer(l); });
    return f;
  }

  async function acceptSuggestion() {
    const s = st.sugg;
    if (!s?.features[s.i]) return;
    const f = dropCurrent();
    s.accepted++;
    await commit({ op: "add", geometry: f.geometry.coordinates,
                   note: `accepted suggestion (${f.properties.kind})` });
    focusSuggestion();
  }

  async function dismissSuggestion() {
    const s = st.sugg;
    if (!s?.features[s.i]) return;
    const f = dropCurrent();
    s.dismissed++;
    await commit({ op: "dismiss", geometry: f.geometry.coordinates,
                   note: `dismissed suggestion (${f.properties.kind})` });
    focusSuggestion();
  }

  /* -------------------------------------------------------------- keyboard */

  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "textarea" || tag === "input" || tag === "select") return;
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); return history(e.shiftKey ? "redo" : "undo"); }
    if ((e.ctrlKey || e.metaKey) && k === "y") { e.preventDefault(); return history("redo"); }
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    if (st.mode === "review" && st.sugg) {
      if (k === "a") return acceptSuggestion();
      if (k === "x") return dismissSuggestion();
      if (k === "n" || k === "arrowright") return step(1);
      if (k === "p" || k === "arrowleft") return step(-1);
    }
    if (k === "escape") {
      if (st.draw) return cancelDraw();
      if (st.sel) { clearSelection(); return hint(); }
      return setMode("view");
    }
    if (k === "enter") {
      if (st.draw) return finishDraw();
      if (st.sel) return saveShape();
    }
    if (k === "backspace" && st.draw) { e.preventDefault(); return undoVertex(); }
    if ((k === "delete" || k === "backspace") && st.sel) { e.preventDefault(); return deleteSelected(); }
    const tools = { d: "draw", v: "select", x: "delete", r: "review" };
    if (tools[k] && !(st.mode === "review" && st.sugg)) return setMode(tools[k]);
  });

  return {
    render, setMode, history,
    get mode() { return st.mode; },
    reset() { setMode("view"); },
  };
}
