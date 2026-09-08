// Shared helpers for the gis-agent UI. No build step, no framework.

export const api = {
  async get(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 300)}`);
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });
    if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 300)}`);
    return r.json();
  },
};

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined) continue;
    node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return node;
}

export function toast(message, bad = false, ms = 5000) {
  const node = el("div", { class: `toast${bad ? " bad" : ""}` }, message);
  document.body.append(node);
  setTimeout(() => node.remove(), ms);
}

export function qs(name) {
  return new URLSearchParams(location.search).get(name);
}

export function fmt(n, digits = 3) {
  if (n === null || n === undefined || Number.isNaN(n)) return "--";
  return typeof n === "number" ? n.toFixed(digits) : String(n);
}

/** Remember the selected job across pages so navigation keeps context. */
export const currentJob = {
  get() {
    return qs("job") || localStorage.getItem("gisagent.job") || "";
  },
  set(id) {
    if (id) localStorage.setItem("gisagent.job", id);
  },
};

export function navLink(id) {
  return id ? `?job=${encodeURIComponent(id)}` : "";
}

/** Metric tiles, used on both the run and review pages. */
export function metricTiles(metrics) {
  if (!metrics) return el("p", { class: "empty" }, "not evaluated yet");
  const pick = [
    ["IoU", metrics.iou],
    ["F1", metrics.f1],
    ["precision", metrics.precision],
    ["recall", metrics.recall],
    ["relaxed F1", metrics.relaxed_f1],
  ];
  return el(
    "div",
    { class: "metrics" },
    pick.map(([k, v]) =>
      el("div", { class: "metric" },
        el("div", { class: "v" }, fmt(v)),
        el("div", { class: "k" }, k))
    )
  );
}

export function healthBadges(h) {
  const badge = (ok, label) =>
    el("span", { class: `pill ${ok ? "ok" : "no"}` },
      el("span", { class: "dot" }), label);
  return [
    badge(h.cuda, h.gpu ? `GPU ${h.gpu.replace(/NVIDIA GeForce /, "")}` : "no GPU"),
    badge(h.qgis, "QGIS"),
    badge(h.sam_token, "SAM 3"),
    badge(h.llm_configured, h.llm_model?.split("/").pop() || "LLM"),
  ];
}
