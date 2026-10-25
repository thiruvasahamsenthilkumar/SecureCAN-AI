// Dashboard for SecureCAN AI. Renders the output of scripts/live_demo.py, which
// bakes its results into data.js. No synthetic numbers here -- if data.js is
// absent the page says so rather than inventing a feed.

const D = window.SECURECAN || null;
const CLASS_COLORS = {
  NORMAL: "#00ff41", DOS: "#ff0000", FUZZY: "#ffa500",
  FAKE_RPM: "#ff006e", FAKE_GEAR: "#00d4ff",
};

function setText(id, v) {
  const el = document.getElementById(id);
  if (el) el.textContent = v;
}

function noData() {
  document.querySelectorAll(".stat-value").forEach((e) => (e.textContent = "--"));
  const c = document.getElementById("alertsContainer");
  if (c) {
    c.innerHTML = `<div class="alert-placeholder">
      <p><strong>No run data found.</strong></p>
      <p>Generate it with <code>python scripts/live_demo.py</code></p>
    </div>`;
  }
  const dot = document.querySelector(".status-dot");
  const txt = document.querySelector(".status-text");
  if (dot) dot.style.background = "#888";
  if (txt) { txt.textContent = "No Data"; txt.style.color = "#888"; }
}

function renderStats() {
  const s = D.summary;
  setText("totalMessages", s.frames_replayed.toLocaleString());
  setText("attackCount", s.alerts_raised.toLocaleString());
  const txt = document.querySelector(".status-text");
  if (txt) txt.textContent = `Replay ${s.throughput_fps.toFixed(0)} f/s`;
}

// Predicted class over the replay, with ground truth dashed beneath it.
function renderTimeline() {
  const el = document.getElementById("detectionChart");
  if (!el || !window.Chart) return;

  const labels = Object.keys(CLASS_COLORS);
  const toY = (name) => labels.indexOf(name);
  const pts = D.detections;

  new Chart(el, {
    type: "line",
    data: {
      labels: pts.map((p) => p.bus_time.toFixed(1)),
      datasets: [
        {
          label: "predicted",
          data: pts.map((p) => toY(p.predicted)),
          borderColor: "#00d4ff",
          backgroundColor: "rgba(0,212,255,.12)",
          stepped: true, borderWidth: 2, pointRadius: 0, fill: true,
        },
        {
          label: "actual",
          data: pts.map((p) => toY(p.actual)),
          borderColor: "rgba(255,255,255,.4)",
          stepped: true, borderWidth: 1, pointRadius: 0,
          borderDash: [4, 3], fill: false,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        y: {
          min: -0.5, max: labels.length - 0.5,
          ticks: { callback: (v) => labels[v] || "", color: "#888", font: { size: 10 } },
          grid: { color: "rgba(255,255,255,.05)" },
        },
        x: {
          title: { display: true, text: "bus time (s)", color: "#888" },
          ticks: { color: "#888", maxTicksLimit: 12 },
          grid: { display: false },
        },
      },
      plugins: { legend: { labels: { color: "#e0e0e0", boxWidth: 12 } } },
    },
  });
}

function renderPerClass() {
  const el = document.getElementById("attackTypeChart");
  if (!el || !window.Chart) return;

  const pc = D.summary.per_class;
  const names = Object.keys(pc);

  new Chart(el, {
    type: "bar",
    data: {
      labels: names,
      datasets: [{
        label: "recall %",
        data: names.map((n) => pc[n].recall_pct),
        backgroundColor: names.map((n) => (CLASS_COLORS[n] || "#888") + "cc"),
        borderColor: names.map((n) => CLASS_COLORS[n] || "#888"),
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 100, ticks: { color: "#888" },
             grid: { color: "rgba(255,255,255,.05)" } },
        x: { ticks: { color: "#888", font: { size: 10 } }, grid: { display: false } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { afterLabel: (c) => `${pc[names[c.dataIndex]].windows} windows` } },
      },
    },
  });
}

// Light up an ECU when a class implicating it was flagged.
const ECU_FOR_CLASS = { FAKE_RPM: "ecu1", FAKE_GEAR: "ecu2", FUZZY: "ecu3", DOS: "ecu4" };

function renderNetwork() {
  const hit = new Set(
    D.detections.filter((d) => d.predicted !== "NORMAL").map((d) => ECU_FOR_CLASS[d.predicted])
  );
  Object.values(ECU_FOR_CLASS).forEach((id) => {
    const node = document.getElementById(id);
    const badge = node && node.querySelector(".node-status");
    if (!badge) return;
    badge.textContent = hit.has(id) ? "Flagged" : "Clean";
    badge.className = "node-status " + (hit.has(id) ? "alert" : "safe");
  });
}

function renderAlerts() {
  const box = document.getElementById("alertsContainer");
  if (!box) return;

  const alerts = D.detections.filter((d) => d.predicted !== "NORMAL");
  if (!alerts.length) {
    box.innerHTML = `<div class="alert-placeholder"><p>No attacks flagged in this run.</p></div>`;
    return;
  }

  box.innerHTML = "";
  alerts.slice(0, 120).forEach((a) => {
    const div = document.createElement("div");
    div.className = "alert-item" + (a.correct ? " danger" : "");
    div.innerHTML = `
      <div class="alert-time">bus t=${a.bus_time.toFixed(2)}s &middot; frame ${a.frame}</div>
      <div class="alert-message">${a.predicted}${
        a.correct ? "" : ` <span class="tag-sim">actual ${a.actual}</span>`}</div>
      <div class="alert-details">
        <span>confidence ${(a.confidence * 100).toFixed(0)}%</span>
        <span>${a.msg_rate.toFixed(0)} msg/s</span>
        <span>latency ${a.latency_ms.toFixed(1)} ms</span>
        <span>byte entropy ${a.byte_entropy.toFixed(2)}</span>
      </div>`;
    box.appendChild(div);
  });

  if (alerts.length > 120) {
    const n = document.createElement("p");
    n.className = "metrics-note";
    n.textContent = `Showing 120 of ${alerts.length} alerts.`;
    box.appendChild(n);
  }
}

function renderFooter() {
  const s = D.summary;
  const f = document.querySelector(".footer p");
  if (!f) return;
  const cpp = D.cpp
    ? ` &middot; C++/sklearn ${D.cpp.end_to_end_agreement_pct}% @ ${D.cpp.host_inference_ns} ns`
    : "";
  f.innerHTML = `Replay ${s.frames_replayed.toLocaleString()} frames &middot; `
    + `${s.windows_classified} windows &middot; accuracy ${s.accuracy_pct}% &middot; `
    + `false positives ${s.false_positive_rate_pct}% &middot; `
    + `held-out ${D.model.heldout_accuracy_pct}%${cpp} &middot; ${D.generated}`;
}

function clearAlerts() {
  const box = document.getElementById("alertsContainer");
  if (box) box.innerHTML = `<div class="alert-placeholder"><p>Cleared &mdash; reload to restore.</p></div>`;
}

document.addEventListener("DOMContentLoaded", () => {
  if (!D) return noData();
  renderStats();
  renderTimeline();
  renderPerClass();
  renderNetwork();
  renderAlerts();
  renderFooter();
});
