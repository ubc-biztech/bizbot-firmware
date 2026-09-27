// BizBot tuner: buttons for gains, live readouts, a pitch trace, and a log.
// Talks to the bridge in vite.config.js over a WebSocket at /bridge.

const $ = (id) => document.getElementById(id);

// --- state -----------------------------------------------------------------
const gains = { kp: 0.04, ki: 0, kd: 0.001, trim: 0 };
const live = {}; // last key=value telemetry
let haveGains = false;
let lastSentAt = 0;
const SCOPE_SECONDS = 15;
const samples = []; // {t, pitch, target}

// --- websocket --------------------------------------------------------------
let ws;
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/bridge`);
  ws.onopen = () => setPill("pill-bridge", "bridge", true);
  ws.onclose = () => {
    setPill("pill-bridge", "bridge", false);
    setPill("pill-robot", "robot", false);
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => handle(JSON.parse(ev.data));
}
function send(cmd) {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ cmd }));
    lastSentAt = Date.now();
    logLine("> " + cmd);
  }
}

// --- incoming ---------------------------------------------------------------
let lastRobotLineAt = 0;
function handle(msg) {
  if (msg.type === "status") {
    robotStatus = msg;
    renderRobotPill();
    return;
  }
  if (msg.type === "log") {
    logLine("! " + msg.text);
    return;
  }
  if (msg.type !== "line") return;
  lastRobotLineAt = Date.now();
  const line = msg.line;
  if (line === "OK STOP") return;
  if (line.startsWith("OK ENABLED")) setEnabled(true);
  if (line.startsWith("OK DISABLED") || line.startsWith("FAULT")) setEnabled(false);

  const isTelemetry = line.startsWith("PID ") || line.startsWith("STATE ");
  if (isTelemetry) {
    parseKv(line);
    if (line.startsWith("STATE ")) applyStateGains();
    pushSample(msg.t);
    renderReadouts();
    if ($("show-telemetry").checked) logLine("< " + line);
    return;
  }

  logLine("< " + line);
  if (line.startsWith("FAULT")) showFault(line);
  if (line.startsWith("ERR")) showFault(line, true);
  if (line.startsWith("OK ENABLED")) hideFault();
}


function parseKv(line) {
  for (const tok of line.split(/\s+/).slice(1)) {
    const i = tok.indexOf("=");
    if (i > 0) live[tok.slice(0, i)] = Number(tok.slice(i + 1));
  }
  if ("enabled" in live) setEnabled(!!live.enabled);
}

// --- balancing switch --------------------------------------------------------
// The switch shows what the ROBOT says, refreshed from every STATE line. A
// click sends the command and shows "pending" until the robot confirms.
let robotStatus = { connected: false, alive: false };
let enabledKnown = null;
const sw = $("switch");
function setEnabled(on) {
  enabledKnown = on;
  sw.classList.remove("pending", "unknown");
  sw.classList.toggle("on", on);
  sw.classList.toggle("off", !on);
  sw.setAttribute("aria-pressed", String(on));
  sw.querySelector(".label").textContent = on ? "BALANCING ON" : "BALANCING OFF";
}
function setUnknown() {
  enabledKnown = null;
  sw.classList.remove("on", "pending");
  sw.classList.add("off", "unknown");
  sw.querySelector(".label").textContent = "NO ROBOT";
}
sw.onclick = () => {
  if (enabledKnown === null) return send("GET_STATE");
  sw.classList.add("pending");
  send(enabledKnown ? "DISABLE" : "ENABLE");
};
function renderRobotPill() {
  const el = $("pill-robot");
  const responsive = robotStatus.alive && Date.now() - lastRobotLineAt < 2000;
  el.classList.toggle("on", responsive);
  el.classList.toggle("bad", robotStatus.connected && !responsive);
  el.textContent = !robotStatus.connected
    ? "robot: connecting"
    : responsive
      ? `robot ${robotStatus.host}`
      : "robot: no response";
  if (!responsive) setUnknown();
}
setInterval(renderRobotPill, 500);

// The robot's gains are the source of truth, but don't fight the user
// while they are clicking: only refresh inputs when nothing was sent recently.
function applyStateGains() {
  if (Date.now() - lastSentAt < 1500 && haveGains) return;
  for (const k of ["kp", "ki", "kd", "trim"]) {
    if (k in live) gains[k] = live[k];
  }
  haveGains = true;
  renderGains();
}

// --- gains UI ---------------------------------------------------------------
const rows = {
  "pid-gains": [
    { key: "kp", steps: [0.001, 0.005, 0.01, 0.02], def: 0.01, min: 0, max: 5, send: sendPid },
    { key: "ki", steps: [0.0005, 0.001, 0.005], def: 0.001, min: 0, max: 5, send: sendPid },
    { key: "kd", steps: [0.0001, 0.0005, 0.001, 0.002], def: 0.0005, min: 0, max: 5, send: sendPid },
  ],
  "trim-gains": [
    { key: "trim", steps: [0.1, 0.25, 0.5, 1], def: 0.25, min: -15, max: 15, send: sendTrim },
  ],
};
const inputs = {};

function buildGains() {
  for (const [container, list] of Object.entries(rows)) {
    const box = $(container);
    for (const row of list) {
      const el = document.createElement("div");
      el.className = "gain";
      el.innerHTML = `
        <label>${row.key}</label>
        <div class="ctl">
          <button class="step" data-dir="-1">−</button>
          <input type="number" step="any" />
          <button class="step" data-dir="1">+</button>
        </div>
        <select title="step size">${row.steps
          .map((s) => `<option value="${s}" ${s === row.def ? "selected" : ""}>±${s}</option>`)
          .join("")}</select>`;
      box.appendChild(el);
      const input = el.querySelector("input");
      const select = el.querySelector("select");
      inputs[row.key] = input;
      for (const b of el.querySelectorAll("button.step")) {
        b.onclick = () => {
          const step = Number(select.value) * Number(b.dataset.dir);
          setGain(row, gains[row.key] + step);
        };
      }
      input.onchange = () => setGain(row, Number(input.value));
      input.onkeydown = (e) => {
        if (e.key === "Enter") input.blur();
      };
    }
  }
  renderGains();
}

function setGain(row, value) {
  if (!Number.isFinite(value)) return renderGains();
  value = Math.min(row.max, Math.max(row.min, value));
  gains[row.key] = Number(value.toFixed(5));
  renderGains();
  row.send();
}
function renderGains() {
  for (const [k, input] of Object.entries(inputs)) {
    if (document.activeElement !== input) input.value = fmt(gains[k]);
  }
}
const fmt = (v) => (Number.isFinite(v) ? Number(v.toFixed(5)).toString() : "");

function sendPid() {
  send(`SET_PID ${fmt(gains.kp)} ${fmt(gains.ki)} ${fmt(gains.kd)}`);
}
function sendTrim() {
  send(`SET_TRIM ${fmt(gains.trim)}`);
}

// --- readouts ----------------------------------------------------------------
const READOUTS = [
  ["pitch", "pitch °"],
  ["target", "target °"],
  ["pitch_rate", "rate °/s"],
  ["out", "motor out"],
  ["cmd", "hover cmd"],
  ["trim", "trim °"],
  ["wheel_l", "wheel L"],
  ["wheel_r", "wheel R"],
  ["battery", "battery V"],
];
function renderReadouts() {
  const box = $("readouts");
  if (!box.children.length) {
    for (const [k, label] of READOUTS) {
      const d = document.createElement("div");
      d.className = "ro";
      d.innerHTML = `<div class="k">${label}</div><div class="v" id="ro-${k}">–</div>`;
      box.appendChild(d);
    }
  }
  for (const [k] of READOUTS) {
    const v = live[k];
    $("ro-" + k).textContent = Number.isFinite(v) ? Number(v.toFixed(3)).toString() : "–";
  }
}

// --- scope -------------------------------------------------------------------
function pushSample(t) {
  if (!Number.isFinite(live.pitch)) return;
  samples.push({ t, pitch: live.pitch, target: Number.isFinite(live.target) ? live.target : 0 });
  const cutoff = t - SCOPE_SECONDS * 1000;
  while (samples.length && samples[0].t < cutoff) samples.shift();
}

const canvas = $("scope");
const ctx = canvas.getContext("2d");
let hoverX = null;
canvas.onmousemove = (e) => (hoverX = e.offsetX);
canvas.onmouseleave = () => (hoverX = null);

function drawScope() {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const now = Date.now();
  const t0 = now - SCOPE_SECONDS * 1000;
  const pad = { l: 36, r: 8, t: 8, b: 18 };
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;

  // Symmetric range around zero so the balance point reads as the centre line.
  let range = 2;
  for (const s of samples) range = Math.max(range, Math.abs(s.pitch), Math.abs(s.target));
  range = Math.ceil(range * 1.1);
  const x = (t) => pad.l + ((t - t0) / (SCOPE_SECONDS * 1000)) * pw;
  const y = (v) => pad.t + ph / 2 - (v / range) * (ph / 2);

  // grid: centre line plus range ticks, kept recessive
  ctx.strokeStyle = "#3a3a38";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#8a8a84";
  ctx.font = "11px system-ui";
  ctx.textAlign = "right";
  for (const v of [-range, -range / 2, 0, range / 2, range]) {
    ctx.beginPath();
    ctx.moveTo(pad.l, y(v));
    ctx.lineTo(w - pad.r, y(v));
    ctx.stroke();
    ctx.fillText(v.toFixed(0) + "°", pad.l - 6, y(v) + 4);
  }
  ctx.textAlign = "center";
  for (let s = 0; s <= SCOPE_SECONDS; s += 5) {
    ctx.fillText(`-${SCOPE_SECONDS - s}s`, x(t0 + s * 1000), h - 4);
  }

  const drawSeries = (key, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    let first = true;
    for (const s of samples) {
      const px = x(s.t);
      const py = y(s[key]);
      if (first) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
      first = false;
    }
    ctx.stroke();
  };
  drawSeries("target", "#d95926");
  drawSeries("pitch", "#3987e5");

  // hover crosshair + readout
  const readout = $("hover-readout");
  if (hoverX !== null && samples.length) {
    const tHover = t0 + ((hoverX - pad.l) / pw) * SCOPE_SECONDS * 1000;
    let best = samples[0];
    for (const s of samples) if (Math.abs(s.t - tHover) < Math.abs(best.t - tHover)) best = s;
    ctx.strokeStyle = "#c3c2b7";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x(best.t), pad.t);
    ctx.lineTo(x(best.t), pad.t + ph);
    ctx.stroke();
    for (const [k, c] of [["pitch", "#3987e5"], ["target", "#d95926"]]) {
      ctx.fillStyle = c;
      ctx.beginPath();
      ctx.arc(x(best.t), y(best[k]), 4, 0, Math.PI * 2);
      ctx.fill();
    }
    readout.textContent = `pitch ${best.pitch.toFixed(2)}°  target ${best.target.toFixed(2)}°  (${((best.t - now) / 1000).toFixed(1)} s)`;
  } else {
    readout.textContent = "";
  }
  requestAnimationFrame(drawScope);
}

// --- log / faults ------------------------------------------------------------
const logEl = $("log");
const logLines = [];
function logLine(text) {
  const ts = new Date().toLocaleTimeString([], { hour12: false });
  logLines.push(`${ts} ${text}`);
  if (logLines.length > 200) logLines.shift();
  logEl.textContent = logLines.join("\n");
  logEl.scrollTop = logEl.scrollHeight;
}
function showFault(text, soft = false) {
  const el = $("fault");
  el.textContent = text;
  el.hidden = false;
  if (soft) setTimeout(hideFault, 4000);
}
function hideFault() {
  $("fault").hidden = true;
}

function setPill(id, text, on, cls = "on") {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle(cls, on);
}

// --- wiring ------------------------------------------------------------------
$("btn-zero").onclick = () => send("ZERO_IMU");
// --- presets (localStorage) ---------------------------------------------------
const PRESET_KEY = "bizbot-presets";
function loadPresets() {
  try { return JSON.parse(localStorage.getItem(PRESET_KEY)) || {}; } catch { return {}; }
}
function savePresets(p) {
  try { localStorage.setItem(PRESET_KEY, JSON.stringify(p)); } catch {}
}
function renderPresets() {
  const box = $("presets");
  box.innerHTML = "";
  const presets = loadPresets();
  for (const [name, g] of Object.entries(presets)) {
    const el = document.createElement("span");
    el.className = "preset";
    el.innerHTML = `<button title="kp ${g.kp} ki ${g.ki} kd ${g.kd}"></button><button class="x" title="delete">×</button>`;
    el.firstChild.textContent = name;
    el.firstChild.onclick = () => {
      Object.assign(gains, { kp: g.kp, ki: g.ki, kd: g.kd });
      renderGains();
      sendPid();
    };
    el.lastChild.onclick = () => { delete presets[name]; savePresets(presets); renderPresets(); };
    box.appendChild(el);
  }
  if (!Object.keys(presets).length) box.textContent = "none yet";
}
$("btn-save-preset").onclick = () => {
  const name = $("preset-name").value.trim() || `set ${new Date().toLocaleTimeString([], { hour12: false })}`;
  const presets = loadPresets();
  presets[name] = { kp: gains.kp, ki: gains.ki, kd: gains.kd };
  savePresets(presets);
  $("preset-name").value = "";
  renderPresets();
};
renderPresets();

// --- CSV export of the visible trace -----------------------------------------
$("btn-csv").onclick = () => {
  const rows = ["t_ms,pitch_deg,target_deg", ...samples.map((s) => `${s.t},${s.pitch},${s.target}`)];
  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `trace-${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
};

$("btn-state").onclick = () => send("GET_STATE");
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") send("DISABLE");
});

buildGains();
renderReadouts();
connect();
requestAnimationFrame(drawScope);
