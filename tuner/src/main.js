// BizBot tuner: buttons for gains, live readouts, a pitch trace, and a log.
// Talks to the bridge in vite.config.js over a WebSocket at /bridge.

const $ = (id) => document.getElementById(id);

// --- state -----------------------------------------------------------------
// Gains live in this browser (localStorage) and are pushed TO the robot
// whenever it (re)connects. The robot's own values are never read back.
const GAINS_KEY = "bizbot-gains";
const gains = { kp: 0.04, ki: 0, kd: 0.001, trim: 0, vkp: 0, vki: 0 };
try { Object.assign(gains, JSON.parse(localStorage.getItem(GAINS_KEY)) || {}); } catch {}
function persistGains() { try { localStorage.setItem(GAINS_KEY, JSON.stringify(gains)); } catch {} }
const live = {}; // last key=value telemetry
let haveGains = true;
let pushedToRobot = false;
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
    if (!msg.connected) pushedToRobot = false; // push again on the next connect
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
    renderReadouts();
    if ($("show-telemetry").checked) logLine("< " + line);
    return;
  }

  logLine("< " + line);
  if (line.startsWith("FAULT")) showFault(line);
  if (line.startsWith("ERR")) showFault(line, true);
  if (line.startsWith("OK ENABLED")) hideFault();
  if (line === "OK SET_VEL_SIGN" && pendingVelSign !== null) {
    velSign = pendingVelSign;
    pendingVelSign = null;
    renderVsign();
  }
}


function parseKv(line) {
  for (const tok of line.split(/\s+/).slice(1)) {
    const i = tok.indexOf("=");
    if (i > 0) live[tok.slice(0, i)] = Number(tok.slice(i + 1));
  }
  if (line.startsWith("STATE ") && "enabled" in live) setEnabled(!!live.enabled);
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
  // Same as keyboard_controls.py: just send ENABLE, don't wait for STATE.
  sw.classList.add("pending");
  send(enabledKnown === true ? "DISABLE" : "ENABLE");
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
  if (!robotStatus.connected) setUnknown();
}
setInterval(renderRobotPill, 500);

// First telemetry after a (re)connect: overwrite the robot with local gains.
function pushGainsToRobot() {
  pushedToRobot = true;
  sendTrim();
  logLine("! pushed local gains to robot");
}

// --- gains UI ---------------------------------------------------------------
const rows = {
  "pid-gains": [
    { key: "kp", steps: [0.001, 0.005, 0.01, 0.02], def: 0.01, min: 0, max: 5, send: sendPid },
    { key: "ki", steps: [0.0005, 0.001, 0.005], def: 0.001, min: 0, max: 5, send: sendPid },
    { key: "kd", steps: [0.0001, 0.0005, 0.001, 0.002], def: 0.0005, min: 0, max: 5, send: sendPid },
  ],
  "vel-gains": [
    { key: "vkp", steps: [0.01, 0.02, 0.05, 0.1], def: 0.02, min: 0, max: 1, send: sendVel },
    { key: "vki", steps: [0.01, 0.02, 0.05, 0.1], def: 0.02, min: 0, max: 1, send: sendVel },
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
  persistGains();
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
function sendVel() {
  send(`SET_VEL ${fmt(gains.vkp)} ${fmt(gains.vki)}`);
}
function sendTrim() {
  send(`SET_TRIM ${fmt(gains.trim)}`);
}

// --- readouts ----------------------------------------------------------------
const READOUTS = [
  ["pitch", "pitch °"],
  ["pitch_rate", "rate °/s"],
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
// Firmware boots with VEL_SIGN -1 and STATE does not report it, so track it here.
let velSign = -1;
let pendingVelSign = null;
function renderVsign() { $("btn-vsign").textContent = `Sign ${velSign > 0 ? "+1" : "−1"} (flip)`; }
$("btn-vsign").onclick = () => { pendingVelSign = -velSign; send(`SET_VEL_SIGN ${pendingVelSign}`); };
renderVsign();
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
      persistGains();
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


$("btn-state").onclick = () => send("GET_STATE");
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") send("DISABLE");
});

buildGains();
renderReadouts();
connect();
