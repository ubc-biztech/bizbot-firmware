// WebSocket <-> keyboard_controls.py bridge. The Python tool owns the robot
// connection (it ENABLEs on connect and keeps the STOP keepalive going);
// this just spawns it, pipes command lines into its stdin, and forwards its
// stdout lines to every browser tab. Closing the last tab kills it, which
// makes the Python tool send DISABLE.
import { spawn, execSync } from "node:child_process";
import path from "node:path";
import { WebSocketServer } from "ws";

const ROOT = path.resolve(import.meta.dirname, "../..");
const PYTHON = process.env.BIZBOT_PYTHON || path.join(ROOT, ".venv/bin/python");
const SCRIPT = path.join(ROOT, "tools/keyboard_controls.py");

export function attachBridge(httpServer, { host, port, log = console.log }) {
  const wss = new WebSocketServer({ noServer: true });
  const clients = new Set();
  let child = null;
  let rx = "";

  const broadcast = (msg) => {
    const text = JSON.stringify(msg);
    for (const c of clients) if (c.readyState === c.OPEN) c.send(text);
  };
  const status = () =>
    broadcast({ type: "status", connected: !!child, alive: !!child, host, port });

  function start() {
    if (child) return;
    killStrays();
    // -u: unbuffered stdout, otherwise Python holds replies in an 8 KB buffer.
    const args = ["-u", SCRIPT, "--transport", "wifi", "--host", host, "--port", String(port)];
    log(`[bridge] spawning ${PYTHON} ${args.join(" ")}`);
    child = spawn(PYTHON, args, { cwd: ROOT, stdio: ["pipe", "pipe", "pipe"] });
    const onData = (chunk) => {
      rx += chunk.toString("utf8");
      let i;
      while ((i = rx.indexOf("\n")) >= 0) {
        let line = rx.slice(0, i).replace(/\r$/, "");
        rx = rx.slice(i + 1);
        if (line.startsWith("< ")) line = line.slice(2);
        else if (line.startsWith("> ")) continue; // our own echo
        else if (line) { broadcast({ type: "log", text: line }); continue; }
        if (line) broadcast({ type: "line", line, t: Date.now() });
      }
    };
    child.stdout.on("data", onData);
    child.stderr.on("data", (c) => broadcast({ type: "log", text: c.toString("utf8").trim() }));
    const me = child;
    child.on("exit", (code) => {
      log(`[bridge] keyboard_controls.py exited (${code})`);
      if (child !== me) return; // an old, already-replaced child; ignore it
      child = null;
      status();
      if (clients.size > 0) setTimeout(start, 2000);
    });
    status();
  }
  function stop() {
    if (!child) return;
    child.kill("SIGINT"); // its finally: sends DISABLE, then disconnects
    child = null;
  }
  // Singleton: kill any keyboard_controls.py left over from a previous bridge
  // process before spawning ours. Only one may ever talk to the robot.
  function killStrays() {
    try { execSync(`pkill -INT -f "${SCRIPT}"`, { stdio: "ignore" }); } catch {}
  }

  httpServer.on("close", () => { stop(); for (const c of clients) c.terminate(); });

  wss.on("connection", (ws) => {
    clients.add(ws);
    start();
    ws.send(JSON.stringify({ type: "status", connected: !!child, alive: !!child, host, port }));
    ws.on("message", (data) => {
      let msg;
      try { msg = JSON.parse(data.toString()); } catch { return; }
      if (typeof msg.cmd !== "string") return;
      const cmd = msg.cmd.trim();
      if (!/^[A-Z_]+( [-0-9.e]+)*$/.test(cmd)) return;
      if (child) child.stdin.write(cmd + "\n");
      else ws.send(JSON.stringify({ type: "log", text: "python tool not running" }));
    });
    ws.on("close", () => {
      clients.delete(ws);
      if (clients.size === 0) { stop(); log("[bridge] last tab closed; stopped python"); }
    });
  });

  httpServer.on("upgrade", (req, socket, head) => {
    if (req.url !== "/bridge") return;
    wss.handleUpgrade(req, socket, head, (ws) => wss.emit("connection", ws, req));
  });
}
