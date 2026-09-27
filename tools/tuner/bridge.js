// TCP <-> WebSocket bridge to the ESP32's line protocol (port 3333).
//
// A browser cannot open a raw TCP socket, so the Vite dev server hosts this
// bridge at ws://<vite>/bridge. It keeps exactly one TCP connection to the
// robot (the firmware accepts a single controller), forwards every line to
// all browser tabs, and sends the STOP keepalive the firmware needs while
// enabled. When the last browser tab goes away it sends DISABLE, so closing
// the page always leaves the robot stopped.
import net from "node:net";
import { WebSocketServer } from "ws";

const KEEPALIVE_MS = 200;
const STATE_POLL_MS = 500;
const RECONNECT_MS = 2000;
// The firmware serves one client; extra connections are accepted by the
// ESP32's TCP stack but never read. If the robot stays silent this long
// after we started polling it, drop the socket and try again.
const SILENT_MS = 3000;
// A connect attempt that gets no SYN-ACK (robot rebooting, Wi-Fi hop) would
// otherwise hang for the OS default of over a minute.
const CONNECT_TIMEOUT_MS = 4000;

export function attachBridge(httpServer, { host, port, log = console.log }) {
  const wss = new WebSocketServer({ noServer: true });
  const clients = new Set();
  const timers = [];
  let closed = false;
  let sock = null;
  let connected = false;
  let rx = "";
  let lastRxAt = 0;
  let alive = false;
  let reconnectTimer = null;

  const broadcast = (msg) => {
    const text = JSON.stringify(msg);
    for (const c of clients) if (c.readyState === c.OPEN) c.send(text);
  };

  const write = (cmd) => {
    if (!connected) return false;
    sock.write(cmd + "\n");
    return true;
  };

  function connect() {
    sock = net.createConnection({ host, port });
    sock.setNoDelay(true);
    sock.setTimeout(CONNECT_TIMEOUT_MS);
    sock.on("timeout", () => {
      if (!connected) {
        log(`[bridge] connect to ${host}:${port} timed out; retrying`);
        sock.destroy();
      }
    });
    sock.on("connect", () => {
      sock.setTimeout(0); // silence detection takes over once connected
      connected = true;
      alive = false;
      rx = "";
      lastRxAt = Date.now();
      log(`[bridge] connected to ${host}:${port}`);
      broadcast({ type: "status", connected: true, alive, host, port });
      write("GET_STATE");
    });
    sock.on("data", (chunk) => {
      lastRxAt = Date.now();
      if (!alive) {
        alive = true;
        broadcast({ type: "status", connected: true, alive, host, port });
      }
      rx += chunk.toString("utf8");
      let i;
      while ((i = rx.indexOf("\n")) >= 0) {
        const line = rx.slice(0, i).replace(/\r$/, "");
        rx = rx.slice(i + 1);
        if (line) broadcast({ type: "line", line, t: Date.now() });
      }
    });
    sock.on("error", (err) => {
      broadcast({ type: "log", text: `tcp: ${err.message}` });
    });
    sock.on("close", () => {
      if (connected) log(`[bridge] disconnected from ${host}:${port}`);
      connected = false;
      alive = false;
      broadcast({ type: "status", connected: false, alive, host, port });
      if (!closed) reconnectTimer = setTimeout(connect, RECONNECT_MS);
    });
  }
  connect();

  timers.push(
    setInterval(() => {
      if (connected && Date.now() - lastRxAt > SILENT_MS) {
        log(`[bridge] robot silent for ${SILENT_MS} ms; reconnecting`);
        sock.destroy();
      }
    }, 500),
  );

  timers.push(
    setInterval(() => {
      if (clients.size > 0) write("STOP");
    }, KEEPALIVE_MS),
    setInterval(() => {
      if (clients.size > 0) write("GET_STATE");
    }, STATE_POLL_MS),
  );

  // Vite restarts the dev server in-process when a config dependency (this
  // file) changes. Tear everything down so the old bridge does not keep
  // fighting the new one for the robot's single client slot.
  httpServer.on("close", () => {
    closed = true;
    for (const t of timers) clearInterval(t);
    if (reconnectTimer) clearTimeout(reconnectTimer);
    write("DISABLE");
    sock?.destroy();
    for (const c of clients) c.terminate();
    log("[bridge] closed");
  });

  wss.on("connection", (ws) => {
    clients.add(ws);
    ws.send(JSON.stringify({ type: "status", connected, alive, host, port }));
    ws.on("message", (data) => {
      let msg;
      try {
        msg = JSON.parse(data.toString());
      } catch {
        return;
      }
      if (typeof msg.cmd !== "string") return;
      const cmd = msg.cmd.trim();
      if (!/^[A-Z_]+( [-0-9.e]+)*$/.test(cmd)) return; // firmware grammar only
      if (!write(cmd)) ws.send(JSON.stringify({ type: "log", text: "not connected to robot" }));
    });
    ws.on("close", () => {
      clients.delete(ws);
      if (clients.size === 0) {
        write("DISABLE");
        log("[bridge] last tab closed; sent DISABLE");
      }
    });
  });

  httpServer.on("upgrade", (req, socket, head) => {
    if (req.url !== "/bridge") return; // leave Vite's HMR socket alone
    wss.handleUpgrade(req, socket, head, (ws) => wss.emit("connection", ws, req));
  });
}
