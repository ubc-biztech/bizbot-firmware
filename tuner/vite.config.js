import { defineConfig } from "vite";
import fs from "node:fs";
import path from "node:path";
import { attachBridge } from "./bridge.js";

// The robot serves one controller. Two tuner servers would kick each other
// off every few seconds, so refuse to start if another one is alive.
const LOCK = path.join(import.meta.dirname, ".bridge.lock");
function takeLock() {
  try {
    const pid = Number(fs.readFileSync(LOCK, "utf8"));
    if (pid && pid !== process.pid) {
      try {
        process.kill(pid, 0); // throws if that process is gone
        console.error(`\nAnother tuner is already running (pid ${pid}). Use that one, or stop it first.\n`);
        process.exit(1);
      } catch (e) {
        if (e.code !== "ESRCH") throw e; // stale lock, fall through
      }
    }
  } catch (e) {
    if (e.code !== "ENOENT") throw e;
  }
  fs.writeFileSync(LOCK, String(process.pid));
  const drop = () => { try { if (fs.readFileSync(LOCK, "utf8") === String(process.pid)) fs.unlinkSync(LOCK); } catch {} };
  process.on("exit", drop);
  for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) process.on(sig, () => process.exit(0));
}
takeLock();

// Robot address. Override with BIZBOT_HOST / BIZBOT_PORT, e.g. when testing
// against a fake robot on localhost.
const host = process.env.BIZBOT_HOST || "192.168.4.1";
const port = Number(process.env.BIZBOT_PORT || 3333);

export default defineConfig({
  plugins: [
    {
      name: "bizbot-bridge",
      configureServer(server) {
        attachBridge(server.httpServer, {
          host,
          port,
          log: (m) => server.config.logger.info(m),
        });
      },
    },
  ],
});
