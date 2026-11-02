# Rules for AI assistants in this repo (Claude, Codex)

`CLAUDE.md` is a link to this file; edit only this one.

## Scope: do exactly what was asked, nothing else

- Change only what the user's message asks for. Do not refactor, "clean up",
  remove guards, rename, restructure, or "improve" anything adjacent.
- If a requested change appears to require touching something else, stop and
  say so in one sentence before doing it. Never bundle it silently.
- When rewriting a file, preserve every behaviour the old version had unless
  the user asked for that behaviour to go. List anything dropped.
- After a change, state every file and behaviour that changed, including side
  effects, in plain words. No omissions.
- Do not add features, commands, telemetry, docs, comments, or hints that were
  not requested.
- Do not commit, flash, or restart anything unless told to in that message.
- One change per turn, atomic and complete. Never leave a change half done
  while starting another. A change that spans firmware and tooling is one
  change only if the user asked for both; otherwise ask which first.

## When the user reports a bug

- Find the cause in the code before proposing anything. No guessing dressed up
  as diagnosis.
- Say what the cause is, then fix that and only that.

## This project

- Firmware is flashed with the hoverboard OFF; flashing with it on stalls its
  UART (needs a hoverboard power cycle with the ESP32 up).
- Joining the robot's Wi-Fi kills this session; ask the user to collect logs
  rather than simulating hardware.
- `tools/tuner/` drives the robot by spawning `tools/keyboard_controls.py`.
  Python must be started with `-u` or its output is block-buffered.
- Never put a call with side effects, or an expensive one, inside
  `constrain`, `min`, `max`, `abs` or any other Arduino macro: they paste the
  argument in up to three times. Store the result first, then clamp it.
  (2026-11-01: `constrain(bizbot_policy_tick(...), ...)` ran the RL policy
  3x per tick, stepping its history, integral and rate limit 3x and taking
  7.3 ms instead of 2.4 ms; the robot death-wobbled until it was split.)
