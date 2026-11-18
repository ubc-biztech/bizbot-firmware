# Common BizBot commands. Run `make` for the list.
# Override any variable on the command line, e.g. `make flash PORT=/dev/cu.usbserial-10`.

PIO    ?= $(HOME)/.platformio/penv/bin/pio
PY     ?= .venv/bin/python
# First USB serial adapter plugged in; its number changes with the board and USB port.
PORT   ?= $(firstword $(wildcard /dev/cu.usbserial-*))
POLICY ?= model/mj/runs/fit_tgt_E/final.pt
NEW    ?= model/mj/runs/tgt_ft_C/final.pt
ROBOT  ?= $(HOME)/bizbot_sysid/real2sim/v3/robot_measured_delay.json
OUT    ?= model/deploy

.DEFAULT_GOAL := help
.PHONY: help build flash servo-sweep drive tuner sim sim-compare export test

help:
	@echo "make build        build the robot firmware (firmware/)"
	@echo "make flash        build and upload it to PORT=$(PORT); turn the hoverboard OFF first"
	@echo "make servo-sweep  upload the separate servo test (firmware/servo-sweep) to PORT"
	@echo "make drive        connect over BizBot-Control Wi-Fi and open the drive page (balances on connect), log to ~/bizbot_<time>.log"
	@echo "make tuner        browser tuner at http://localhost:5173 (tools/tuner)"
	@echo "make sim          MuJoCo viewer: POLICY=$(POLICY) on ROBOT"
	@echo "make sim-compare  MuJoCo viewer with two tabs: RL old=POLICY (orange) and RL new=NEW (red)"
	@echo "make export       export POLICY to C in OUT=$(OUT) (runs the host test)"
	@echo "make test         sim/drive unit tests (model/mj/tests)"

build:
	$(PIO) run -d firmware

flash:
	@echo ">>> hoverboard must be OFF while flashing (flashing with it on stalls its UART)"
	$(PIO) run -d firmware -t upload --upload-port $(PORT)

servo-sweep:
	$(PIO) run -d firmware/servo-sweep -t upload --upload-port $(PORT)

drive:
	$(PY) -u tools/drive_ui.py --transport wifi 2>&1 | tee $(HOME)/bizbot_$$(date +%Y%m%d_%H%M%S).log

tuner:
	cd tools/tuner && npm run dev

sim:
	$(PY) -u model/mj/watch.py --mode RL --still-start --policy $(POLICY) --robot $(ROBOT)

sim-compare:
	$(PY) -u model/mj/watch.py --mode "RL new" --still-start --policy "RL old=$(POLICY)" --policy "RL new=$(NEW)" --robot $(ROBOT)

export:
	$(PY) -u model/mj/export_policy.py $(POLICY) --out $(OUT)

# tools/tests needs a Python with Tkinter, which .venv does not have, so it is not run here.
test:
	cd model/mj/tests && $(abspath $(PY)) -m unittest discover -s .
