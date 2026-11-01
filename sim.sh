#!/bin/zsh
# Open the MuJoCo sim in one browser tab: PID, PID Daniel, RL old and RL best on the fitted carpet robot.
#
#   tools/sim.sh                      start on the PID tab
#   tools/sim.sh --mode "RL best"     any model/mj/watch.py option overrides or adds to these
#   BIZBOT_ROBOT=<robot.json> tools/sim.sh
cd "${0:A:h}/.." || exit 1
ROBOT="${BIZBOT_ROBOT:-$HOME/bizbot_sysid/real2sim/v2/fitted_robot_carpet.json}"
exec .venv/bin/python -u model/mj/watch.py --mode PID --still-start \
  --policy "RL old=model/mj/runs/ft_merged/ckpt_0040M.pt" \
  --policy "RL best=model/mj/runs/fit_ft_I/ckpt_0080M.pt" \
  --robot "$ROBOT" "$@"
