# BizBot balance policy: contract

Source: `model/mj/runs/ft_merged/ckpt_0040M.pt`. Network: 162 inputs -> 128 ELU -> 128 ELU -> 1 tanh,
37505 parameters, about 37248 multiply-adds per tick.

## Call pattern
- `bizbot_policy_reset(&s)` when balancing (re)starts.
- `bizbot_policy_tick(&s, pitch_deg, accel_fwd_mps2, wheel_rpm, battery_v)` every
  10 ms with the latest readings; send the returned value to the hoverboard as the
  speed command (steer 0). It is already rate limited (0.25 of full scale per tick) and
  scaled to +-500 (truncated toward zero).

## Inputs (signs are what the sim used; check them on the robot)
| input | units | sign | note |
|---|---|---|---|
| pitch_deg | degrees | + leaning forward | the IMU reading; an offset up to about +-3 deg is fine |
| accel_fwd_mps2 | m/s^2 | + accelerating forward | raw accelerometer along the forward axis, gravity included: about -9.81*sin(pitch) at rest |
| wheel_rpm | rpm | + rolling forward | mean of left and right after their individual signs are fixed |
| battery_v | volts | | |

The runtime keeps the last 40 ticks of (pitch/10, accel/9.81, rpm/60, its own command),
a leaky sum of wheel revolutions (decays over 20 s, clipped to +-2.0),
and appends (battery-37)/5 and that sum/2.0; the network normalises each input with the
training statistics and clips to +-10.0.

## Output
Hoverboard speed command, + = wheels drive the robot forward. The hoverboard must be in speed mode.

## Verified
`test_policy.c` replays 4 simulated robots x 400 ticks through the C runtime and
checks it reproduces the sim's commands.
