---
name: go2-stairs
description: Project skill for the Unitree Go2 stair-climbing RL work in /home/an/rl01 - uv environment, MuJoCo model, stand/walk/stairs scripts, the local MuJoCo + PyTorch PPO training pipeline (env.py, ppo.py, train.py, plot.py, play.py), the sim2sim contract (joint order, PD gains, observation and action conventions), how to launch, monitor, resume and stop training safely, evaluation, and what to record. Use whenever the user asks to set up, run, train, evaluate, debug, export, deploy or document anything in rl01 (Go2, MuJoCo, Isaac Lab, rsl_rl, stairs, velocity commands, PD gains, joint order, checkpoints, curves). Always read the work-discipline skill with it, and cautious.md for the layer you are touching.
---

# go2-stairs

Remember to do a smoke test before starting training; after finishing each training segment, remember to check the PNGs of consecutive frames, poses, and training curves for verification.

Project `/home/an/rl01` (git repository, nothing committed; commit only when the user asks).
Read the `work-discipline` skill with this one: supervision, ownership, concurrency, debugging, records and cleanup rules live there. `cautious.md` in the project root is the pitfall list; read the section for the layer you change.

Goal: a Go2 policy that climbs stairs. Hardware: 16-core CPU, 46 GiB RAM, RTX 5070 12 GB. Local MuJoCo + PyTorch training runs at ~50k samples/s; large Isaac Lab runs belong on a VPS (an RT-core GPU such as 4090 / L40S / A6000, run with `--headless`; check the current Isaac Sim requirements before renting).

## Ground rules
- **uv only**: `uv run python ...`, `uv add <pkg>`, commit-worthy files are `pyproject.toml` and `uv.lock`. No `pip install` into the venv; keep conda's base environment deactivated. The agent's shell may lack `uv` on PATH: use `~/.local/bin/uv`.
- Scripts import the package as `from rl01.x import ...` and run as `uv run python src/rl01/<script>.py`.
- `models/unitree_go2/` is MuJoCo Menagerie (MJCF, `.obj` meshes). `go2_description/` is a ROS URDF with `.dae` meshes that MuJoCo cannot read: never use it for simulation.
- Real dog: nothing is sent to hardware unless the user asks in this conversation. First hung up, remote in hand, area clear, low gains and small action scale. Never run the vendor motion service and our low-level commands together. Credentials never go into files or chat.
- Do not invent versions, task names, paths or flags (Isaac Sim / Isaac Lab change quickly). Check the installed version or `ls` the repo; say "not checked" otherwise.

## Layout and commands
| File | Purpose |
|---|---|
| `src/rl01/sim.py` | `build_model(step_h=...)` (optional stairs), `simulate(view=True)` stays open until the window is closed |
| `src/rl01/stand.py`, `walk.py` | PD stand test; open-loop trot (`--stairs <h>`, `--view`) |
| `src/rl01/env.py` | `Go2Env`: vectorized MuJoCo env on `mujoco.rollout.Rollout` (thread pool), per-env model copies, `EnvCfg` |
| `src/rl01/ppo.py` | Asymmetric actor-critic and PPO (GAE, adaptive KL learning rate, clipped value loss, optional left-right mirror-symmetry loss) |
| `src/rl01/symmetry.py` | Left-right mirror maps for observations, actions and states (verified against physics) |
| `src/rl01/train.py` | Training loop. "Step" = one PPO iteration (envs x 24 steps). Prints every 2.5% of total, saves a checkpoint every 10%, writes `metrics.csv` every iteration |
| `src/rl01/plot.py` | `curves.png` from `metrics.csv` (works during training); dotted lines mark checkpoints |
| `src/rl01/posture.py` | Per-leg posture report (joint deviation, foot outward offset mm, contact fraction, lift apex, swing fraction and mean swing-foot height, left-right differences) plus wobble metrics (roll/pitch RMS, angular rate RMS, body height ripple); always read the wobble line, speed tracking alone hides it and a front/side/top render; use the same script before and after any posture change |
| `src/rl01/blend.py` | `BlendedPolicy` (speed-gated action blend of two policies) and `save_blend`; `play.py`/`posture.py` accept `"low.pt,high.pt[,v_lo,v_hi]"` or a packed file |
| `src/rl01/play.py` | `--eval` (headless speed sweep with pushes), viewer with `--vx` or `--sweep` |
| `tools/proc.sh` | Register, list and stop background jobs by verified identity (pgid + start time + command line); see work-discipline 3.2 |

```bash
# long-lived jobs always go through tools/proc.sh (registers identity, own session, clean stop)
bash tools/proc.sh start train_<run> runs/train_<run>.log -- ~/.local/bin/uv run python src/rl01/train.py --iterations 5000 --name <run>
bash tools/proc.sh start train_<run> runs/train_<run>.log -- ~/.local/bin/uv run python src/rl01/train.py --resume runs/<run>/model_<n>.pt --iterations 5000
# fine-tune after a reward change: weights only, iteration restarts at 0, optimizer reset
bash tools/proc.sh start train_<run> runs/train_<run>.log -- ~/.local/bin/uv run python src/rl01/train.py --iterations 2000 --name <run> --init-from runs/<old>/model_<n>.pt --sym-coef 1.0 --lr 3e-4 --w-foot-clear -150
bash tools/proc.sh list
bash tools/proc.sh stop train_<run>        # or: stop-all
# short, bounded commands run directly
uv run python src/rl01/plot.py runs/<run>
uv run python src/rl01/play.py runs/<run>/model_<n>.pt --eval
uv run python src/rl01/play.py runs/<run>/model_<n>.pt --sweep
MUJOCO_GL=egl uv run python src/rl01/posture.py runs/<run>/model_<n>.pt --png out.png
```
Smoke-test any change to `env.py`, `ppo.py` or `train.py` first: `--iterations 8 --num-envs 128 --name smoke`, then delete that run directory.

## Contract that sim2sim and deployment must match (from `env.py`)
- Control 50 Hz (physics 0.002 s, 10 substeps). PD through MuJoCo position actuators: `tau = KP*(target - q) - KD*dq`, KP=20, KD=0.5, clipped to torque limits 23.7 (hip, thigh) / 45.43 (calf).
- Action `a` (12): `target = DEFAULT_Q + 0.25 * clip(a, -10, 10)`. Joint order **FL, FR, RL, RR**, each hip, thigh, calf. Unitree SDK order is FR, FL, RR, RL; Isaac Lab orders by type (all hips, then thighs, then calves). Map by joint name, never by index.
- `DEFAULT_Q` = hip +0.1/-0.1 (left/right), thigh 0.8 front / 1.0 rear, calf -1.5.
- Actor observation (45 per frame; optionally stacked over `--obs-taps` e.g. `0 1 2 3 4 6 9 13 18 24` -> 450 dims): base angular velocity x0.25 (body frame), projected gravity, command x(2, 2, 0.25), `q - DEFAULT_Q`, `dq` x0.05, last action. No linear velocity (the real dog does not measure it). Flat-ground critic adds 6 privileged values (body-frame linear velocity x2, base height, last push); stair critic additionally receives 15 terrain-height samples.
- Commands in the body frame; `vx` sampled in `[0, vmax]`, 10% of environments get 0. `vy` and yaw ranges default to 0 (forward only).
- Training model uses faster contact settings (pyramidal cone, impratio 1, foot condim 3, 30 solver iterations) and keeps Menagerie's joint damping 2 / frictionloss 0.2. Differences from Isaac Lab and from the original scene must be aligned or randomized before trusting sim2sim.
- Randomization and disturbances: random horizontal velocity pushes up to 0.6 m/s about every 5 s, observation noise, per-env foot friction 0.4 to 1.2 and base mass -1 to +1.5 kg.

## Current status (2026-11-20, updated after `v9`)
- Added 20-step staircases (`STAIR_STAGES` now 12 stages: `5/10/15 cm × 5/10/15/20 steps`; `STAIR_MAX_STEPS=20`), shared flat/stair speed ladder (`flat`: `[0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]` via `--stair-speed-ladder`; `stairs`: strictly capped by `--stair-vx-max 1.0`), continuous stair lateral/heading penalties (`--w-stair-y`, `--w-stair-yaw`, `--ang-sigma`), and 360° XY + Z + 3-axis angular pushes (`--push-ang`, `--push-z`, `--push-interval`).
- Current `v9` checkpoints kept in `runs/go2_stairs_v9/` for next session's continuation (trained 1000 iters from `v7/model_800.pt`, 35% flat + 65% 12-stage stairs, heavy DR `kp/kd 0.82–1.18, damp 0.45–1.25, com ±0.035, fric 0.35–1.25, mass -1.2–2.0`, 4.5s multi-dir pushes):
  - **`v9/model_800.pt`** (best slow/mid 15–20 step stair + top platform): `stairs_eval` `15cm/15`: **`16/16`** (`max_y mean=0.187 max=0.295`, top tilt `0.92°`), `15cm/20`: **`16/16`** (`max_y mean=0.205 max=0.336`, top tilt `0.88°`); sweep `s11@0.3/0.5`: `16/16` & `16/16` (`0.8/1.0`: `7/16`, `6/16`); flat pushes `0/0/0/2/19%`.
  - **`v9/model_400.pt`** (strongest DR + 15–20 step `1.0 m/s` stairs): sweep `s5`: `16/16/13/14`, `s8`: `16/16/7/14`, `s10`: `16/15/15/11`, `s11`: `16/14/4/13`; heavy DR+noise+multi-dir push `s5`: `22/32` (`19/32` w/ push), `s8`: `16/32` (`14/32`), `s11`: `12/32` (`10/32`); flat pushes `0/0/0/2/25%`.
  - **`v9/model_1000.pt`** (cleanest `v9` flat `2.0 m/s` posture): flat pushes `0/0/0/0/14%` (`vx=2.02`, `posture` 2.0 m/s pitch RMS `3.52°`, `foot_out=-23.0 mm`), `stairs_eval` `15/16` on `s8/s11`, but `s11@0.8` dropped to `1/16`.
- Known issue identified by user live replay & confirmed quantitatively: blind climb stumbles at Step 1 (`|wz|` jumps `0.037 -> 0.164 rad/s`, 4.4×) and Step N transition because `on_approach_walk` (`fw_x < 0.68 m`) enforces low `foot_cap` until 12 cm from Step 1; any resulting yaw twist persists along the blind 20-step climb. Smoothing entry/exit `foot_cap`/`climb_tgt` transitions is the next priority.
- Prior baselines retained: `runs/go2_stairs_v8/` (diagnostic evidence for `stair_init_vmax=0.5` curriculum lock), `runs/go2_stairs_v7/model_800.pt`, `runs/go2_stairs_v6/model_1000.pt`, `runs/go2_stairs_v5/model_100.pt`.
- `train.py` flags: `v7` flags plus `--stair-vx-max`, `--stair-init-vmax`, `--stair-speed-ladder`, `--stair-lateral-limit`, `--w-stair-y`, `--w-stair-yaw`, `--ang-sigma`, `--friction-range`, `--mass-range`, `--push-ang`, `--push-z`, `--push-interval`, `--cmd-resample-s`. `stairs_play.py --steps` now supports `5/10/15/20`.
- No sim2sim or real-robot verification has been run yet.

## Order of work
1. Stand and open-loop checks in MuJoCo (done).
2. Flat velocity and low-speed gait training were completed previously; their checkpoints were deleted at the user’s request.
3. Continue the paused stair curriculum from the retained checkpoint when the user asks, checking gait, curves, and independent evaluations between training intervals. Explore a blind estimator or teacher-student if taller steps require it.
4. VPS Isaac Lab run if needed (built-in Go2 rough-terrain task first, untouched).
5. sim2sim: export the policy, run it in an original-settings MuJoCo scene with the contract above.
6. Real dog: hung up, standing, short tap, then stairs.

## Process and concurrency essentials (full rules: work-discipline section 3)
- Before launching: `bash tools/proc.sh list`, read-only look at other load (`ps`, `nvidia-smi`), one CPU-heavy job at a time. Jobs you did not start are not yours to stop.
- Launch and stop only through `tools/proc.sh`; monitor the log with a filter that includes `Traceback|Error|Killed` and the completion line, and re-arm monitors per run.
- Do not run CPU-heavy evaluations while training unless needed; they slow the run. Resume restores optimizer, learning rate and curriculum; truncate `metrics.csv` to the checkpoint iteration first.
- Read every progress event critically: survival (flat max 1000, stair max 2250), falls vs timeouts, tracking, actual vs commanded speed, curriculum cap, body height in evaluations.
- When the work ends: `proc.sh stop-all`, stop leftover monitors and background tasks, confirm with `proc.sh list` plus a read-only `ps`, and say so in the final answer (work-discipline 3.5).

## Verify before claiming
- Low loss or rising reward is not success: evaluate and look at behavior. Decompose rewards (`env.term_means`) before changing weights. Confirm what an observation, action or reward term means before trusting it.
- Look at pictures (render a frame, replay) for terrain and gait.
- Speed tracking, falls and contact fraction do not show wobble: also check the wobble metrics, and treat what the user sees as evidence before your own metrics.
- State the verification layer every time: MuJoCo only, sim2sim, real standing, real motion.

## Records
Append a Chinese section per session to `docs/record.md`: goal and environment, then per problem phenomenon, attempts, cause (or "unknown" with what is excluded), choice and files changed, verification (and how many runs), then a conclusion with the verification layer and next step. New pitfalls go to `cautious.md`; changed commands or conventions go to this skill. Search for stale text with `grep -rn`.

## Stair curriculum (2026-11-18, experimental)
- `train.py --stairs --track-sigma 0.05`: simulated riser heights 0.05/0.10/0.15 m for each of 5/10/15 steps, always 0.25 m tread; 1.2 m total lateral width is an assumption, not user-measured. Stage advances on at least 128 current-stage finished episodes with at least 70% success in the recent window. Completion means the four feet remain within the platform bounds for 0.5 s and each actually touches the top during that interval, allowing alternating swing. Initial approach always points at stairs. DR retains friction, mass, noise and pushes; actor remains blind and critic adds 15 terrain heights. Resume resets rollout state and RNG, so is not exact replay.
- Evaluate and inspect using `MUJOCO_GL=egl uv run python src/rl01/stairs_eval.py <checkpoint> --stage 0 --num-envs 16 --seconds 20 --out <png>` and `uv run python src/rl01/plot.py <run>`; evaluate the same seeds and settings for comparison. Reports include per-level four-foot positions and lateral drift; position on its own is not proof of foot contact. Pause the owned 1024-env run before rendering if memory is low. Earlier v1 stalled and v2 showed lateral deviation; those checkpoints have been deleted per the user's request, and v3 introduced lateral termination. Check `docs/record.md` for latest results.

- `--stair-front-bonus 0.2 --stair-rear-bonus 0.15` rewards a foot once per new tread after three consecutive touch readings, with a front foot reaching that tread before rear reward. `--w-diagonal 0.1` rewards diagonal support while moving; inspect gait and actual success rather than assuming it works. `stairs_eval.py` can capture dense frames via `--frame-start 4 --frame-interval 0.1` and reports top-platform foot lift and tilt.
- Mixed flat + stairs and top-platform walk options in `train.py`: `--flat-frac 0.25`, `--stair-mix-prev 0.3`, `--stair-top 2.0 --stair-top-steps 100`, `--riser-clear-margin 0.045`, `--obs-taps 0 1 2 3 4 6 9 13 18 24`, plus `speed_clear` + Huber `w_foot_cap -120` + `w_orient_low -20 --w-ang-vel-low -0.4` applied on flat envs, approach (`<0.68 m`), and top platform.
- Interactive stair viewer: `uv run python src/rl01/stairs_play.py runs/go2_stairs_v6/model_1000.pt --height 15 --steps 10 --push 0`. Height choices 5/10/15 cm, steps 5/10/15; close the window to exit. For a window-free execution check use `--headless-seconds 1`. Pause CPU-heavy training before the user views it.
