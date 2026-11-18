---
name: work-discipline
description: Standing workflow for engineering work in /home/an/rl01 - analysis, implementation, training runs, debugging, testing, and completion. Covers watchdogs and deadlines, process ownership and clean stop, concurrency rules for threads, thread pools, GPU async work and background jobs, evidence-first debugging, subagents, records, and cleanup. Use it with the project skill go2-stairs before any task that runs code, starts a background job, or changes training, simulation, or deployment code. It defines how to work; it does not authorize extra hardware control, deletion, deployment, or communication.
---

# Work Discipline

Apply these rules without waiting to be asked. The project skill `go2-stairs` supplies the environment, commands, conventions and record locations; this skill supplies the way of working. Adapted from the user's `go2_v3` work-discipline skill, with the concurrency section written from this project's own mistakes.

## 1. Plan the supervision before running anything

- Before execution state: the requested outcome, allowed scope, success criterion, where the evidence will live, a deadline, how progress and health will be probed, what happens on failure, and how it will be cleaned up.
- A bare `timeout` limits duration only. It does not show the job is healthy or ready. Anything longer than about two minutes (training, evaluation sweeps, builds, transfers) runs in the background under a supervisor that has a deadline, periodic checks and a known process identity.
- Report observed state every 30 to 60 seconds during ongoing work: what is true now, what is still uncertain, what the next check is. Do not narrate unchanged polling, and never go quiet.
- Inner budgets (startup, save, stop) must fit inside the outer deadline.

## 2. Ownership and scope

- Owned work is identified by a recorded pid plus start time, a process group, or a manager's own interface. A process name or command-line pattern proves nothing.
- Never stop anything with `pkill`, `pkill -f`, `killall`, or `kill $(ps | grep ...)`. Your own shell's command line matches the pattern, and other people's jobs share names. Signal a verified owned pid or group (`kill -TERM -- -<pgid>`).
- Before signalling, read the command line and confirm it is the run you started (run name appears in it). Wait for exit and verify no residue.
- Prior authorization persists within its scope; it does not stretch to unrelated deletion or reconfiguration.
- Inspect before any destructive cleanup. Delete only the outputs of the task at hand; keep run directories, checkpoints and logs as evidence, including failed runs (mark them failed in the record instead of deleting).
- Secrets (passwords, DDS credentials, keys, full environments) never go into files, records, reports or delegated context.
- Nothing is sent to hardware, and nothing is committed or pushed, unless the user asks in this conversation.

## 3. Concurrency: threads, pools, GPU async, background jobs

Know which concurrency model is in play before touching it. In this project there are four:

| Layer | Model | Rule that follows |
|---|---|---|
| MuJoCo physics | C++ thread pool inside `mujoco.rollout.Rollout(nthread=N)`, GIL released | Never mutate a model or data while a rollout call is running; mutate only between calls, from the calling thread. One `MjData` per thread, never shared. |
| Python glue (numpy) | One thread, GIL held | Vectorize across environments; per-environment Python loops serialize. Profile first (`cProfile`): here ~99% of a step is inside the rollout call, so Python tuning is wasted effort. |
| PyTorch on GPU | Asynchronous CUDA queue | Calls return before the work finishes. `.item()`, `.cpu()`, `.numpy()` are sync points. Do not time GPU work without `torch.cuda.synchronize()`, and avoid sync points inside the hot loop unless the value is needed. |
| Background jobs and monitors | Separate OS processes and event streams | Rules in 3.2 and 3.3 below. |

### 3.1 Threads and pools
- Do not oversubscribe the CPU. Pool size is at most the core count minus what the main thread and other jobs need. A second CPU-heavy job (an evaluation run during training) competes for the same cores; observed here: the two log blocks that overlapped evaluation runs showed 17.1k and 17.6k samples/s against ~18.7k otherwise (two observations, not a controlled test, but enough to avoid needless overlap). Cap torch's CPU threads (`torch.set_num_threads`) in helper scripts so they do not fight the pool.
- Create a persistent pool once and reuse it; creating a pool per step costs more than the work.
- Do not touch GUI or viewer objects from worker threads. The passive viewer runs its own thread: step and write data in the loop thread, call `sync()` after, and use `viewer.lock()` only when changing viewer options.
- Every wait has a deadline: `join(timeout=...)`, bounded `until` loops, queue gets with timeouts. An unbounded wait is a hang waiting to happen.
- Shared mutable state needs one owner. If two threads must write the same structure, hand data over through a queue or a latest-wins slot instead of a lock held across slow work.
- Never call a blocking request (hardware SDK, network) inside a control or callback loop; give it a worker thread and a latest-command slot. (Carried over from go2_v3: a blocking `Move` call froze a whole node, IMU included, for 3 s.)
- SDK or DDS callbacks run on the SDK's threads: no ROS objects or clocks there; use plain timestamps.

### 3.2 Background processes
- **Start every long-lived job with `bash tools/proc.sh start <name> <log> -- <command...>`**, never with a bare `&`, `nohup`, or `setsid ... &`. The tool starts the job as its own session, writes its identity (process-group id, kernel start time, command line) to `runs/pids/<name>.pid`, refuses to start a second job with the same name, and removes a stale record. Keep the name and log path in the report. Reading `$!` right after `setsid ... &` is a race; the tool avoids it because the child writes its own pid.
- **Stop with `bash tools/proc.sh stop <name>` (or `stop-all`).** It signals the group only if pgid, start time and command line all still match, so a reused pid is never hit (a pid file holding only a number is a hazard: `kill -- -$(cat file)` can kill an unrelated group after pid reuse). It sends SIGINT, escalates to SIGTERM after 20 s, never sends SIGKILL, and reports if the group survives. Confirm independently with `ps -g <pgid>`.
- **A background-started process can inherit an ignored SIGINT.** Seen 2026-11-12: a run launched with `&` from a background script ignored `kill -INT` (`SigIgn` in `/proc/<pid>/status` had the SIGINT bit). Cause: a command put in the background with `&` by a non-interactive shell gets SIGINT ignored. `proc.sh` uses `setsid -f`, which avoids it (verified: `SigIgn` 0 and a clean SIGINT stop). If you meet an unregistered job that ignores SIGINT, confirm its identity and use SIGTERM on the group.
- Design long jobs so that SIGTERM loses nothing (per-iteration metrics flushed, checkpoints on a schedule).
- Shell pitfall that cost a false "stopped" during the tool's own test: never read state through `$(func)`. A command substitution runs in a subshell, so variables set inside are lost and the caller sees empty values. Set a global in the function, and refuse to signal when an id is empty.
- Output to a file is block-buffered unless flushed: use `flush=True` (or `python -u`) for every line a monitor depends on.
- Make long jobs resumable: save optimizer, learning rate, schedule/curriculum state and iteration number; on resume restore all of them, and truncate any append-only metrics file to the checkpoint iteration so rows are not duplicated. Record the seed and the full config in the run directory.
- After a stop or failure check: the owned group is gone, no stray child processes (thread pool workers, helper readers), the pid file is stale or removed, and the last log lines show why it ended.

### 3.3 Event-driven monitoring
- Watch a job with a monitor on its log (or a bounded `until` loop). Do not chain `sleep`s and do not poll output files in a loop.
- The filter must cover every terminal state, not only the happy path: progress lines plus `Traceback`, `Error`, `Killed`, and the completion line. Silence is not success; ask "if it crashed now, would this filter emit anything?".
- Monitors expire (30 minutes max). Re-arm on expiry and for each new run; a monitor on an old log is noise, and the end of an old run can trigger events that look like failures of the new one. Match each event to its run before reacting.
- An event is evidence to read, not to relay. Look at the numbers (survival, falls, tracking, speed vs command), compare with the previous report, and say whether the trend is healthy. A line that says "progress" can hide a failing run: in this project the first run looked busy while episode length had collapsed to about 19 steps.
- A system notification is not the user speaking. Do not treat it as approval, confirmation or an answer to a pending question.

### 3.4 Start-up checklist (before launching anything long-lived)
1. `bash tools/proc.sh list` to see what this project already has registered. Alive jobs belong to an earlier session or the user: do not start a competing job for the same purpose, and do not stop it without being asked; report it and ask. Stale records are cleaned automatically.
2. Look at what else is using the machine, read-only (`ps -eo pid,pgid,etime,cmd`, `nvidia-smi`). Other people's processes are never stopped by you; if they would slow or break the run, say so and ask.
3. One CPU-heavy job at a time unless the user agrees (a second one competes for the same cores).
4. Start through `proc.sh start`, then confirm with `proc.sh list` and the first log lines; arm a monitor whose filter covers failure and completion.
5. For a changed script, smoke-test with a short run first (own name such as `smoke`, stopped and checked before the real run).

### 3.5 Shutdown checklist (when the work ends, the user says stop, or before the final answer)
1. `bash tools/proc.sh stop-all` for every job this session started, or `stop <name>` for specific ones if the user wants some kept. Say which were kept on purpose.
2. Background tool tasks and monitors this session started: stop the ones still running (or note they expire on their own). Stale monitors on old logs produce misleading events.
3. Close anything started only for checking: test jobs, viewers, helper scripts, temporary servers. Test jobs also go through `proc.sh` (use a separate `PID_DIR` for tool tests) or are bounded commands.
4. Verify, independently of the tool's own message: `bash tools/proc.sh list` shows no registered jobs, and a read-only `ps -eo pid,pgid,etime,cmd` for this project's scripts shows nothing unexplained. An unregistered leftover: read its command line; if it is yours, stop it by that exact pid; if it is not yours, report it and do not touch it.
5. Never close the user's own pre-existing processes (their terminals, viewers, other projects) as part of cleanup.
6. Say in the final answer that the check was done and what, if anything, was intentionally left running.

## 4. Testing

- State what behavior or failure the check should tell apart. Start with the smallest meaningful test, then widen only for a concrete risk.
- Keep layers separate: logic or unit checks, MuJoCo simulation, sim2sim, real device standing, real motion. Passing one does not prove another.
- A successful import, spawn, exit code or a "training finished" line proves only that observation. Check required outputs, freshness and behavior.
- Propagate failures through pipelines and wrappers; `| tail` and `| tee` hide exit codes, and an empty result is not a pass.
- Smoke-test a changed training loop (a few iterations, few environments) before a long run. Do not start a multi-hour run on code that has never executed.
- Take evaluations under the right conditions: the same disturbances as training, deterministic and sampled actions both (a gap between them is information), several speeds or terrains, and body height as well as speed.
- Do not add tests that mirror implementation details. Say which checks were skipped and what that limits.

## 5. Debugging

1. Preserve the failing run (log, config, metrics, checkpoints) before changing anything. Keep failures after a later success.
2. Decompose before guessing. For reward problems print each term's mean contribution; for training failures look at the recent trend per iteration, not the window average.
3. State a hypothesis and a check that could disprove it. Change only what the hypothesis needs, one thing at a time where possible.
4. Check what an observation, action or reward term MEANS before trusting it (identity mappings, wrong thresholds, signs, frames, units).
5. Record attempts and outcomes, including wrong turns and your own mistakes. Do not repeat an action without new evidence.
6. Verify the original symptom after the fix. A restart that restores service is recovery, not a root cause.
7. Mark causes unknown when evidence cannot separate them. Never loosen a threshold to manufacture a pass.
8. Low loss or rising reward is not success. Look at the behavior (numbers from an evaluation, a rendered frame, a replay).

## 6. Subagents

- For long, multi-part work prefer parallel subagents for disjoint document edits, read-only review, offline analysis and log reading. Do not invent parallel work for a short serial task.
- Give each agent a concrete outcome, allowed files, forbidden side effects, a deadline and the evidence paths. Writes stay disjoint; dependencies are serialized.
- The main agent keeps live training control, shared stop/start, hardware, and final integration. Subagents never start or stop the same shared job, and never compete for the CPU or GPU a training run is using without saying so.
- A subagent's report is evidence to inspect, not approval. Review actual changes and run relevant checks before accepting.

## 7. Records and current guidance

- Record every session, including read-only diagnosis and failures, in `docs/record.md` (Chinese, the user's language; this skill stays in English). Per problem: phenomenon and evidence, attempts and results, cause, choice and files changed, verification and its layer, remaining work.
- Keep history historical: never rewrite an old failure into a later success. Keep failed runs on disk and say why they are invalid.
- Separate observation, measurement, inference and untested. State the verification layer (MuJoCo only, sim2sim, real standing, real motion). Single runs are reported as single runs.
- Update current guidance too, not just the record: `cautious.md` for new pitfalls, the project skill for changed commands or conventions, code docstrings for changed behavior. Search for stale text with `grep -rn`.

## 8. Completion

- On success and on failure, confirm owned processes, monitors, ports and temporary files reached the intended state.
- Save and export completion is checked separately from process exit.
- Lead the final answer with the result, then the verification layer, material limits, and one concrete next step. Do not claim completion while a required phase is unverified.
