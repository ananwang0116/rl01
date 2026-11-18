"""Independent stair evaluation and side-view filmstrip; MuJoCo only."""
import argparse
from pathlib import Path
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from rl01.env import EnvCfg, Go2Env, STAIR_STAGES, STAIR_X0, build_train_model
from rl01.play import load_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--stage', type=int, default=0, choices=range(len(STAIR_STAGES)))
    p.add_argument('--num-envs', type=int, default=16)
    p.add_argument('--seed', type=int, default=719)
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--out', required=True)
    p.add_argument('--frame-start', type=float, default=0.0)
    p.add_argument('--frame-interval', type=float, default=3.0)
    a = p.parse_args()
    torch.set_num_threads(2)
    ac, ck = load_policy(a.checkpoint)
    cfg = EnvCfg(stairs=True, episode_s=a.seconds, push_vel=0.0, noise=False, speed_clear=False, w_foot_clear=-150)
    env = Go2Env(a.num_envs, cfg, seed=a.seed, nthread=4)
    env.stage = a.stage
    env.reset_idx(np.arange(a.num_envs))
    env._compute_obs()
    # One episode per environment; reset samples are ignored in summary.
    active = np.ones(a.num_envs, bool)
    max_x = env.state[:, 1].copy()
    first_success = first_fall = first_timeout = 0
    max_abs_y = np.abs(env.state[:, 2]).copy()
    max_level = np.zeros(a.num_envs, dtype=int)
    first_front = np.full(a.num_envs, -1, dtype=int)
    first_rear = np.full(a.num_envs, -1, dtype=int)
    diag_count = moving_count = 0
    swing_z = [[], [], [], []]
    model = build_train_model(visual=True, stair_stage=a.stage)
    # Match the viewer: disable the light shadow that produces black render artifacts.
    model.light_castshadow[:] = False
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 360, 640)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.azimuth = 90; cam.elevation = -12; cam.distance = 3
    frames = []
    with torch.no_grad():
        for step in range(int(a.seconds / env.dt)):
            obs, _ = env.get_obs()
            act = ac.actor(torch.as_tensor(obs)).numpy()
            _, done, _ = env.step(act)
            max_x[active & ~done] = np.maximum(max_x[active & ~done], env.state[active & ~done, 1])
            success, fall, timeout = env.last_outcome
            first_success += int((active & success).sum())
            first_fall += int((active & fall).sum())
            first_timeout += int((active & timeout).sum())
            max_abs_y[active & ~done] = np.maximum(max_abs_y[active & ~done], np.abs(env.state[active & ~done, 2]))
            feet = env.sens_buf[:, -1, :12].reshape(a.num_envs, 4, 3)
            height, count = STAIR_STAGES[a.stage]
            touch = env.sens_buf[:, -1, 24:28] > 0.1
            on_first = touch & (feet[:, :, 0] > STAIR_X0) & (feet[:, :, 2] > height + 0.015) & (np.abs(feet[:, :, 1]) < 0.6)
            moving = active & (env.lin_b[:, 0] > 0.08)
            diag = (touch[:, 0] & touch[:, 3] & ~touch[:, 1] & ~touch[:, 2]) | (touch[:, 1] & touch[:, 2] & ~touch[:, 0] & ~touch[:, 3])
            diag_count += int((moving & diag).sum()); moving_count += int(moving.sum())
            hs = np.array([STAIR_STAGES[i][0] for i in env.env_stage])[:, None]
            ground = np.minimum(env.terrain_height(feet[:, :, 0], feet[:, :, 1]), np.maximum(0, np.floor(feet[:, :, 2] / hs) * hs))
            for leg in range(4):
                z = feet[:, leg, 2] - ground[:, leg]
                swing_z[leg].extend(z[moving & ~touch[:, leg]].tolist())
            front_hit = active & (first_front < 0) & on_first[:, :2].any(axis=1)
            rear_hit = active & (first_rear < 0) & on_first[:, 2:].any(axis=1)
            first_front[front_hit] = step; first_rear[rear_hit] = step
            for level in range(1, count + 1):
                reached = ((feet[:, :, 0] >= STAIR_X0 + (level - 1) * 0.25).all(axis=1) &
                           (feet[:, :, 2] >= level * height - 0.03).all(axis=1) &
                           (np.abs(feet[:, :, 1]) < 0.6).all(axis=1))
                max_level[active & reached] = np.maximum(max_level[active & reached], level)
            active &= ~done
            frame_start = round(a.frame_start / env.dt)
            frame_interval = max(1, round(a.frame_interval / env.dt))
            if step >= frame_start and (step - frame_start) % frame_interval == 0 and len(frames) < 10:
                mujoco.mj_setState(model, data, env.state[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
                mujoco.mj_forward(model, data)
                cam.lookat[:] = [env.state[0, 1] + 0.5, 0, max(0.3, env.state[0, 3])]
                renderer.update_scene(data, camera=cam)
                frames.append((step * env.dt, renderer.render().copy()))
    st = env.pop_stats()
    renderer.close()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    w, h = 640, 360
    canvas = Image.new('RGB', (w * 2, (h + 25) * ((len(frames) + 1) // 2)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (t, frame) in enumerate(frames):
        x, y = (i % 2) * w, (i // 2) * (h + 25)
        canvas.paste(Image.fromarray(frame), (x, y + 25))
        draw.text((x + 5, y + 5), f'{t:.1f} s', fill='black')
    canvas.save(out)
    print(f'stage={a.stage} h={STAIR_STAGES[a.stage][0]:.2f} n={STAIR_STAGES[a.stage][1]} seed={a.seed} envs={a.num_envs} seconds={a.seconds}', flush=True)
    print(f'first episodes: success={first_success} falls={first_fall} timeout={first_timeout}', flush=True)
    print(f'first-episode max x mean={max_x.mean():.3f} median={np.median(max_x):.3f} first_riser={STAIR_X0:.2f}; active_at_end={active.sum()}', flush=True)
    print(f'first episodes: all four feet reach levels >=1..N: {[int((max_level >= k).sum()) for k in range(1, STAIR_STAGES[a.stage][1] + 1)]}; base |y|>0.45: {(max_abs_y > 0.45).sum()}', flush=True)
    print(f'first step contact: front={(first_front >= 0).sum()}, rear={(first_rear >= 0).sum()}, front-before-rear={((first_front >= 0) & (first_rear >= first_front)).sum()}', flush=True)
    lift = [round(float(np.percentile(z, 95)) * 100, 1) if z else None for z in swing_z]
    print(f'moving samples={moving_count}, strict diagonal two-foot support fraction={diag_count / moving_count if moving_count else 0:.3f}; airborne relative foot-centre height 95th percentile cm FL FR RL RR={lift}', flush=True)
    print(f'filmstrip={out}', flush=True)

if __name__ == '__main__': main()
