"""Frozen flat-policy guidance and exact observation-history checkpoint mapping.

The flat teacher supplies action targets only during training. Deployment uses
the single student actor; it does not switch controllers at a terrain boundary.
"""
import torch

from rl01.ppo import ActorCritic


def observation_layout(config, actor_dim):
    frame = 45 + int(config.get('heading_obs', False)) + 2 * int(config.get('gait_phase_obs', False))
    taps = tuple(config.get('obs_taps', (0,)))
    if not taps or taps[0] != 0 or len(set(taps)) != len(taps) or min(taps) < 0:
        raise ValueError(f'Invalid checkpoint observation taps: {taps}')
    if frame * len(taps) != actor_dim:
        raise ValueError(f'Observation contract mismatch: {frame} x {taps} != {actor_dim}')
    return frame, taps


def remap_initial_state(checkpoint, student_state, student_config):
    """Preserve each old lag at its exact time offset; zero newly added inputs."""
    state = dict(checkpoint['model'])
    old_frame, old_taps = observation_layout(checkpoint['env_cfg'], state['actor.0.weight'].shape[1])
    new_frame, new_taps = observation_layout(student_config, student_state['actor.0.weight'].shape[1])
    missing = set(old_taps) - set(new_taps)
    if missing or new_frame < old_frame:
        raise ValueError(f'init-from would discard observations: missing taps {sorted(missing)}, frame {old_frame}->{new_frame}')
    actor = torch.zeros_like(student_state['actor.0.weight'])
    for old_i, tap in enumerate(old_taps):
        new_i = new_taps.index(tap)
        actor[:, new_i * new_frame:new_i * new_frame + old_frame] = state['actor.0.weight'][:, old_i * old_frame:(old_i + 1) * old_frame]
    state['actor.0.weight'] = actor
    old_critic = state['critic.0.weight']
    critic = torch.zeros_like(student_state['critic.0.weight'])
    critic[:, :old_frame] = old_critic[:, :old_frame]
    extras = min(old_critic.shape[1] - old_frame, critic.shape[1] - new_frame)
    critic[:, new_frame:new_frame + extras] = old_critic[:, old_frame:old_frame + extras]
    state['critic.0.weight'] = critic
    return state


class FlatTeacher:
    """Adapt student history to a frozen teacher, without evaluating stair rows."""
    def __init__(self, path, student_config, student_actor_dim, device):
        self.checkpoint = torch.load(path, map_location=device, weights_only=False)
        model, config = self.checkpoint['model'], self.checkpoint['env_cfg']
        if config.get('stairs', False):
            raise ValueError('flat-teacher must be a flat-ground policy')
        self.frame, teacher_taps = observation_layout(config, model['actor.0.weight'].shape[1])
        self.student_frame, student_taps = observation_layout(student_config, student_actor_dim)
        if self.student_frame < self.frame or not set(teacher_taps).issubset(student_taps):
            raise ValueError(f'flat-teacher requires frame {self.frame}, taps {teacher_taps}; student has {self.student_frame}, {student_taps}')
        for name in ('heading_obs', 'gait_phase_obs'):
            if bool(config.get(name, False)) and not bool(student_config.get(name, False)):
                raise ValueError(f'flat-teacher observation feature missing: {name}')
        if config.get('gait_phase_obs', False):
            for name, default in (('gait_frequency', 1 / .44), ('gait_frequency_speed', 0.0)):
                if abs(config.get(name, default) - student_config.get(name, default)) > 1e-9:
                    raise ValueError(f'flat-teacher phase timing differs: {name}')
        self.indices = torch.tensor([student_taps.index(t) for t in teacher_taps], device=device)
        self.actor = ActorCritic(model['actor.0.weight'].shape[1], model['critic.0.weight'].shape[1], model['std'].numel()).to(device)
        self.actor.load_state_dict(model)
        self.actor.eval().requires_grad_(False)
        self.actions = model['std'].numel()

    @torch.no_grad()
    def targets(self, student_obs, is_flat):
        mask = torch.as_tensor(is_flat, dtype=torch.bool, device=student_obs.device)
        target = student_obs.new_zeros((len(student_obs), self.actions))
        if mask.any():
            frames = student_obs[mask].reshape(int(mask.sum()), -1, self.student_frame)
            teacher_obs = frames[:, self.indices, :self.frame].reshape(int(mask.sum()), -1)
            target[mask] = self.actor.actor(teacher_obs)
        return target, mask
