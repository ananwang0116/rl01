"""Unified stair heading/transitions and flat omni/restart training.

The stair checkpoint initializes the actor at identical historical time offsets.
Today's flat policy provides frozen action guidance on dedicated flat samples;
inference uses one actor, with no terrain-based policy switch. Reward terrain
preview is privileged shaping, not an extra sensor in the actor observation.
"""
import sys
from rl01.train import main


if __name__ == '__main__':
    defaults = [
        '--stairs', '--omni', '--heading-obs', '--gait-phase-obs',
        '--stair-heading', '--stair-transition', '--stair-all-stages',
        '--riser-contact-clear', '--w-riser-stall', '-2.0',
        '--flat-foot-cap', '-300', '--flat-clear-speed-gain', '6', '--flat-cmd-step-prob', '.5',
        '--stair-speed-curriculum', '--stair-speed-ladder',
        '--flat-frac', '.50', '--stair-init-vmax', '1.0', '--stair-vx-max', '1.0',
        '--stair-top', '4.5', '--stair-top-steps', '100',
        '--stair-stop-prob', '.15', '--stand-prob', '.25',
        '--stair-lateral-limit', '.40', '--w-stair-y', '-6', '--w-stair-yaw', '-1.5',
        '--riser-clear-margin', '.045', '--w-diagonal', '.1',
        '--stair-front-bonus', '.08', '--stair-rear-bonus', '.06',
        '--obs-taps', '0', '1', '2', '3', '4', '6', '9', '10', '13', '18', '24',
        '--init-vmax', '2.0', '--vx-max', '2.0', '--direction-init', '.6',
        '--reverse-max', '1.0', '--vy-range', '.8', '--yaw-range', '1.2',
        '--cmd-slew-rate', '1.0', '--cmd-resample-s', '4.0',
        '--relative-tracking', '--heading-bound', '.5',
        '--w-heading', '-3', '--w-track-heading', '.8', '--heading-sigma', '.005',
        '--w-cross-vel', '-8', '--w-stop-vel', '-2', '--w-stop-ang', '-1',
        '--track-sigma', '.05', '--ang-sigma', '.08',
        '--w-track-lin', '1.5', '--w-track-ang', '1.2',
        '--maneuver-gait', '--gait-frequency-speed', '.6',
        '--w-maneuver-gait', '1.2', '--w-maneuver-trajectory', '.3',
        '--w-maneuver-height', '-250', '--w-maneuver-pose', '-.15', '--w-maneuver-slide', '-.6',
        '--speed-clear', '--w-foot-clear', '-150', '--w-foot-cap', '-120',
        '--w-orient-low', '-20', '--w-ang-vel-low', '-.4',
        '--friction-range', '.35', '1.25', '--mass-range', '-1.2', '2.0',
        '--kp-range', '.82', '1.18', '--kd-range', '.82', '1.18',
        '--damping-range', '.45', '1.25', '--com-range', '.035',
        '--push', '.65', '--push-ang', '.35', '--push-z', '.15', '--push-interval', '4.5',
        '--robustness-curriculum', '--robustness-init', '.35',
        '--flat-teacher-coef', '.10', '--sym-coef', '.5', '--entropy-coef', '.002',
        '--init-std', '.32', '--lr', '.00005', '--num-envs', '1024', '--nthread', '16',
        '--log-every', '20', '--save-every', '200',
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    main()
