"""专项训练后退姿态、行进转弯与原地交替踏步，继续使用统一速度策略。

CHAMP reference supplies diagonal phases and translation+yaw foot geometry;
PPO still learns torque-limited joint actions in randomized MuJoCo dynamics.
Forward speed-dependent foot clearance is retained. This is a training entry,
not a second inference policy and not a ROS/hardware controller.

uv run python src/rl01/train_maneuver.py --name go2_maneuver_v1 \
    --init-from runs/go2_vel2_v6b/model_300.pt --iterations 1800
Long jobs: launch through tools/proc.sh and tools/supervise.py.
All defaults may be overridden by explicit options below.
"""
import sys

from rl01.train import main


if __name__ == '__main__':
    defaults = [
        '--omni', '--heading-obs', '--relative-tracking', '--heading-bound','0.5',
        '--maneuver-gait','--gait-phase-obs','--turn-focus',
        '--direction-init','0.4','--yaw-range','1.2','--stand-prob','0.25',
        '--w-heading','-1.5','--w-cross-vel','-2','--w-stop-vel','-2','--w-stop-ang','-1',
        '--w-track-lin','2','--w-track-ang','1',
        '--w-maneuver-gait','1.2','--w-maneuver-trajectory','0.3',
        '--w-maneuver-height','-250','--w-maneuver-pose','-0.15','--w-maneuver-slide','-0.6',
        '--speed-clear','--w-foot-clear','-150','--w-orient-low','-20','--w-ang-vel-low','-0.4',
        '--sym-coef','0.5','--lr','0.0003','--entropy-coef','0.003','--init-std','0.5',
        '--push','0.6','--push-ang','0.2','--push-z','0.1','--push-interval','4.5',
        '--friction-range','0.4','1.2','--mass-range','-1','1.5',
        '--kp-range','0.9','1.1','--kd-range','0.9','1.1',
        '--damping-range','0.75','1.2','--com-range','0.02','--cmd-resample-s','4',
        '--num-envs','512','--nthread','8','--log-every','20','--save-every','300',
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    main()
