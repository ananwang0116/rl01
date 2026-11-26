"""Sequential bounded stair speed/stage checks using stair_transition.py."""
import argparse
import json
from pathlib import Path
import sys
import time

from stair_transition import main as evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', nargs='+')
    p.add_argument('--stages', type=int, nargs='+', default=[8, 11])
    p.add_argument('--speeds', type=float, nargs='+', default=[.3, .5, .8, 1.])
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--seconds', type=float, default=35)
    p.add_argument('--wall-seconds', type=float, default=240)
    p.add_argument('--profile', default='clean', choices=['clean', 'training', 'stress'])
    p.add_argument('--sampled', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    deadline = time.monotonic()+args.wall_seconds
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(args.stages)*len(args.speeds)
    for stage in args.stages:
        for speed in args.speeds:
            remaining = deadline-time.monotonic()
            if remaining < 5:
                break
            folder = args.out/f's{stage}_v{speed:g}'
            sys.argv = ['stair_transition.py', *args.checkpoint, '--stage', str(stage), '--vx', str(speed),
                        '--num-envs', str(args.num_envs), '--seconds', str(args.seconds),
                        '--wall-seconds', str(remaining), '--profile', args.profile,
                        '--case', 'continuous', 'stop_restart', '--no-frames', '--out', str(folder)]
            if args.sampled:
                sys.argv.append('--sampled')
            if args.profile == 'stress':
                sys.argv += ['--push', '.65', '--push-ang', '.35', '--push-z', '.15']
            evaluate()
            data = json.loads((folder/'summary.json').read_text())
            results.extend(data['results'])
            count = len(results)
            complete = count == total*len(args.checkpoint)*2 and all(r['status']=='completed' for r in results)
            (args.out/'summary.json').write_text(json.dumps({'complete': complete, 'results': results}, indent=2)+'\n')
            print(f'suite_progress stage={stage} speed={speed:g} cases={count}/{total*len(args.checkpoint)*2}', flush=True)


if __name__ == '__main__':
    main()
