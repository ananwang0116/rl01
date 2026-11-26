"""Resume a training run using its saved command arguments, without retyping flags.

uv run python tools/resume.py runs/<name>/model_200.pt --iterations 2000
Long runs must still be launched through tools/proc.sh and tools/supervise.py.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--iterations', type=int, required=True)
    for key in ('max-seconds', 'num-envs', 'nthread', 'torch-threads', 'log-every', 'save-every'):
        parser.add_argument('--' + key, type=int)
    requested = parser.parse_args()
    params = json.loads((requested.checkpoint.parent / 'config.json').read_text())['args']
    params.update(resume=str(requested.checkpoint), init_from=None, iterations=requested.iterations)
    for key, value in vars(requested).items():
        if key != 'checkpoint' and value is not None:
            params[key] = value
    cmd = [sys.executable, '-u', str(Path(__file__).resolve().parents[1] / 'src/rl01/train.py')]
    for key, value in params.items():
        if value is None or value is False:
            continue
        cmd.append('--' + key.replace('_', '-'))
        if value is not True:
            cmd.extend(map(str, value if isinstance(value, (list, tuple)) else [value]))
    return subprocess.call(cmd)


if __name__ == '__main__':
    sys.exit(main())
