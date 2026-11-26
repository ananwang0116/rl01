"""Run one owned child with an output watchdog and a wall deadline.

Launch through tools/proc.sh. The child stays in that registered process group.
Training handles TERM/INT at an iteration boundary and saves before exit.
"""
import argparse
import os
import selectors
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--silence', type=float, default=180)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('missing child command')
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, bufsize=0)
    started = last_output = last_probe = time.monotonic()
    stopping = None
    reason = None

    def stop(signum, frame):
        nonlocal stopping, reason
        if stopping is None:
            stopping = time.monotonic()
            reason = f'signal {signum}'
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    selector = selectors.DefaultSelector()
    selector.register(child.stdout, selectors.EVENT_READ)
    print(f'supervisor_started pid={os.getpid()} child={child.pid} deadline_s={args.deadline}', flush=True)
    try:
        while True:
            events = selector.select(timeout=1)
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 65536)
                if chunk:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.flush()
                    last_output = time.monotonic()
                else:
                    selector.unregister(key.fileobj)
            now = time.monotonic()
            code = child.poll()
            if code is not None and not selector.get_map():
                print(f'supervisor_complete child_exit={code} elapsed_s={now-started:.1f} reason={reason}', flush=True)
                return code if reason is None else (code or 2)
            if now - last_probe >= 30:
                print(f'supervisor_health child={child.pid} alive={code is None} elapsed_s={now-started:.1f} silence_s={now-last_output:.1f}', flush=True)
                last_probe = now
            if stopping is None and (now - started >= args.deadline or now - last_output >= args.silence):
                reason = 'deadline' if now - started >= args.deadline else 'output watchdog'
                print(f'supervisor_stopping reason={reason}; requesting checkpoint and clean exit', flush=True)
                stopping = now
                child.terminate()
            if stopping is not None and now - stopping > 45:
                print(f'Error: child={child.pid} did not exit within 45s; registered group needs inspection', flush=True)
                return 3
    finally:
        selector.close()
        child.stdout.close()


if __name__ == '__main__':
    sys.exit(main())
