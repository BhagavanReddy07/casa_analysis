"""Dispatch a new upload to the GPU box when it's up; run locally otherwise.

The GPU instance (``ANALYSIS_HOST``) is started and stopped by hand to control
cost, so it is routinely unreachable — that is expected, not an error state.
Every check here is built to fail fast and cheaply rather than block the
uploader waiting on a machine that is off.

Configuration is environment variables, set in the systemd unit alongside
``CASA_DASHBOARD_PASSWORD``, not in the repo — see ``ANALYSIS_HOST`` below.
Unset ``ANALYSIS_HOST`` disables this entirely and every upload runs locally,
which is also what happens on a plain development machine.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

HOST = os.environ.get("ANALYSIS_HOST", "")
USER = os.environ.get("ANALYSIS_USER", "ubuntu")
KEY_PATH = os.environ.get("ANALYSIS_KEY_PATH", "")
PORT = int(os.environ.get("ANALYSIS_PORT", "22"))
APP_DIR = os.environ.get("ANALYSIS_APP_DIR", "~/sperm_CASA")

PROBE_TIMEOUT = 3.0     # seconds — runs synchronously on every upload, must be quick
SSH_TIMEOUT = 15        # seconds for the ssh/scp connection itself
RUN_TIMEOUT = 1800      # seconds for the remote analysis to finish

# Everything a run can produce, in the order they matter: the tracked video
# always exists if the run completed at all, so its absence means the run
# itself failed. The CSV and trajectory file only exist when the detector
# found something — their absence is a legitimate empty result, not a
# transfer failure, and is handled by the caller exactly like a local run
# that found nothing.
_OUTPUT_SUFFIXES = ("_tracked.mp4", "_metrics.csv", "_trajectories.json")


def configured() -> bool:
    """Whether enough is set to even attempt a remote dispatch."""
    return bool(HOST and KEY_PATH and Path(KEY_PATH).exists())


def available() -> bool:
    """Quick reachability check — a TCP connect to the SSH port, nothing more.

    Deliberately not a full SSH handshake: a stopped EC2 instance simply does
    not answer, so a short socket timeout is enough to tell "off" from "up"
    without making an uploader wait out a long connection attempt.
    """
    if not configured():
        return False
    try:
        with socket.create_connection((HOST, PORT), timeout=PROBE_TIMEOUT):
            return True
    except OSError:
        return False


def _ssh(*command: str) -> list[str]:
    return ["ssh", "-i", KEY_PATH, "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={SSH_TIMEOUT}",
            "-o", "StrictHostKeyChecking=accept-new",
            "-p", str(PORT), f"{USER}@{HOST}", *command]


def _scp(source: str, destination: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["scp", "-i", KEY_PATH, "-o", "BatchMode=yes",
         "-o", f"ConnectTimeout={SSH_TIMEOUT}",
         "-o", "StrictHostKeyChecking=accept-new", "-P", str(PORT),
         source, destination],
        capture_output=True, timeout=timeout)


def run_remote(source: Path, min_track_length: int) -> bool:
    """Upload, analyse on the GPU box, and pull the results back.

    Only ``min_track_length`` is forwarded — it is the one tracking setting
    exposed in the sidebar. Everything else (match_thresh, motion_gate, the
    history-based re-acquisition, ...) is left to the remote box's own
    ``TrackerConfig()`` defaults rather than threaded through the CLI, so a
    tuning change only ever has to be deployed once, not kept in sync between
    this dispatch code and the pipeline it calls.

    Returns True on success — remote analysis ran and outputs were copied
    back, whether or not the detector found anything. Returns False for any
    infrastructure failure (transfer, ssh, timeout), which tells the caller
    to fall back to a local run instead of losing the upload entirely.
    """
    remote_video = f"{APP_DIR}/videos/input/{source.name}"
    try:
        upload = _scp(str(source), f"{USER}@{HOST}:{remote_video}", timeout=120)
        if upload.returncode != 0:
            logger.warning("could not upload %s to the analysis box: %s",
                           source.name, upload.stderr.decode(errors="replace"))
            return False

        analysis = subprocess.run(
            _ssh(f"cd {APP_DIR} && .venv/bin/python main.py "
                 f"--source videos/input/{source.name} --track --metrics "
                 f"--min-track-len {min_track_length}"),
            capture_output=True, timeout=RUN_TIMEOUT)
        if analysis.returncode != 0:
            logger.warning("remote analysis of %s failed: %s",
                           source.name, analysis.stderr.decode(errors="replace"))
            return False

        stem = source.stem
        tracked_video_ok = False
        for suffix in _OUTPUT_SUFFIXES:
            remote_path = f"{APP_DIR}/videos/output/{stem}{suffix}"
            local_path = Path("videos/output") / f"{stem}{suffix}"
            result = _scp(f"{USER}@{HOST}:{remote_path}", str(local_path), timeout=60)
            if suffix == "_tracked.mp4":
                tracked_video_ok = result.returncode == 0

        if not tracked_video_ok:
            logger.warning("remote run for %s produced no tracked video", source.name)
            return False
        return True

    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("remote analysis of %s could not complete, "
                       "falling back to local: %s", source.name, exc)
        return False


if __name__ == "__main__":
    # ponytail: one self-check — configured()/available() must degrade to
    # False, not raise, when the environment is unset. That is the default
    # state on a plain dev machine and on any host that has never set
    # ANALYSIS_HOST, so it has to be silent and safe, not an error.
    import unittest.mock as mock

    with mock.patch.dict(os.environ, {}, clear=True):
        import importlib
        import utils.remote_analysis as self_module
        importlib.reload(self_module)
        assert not self_module.configured(), "configured() should be False with no env set"
        assert not self_module.available(), "available() should be False, not raise, when unconfigured"

    with mock.patch.dict(os.environ, {"ANALYSIS_HOST": "127.0.0.1", "ANALYSIS_PORT": "1",
                                      "ANALYSIS_KEY_PATH": __file__}, clear=True):
        importlib.reload(self_module)
        assert self_module.configured(), "configured() should be True once host+key are set"
        # port 1 on localhost: nothing listens there, so this exercises the
        # real refusal path rather than a mock.
        assert not self_module.available(), "available() should be False against a closed port"

    print("remote_analysis.py self-check passed")
