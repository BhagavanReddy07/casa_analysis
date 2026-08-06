#!/bin/sh
# Stop the GPU box when nothing has used it for a while.
#
# On-Demand bills every second the instance is `running`, idle or not — a T4
# left on overnight costs the same as one at 100%. This is the backstop for
# "nobody remembered to stop it", which cost a full day's charge on
# 2026-08-05 for three runs totalling 75 seconds of GPU time.
#
# Run from gpu-idle-stop.timer every IDLE_STEP_MIN minutes. No AWS
# credentials needed: `shutdown -h now` on an EBS-backed instance whose
# InstanceInitiatedShutdownBehavior is `stop` (the default) stops it.
#
# ponytail: a counter file, not a duration query. Sampling every 5 minutes
# cannot see a 30-second lull, which is the point — it never stops the box
# between the scp and the run that follows it.
set -eu

IDLE_STEP_MIN=5     # must match OnUnitActiveSec in the timer
IDLE_LIMIT=6        # consecutive idle samples before stopping => 30 min
COUNT=/run/gpu-idle-stop.count   # /run is tmpfs: a reboot starts the count over

busy() {
    # Any established connection to sshd. This is the widest signal and the
    # only one that covers a whole dispatch: utils/remote_analysis.py holds an
    # SSH or SCP connection through *all three* phases — upload, run, and
    # pulling a result back that can be 145 MB. GPU utilisation alone does
    # not: pass 2 of inference.py re-encodes the annotated video on CPU with
    # the GPU at 0%, and killing the box mid-encode would look exactly like
    # yesterday's failure.
    [ -n "$(ss -H -tn state established 'sport = :22' 2>/dev/null)" ] && return 0

    # A run left going by hand — `main.py --rebuild` from a deploy detaches and
    # outlives the SSH session that started it, so it needs its own check.
    pgrep -f '[m]ain\.py' >/dev/null 2>&1 && return 0

    # Anything holding the GPU, whoever started it.
    [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 0

    return 1
}

if busy; then
    echo 0 > "$COUNT"
    exit 0
fi

n=$(( $(cat "$COUNT" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$COUNT"
[ "$n" -ge "$IDLE_LIMIT" ] || exit 0

logger -t gpu-idle-stop "idle $(( n * IDLE_STEP_MIN )) min — stopping instance"
exec shutdown -h now
