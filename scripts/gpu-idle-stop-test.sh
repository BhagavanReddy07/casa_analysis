#!/bin/sh
# Self-check for gpu-idle-stop.sh. Runs anywhere with sh + sed — no GPU, no
# EC2, no root — by rewriting the real script into a harness copy: every busy
# probe forced to one answer, the counter moved to a temp file, and the
# shutdown replaced with an echo. Testing the actual file rather than a
# reimplementation is the whole point; the thing that must never regress is
# "counts up while idle, resets on any activity, stops only at the limit".
set -eu

SRC="$(dirname "$0")/gpu-idle-stop.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# $1: "idle" or "busy" — what every probe in busy() should report.
harness() {
    if [ "$1" = "busy" ]; then verdict="true"; else verdict="false"; fi
    sed -e "s#^COUNT=.*#COUNT=$TMP/count#" \
        -e "s#^IDLE_LIMIT=.*#IDLE_LIMIT=3#" \
        -e "s#^    \[ -n .*ss -H.*#    $verdict \&\& return 0#" \
        -e "s#^    pgrep .*#    $verdict \&\& return 0#" \
        -e "s#^    \[ -n .*nvidia-smi.*#    $verdict \&\& return 0#" \
        -e "s#^logger .*#:#" \
        -e "s#^exec shutdown -h now#echo STOP#" \
        "$SRC" > "$TMP/h.sh"
    sh "$TMP/h.sh"
}

# Idle samples accumulate, and nothing happens before the limit.
[ "$(harness idle)" = "" ] && [ "$(cat "$TMP/count")" = "1" ] || { echo "FAIL: first idle sample"; exit 1; }
[ "$(harness idle)" = "" ] && [ "$(cat "$TMP/count")" = "2" ] || { echo "FAIL: second idle sample"; exit 1; }

# One sign of life anywhere clears the count — this is what stops a run being
# killed between the upload and the analysis that follows it.
[ "$(harness busy)" = "" ] && [ "$(cat "$TMP/count")" = "0" ] || { echo "FAIL: busy did not reset"; exit 1; }

# Only a full run of consecutive idle samples stops the box.
harness idle >/dev/null; harness idle >/dev/null
[ "$(harness idle)" = "STOP" ] || { echo "FAIL: did not stop at the limit"; exit 1; }

echo "gpu-idle-stop.sh self-check passed"
