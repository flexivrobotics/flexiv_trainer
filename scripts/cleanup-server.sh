#!/usr/bin/env bash
set -euo pipefail

collect_pids() {
    pgrep -f '/flexiv-trainer-server|/flexivtrainer-server| -m flexivtrainer($| )' | sort -u || true
}

mapfile -t pids < <(collect_pids)
if [[ ${#pids[@]} -eq 0 ]]; then
    echo "No lingering Flexiv Trainer server processes found."
    exit 0
fi

for pid in "${pids[@]}"; do
    [[ -n "${pid}" ]] || continue
    cmdline=""
    if [[ -r "/proc/${pid}/cmdline" ]]; then
        cmdline="$(tr '\0' ' ' </proc/"${pid}"/cmdline)"
    fi
    echo "Stopping Flexiv Trainer server (pid ${pid}) ${cmdline}"
    kill -TERM "${pid}" 2>/dev/null || true
done

sleep 1

mapfile -t remaining < <(collect_pids)
for pid in "${remaining[@]}"; do
    [[ -n "${pid}" ]] || continue
    cmdline=""
    if [[ -r "/proc/${pid}/cmdline" ]]; then
        cmdline="$(tr '\0' ' ' </proc/"${pid}"/cmdline)"
    fi
    echo "Force-stopping Flexiv Trainer server (pid ${pid}) ${cmdline}"
    kill -KILL "${pid}" 2>/dev/null || true
done
