#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${FREQTRADE_REPORTS_HOME:-$HOME/freqtrade_reports}"
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"

export FREQTRADE_REPORTS_HOME="$BASE_DIR"
export FREQTRADE_DEPLOY_ROOT="${FREQTRADE_DEPLOY_ROOT:-$HOME/下载/shipan}"

status=0
if REPORT="$("$BASE_DIR/run_monitor_containers.sh" 2>&1)"; then
  :
else
  status=$?
fi

timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
if [[ $status -ne 0 ]]; then
  REPORT=$'Freqtrade 容器详细监控定时任务失败\n'"时间：${timestamp}"$'\n'"退出码：${status}"$'\n\n'"${REPORT}"
fi

printf '%s\n' "$REPORT" > "$LOG_DIR/monitor_containers_last_cron.txt"

sender_args=(--title "Freqtrade RL 容器监控报告")
if [[ "${SEND_DRY_RUN:-0}" == "1" ]]; then
  sender_args+=(--dry-run)
fi

if ! printf '%s\n' "$REPORT" | python3 "$BASE_DIR/send_report_openclaw.py" "${sender_args[@]}"; then
  printf '%s\n' "$REPORT" > "$LOG_DIR/monitor_containers_last_send_failed.txt"
  exit 1
fi

exit "$status"
