#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${FREQTRADE_REPORTS_HOME:-$HOME/freqtrade_reports}"
LOG_DIR="$BASE_DIR/logs"
OUTPUT_DIR="$BASE_DIR/output"
ENV_FILE="$BASE_DIR/telegram.env"

export FREQTRADE_REPORTS_HOME="$BASE_DIR"
export FREQTRADE_DEPLOY_ROOT="${FREQTRADE_DEPLOY_ROOT:-$HOME/下载/shipan}"

run_with_optional_sudo() {
  local sudo_stderr
  if sudo -n true >/dev/null 2>&1; then
    sudo -n -E "$@"
  elif [[ -n "${FREQTRADE_SUDO_PASSWORD:-}" ]]; then
    sudo_stderr="$(mktemp)"
    if ! printf '%s\n' "$FREQTRADE_SUDO_PASSWORD" | sudo -S -p '' -E "$@" 2>"$sudo_stderr"; then
      grep -v '^\[sudo\]' "$sudo_stderr" >&2 || true
      rm -f "$sudo_stderr"
      return 1
    fi
    grep -v '^\[sudo\]' "$sudo_stderr" >&2 || true
    rm -f "$sudo_stderr"
  else
    "$@"
  fi
}

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

run_with_optional_sudo python3 "$BASE_DIR/daily_metrics_report.py" \
  --lookback-hours "${ROLLING_LOOKBACK_HOURS:-336}" \
  --bucket-hours "${ROLLING_BUCKET_HOURS:-4}" \
  --recent-buckets "${ROLLING_RECENT_BUCKETS:-6}" \
  --risk-free-annual "${RISK_FREE_ANNUAL:-0.0}" \
  --min-trades "${MIN_TRADES:-20}" \
  --json-output "$OUTPUT_DIR/rolling_metrics_latest.json" \
  | tee "$LOG_DIR/rolling_report.txt"
