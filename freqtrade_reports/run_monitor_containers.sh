#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${FREQTRADE_REPORTS_HOME:-$HOME/freqtrade_reports}"
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

run_with_optional_sudo python3 "$BASE_DIR/monitor_containers.py" "$@"
