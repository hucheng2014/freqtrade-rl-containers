#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPORTS_HOME = Path(
    os.environ.get('FREQTRADE_REPORTS_HOME', str(Path.home() / 'freqtrade_reports'))
)
OUTPUT_JSON = REPORTS_HOME / 'output' / 'latest_metrics.json'
DECISION_JSON = REPORTS_HOME / 'output' / 'leverage_policy_latest.json'
DECISION_TEXT = REPORTS_HOME / 'logs' / 'leverage_policy_last.txt'
MAX_LEVERAGE = 3
MIN_LEVERAGE = 1
IDENTIFIER_SUFFIX_RE = re.compile(r'_lev\d+x(?:_\d{8}_\d{6})?$')
DOCKER_CMD = shlex.split(os.environ.get('FREQTRADE_DOCKER_CMD', 'docker'))
DEPLOY_ROOT_CANDIDATES = [
    Path(os.environ['FREQTRADE_DEPLOY_ROOT'])
    for _ in [0]
    if os.environ.get('FREQTRADE_DEPLOY_ROOT')
]
DEPLOY_ROOT_CANDIDATES.extend([
    Path.home() / '下载' / 'shipan',
    Path('/opt'),
])

CONTAINER_MAP = {
    'RL_SignalFilter_LIVE': {
        'root_name': 'freqtrade_rl_live',
        'config_name': 'config_rl_live.json',
        'container': 'RL_SignalFilter_LIVE',
    },
    'RL_SelfEvolve_DRY': {
        'root_name': 'freqtrade_rl_selfevolve',
        'config_name': 'config_selfevolve.json',
        'container': 'RL_SelfEvolve_DRY',
    },
    'DogeAI_NoT3_RL_WithBTC_LIVE': {
        'root_name': 'freqtrade_dogeai_not3_rl_live',
        'config_name': 'config.json',
        'container': 'DogeAI_NoT3_RL_WithBTC_LIVE',
    },
}


def resolve_container_root(root_name: str) -> Path:
    for base in DEPLOY_ROOT_CANDIDATES:
        candidate = base / root_name
        if candidate.exists():
            return candidate
    return DEPLOY_ROOT_CANDIDATES[0] / root_name


def resolve_config_path(meta: dict) -> Path:
    return resolve_container_root(meta['root_name']) / meta['config_name']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Apply leverage policy from daily Kelly/Sharpe metrics.')
    parser.add_argument('--metrics-json', default=str(OUTPUT_JSON))
    parser.add_argument('--min-trades', type=int, default=int(os.getenv('MIN_TRADES', '20')))
    parser.add_argument('--telegram-token', default=os.getenv('TELEGRAM_BOT_TOKEN', ''))
    parser.add_argument('--telegram-chat-id', default=os.getenv('TELEGRAM_CHAT_ID', ''))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-telegram', action='store_true')
    return parser.parse_args()


def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    if not token or not chat_id:
        return
    payload = urllib.parse.urlencode({'chat_id': chat_id, 'text': text}).encode('utf-8')
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    request = urllib.request.Request(url, data=payload, method='POST')
    with urllib.request.urlopen(request, timeout=20) as response:
        response.read()


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def current_leverage_from_config(cfg: dict) -> int:
    value = (cfg.get('exchange') or {}).get('leverage', 1)
    try:
        return clamp(int(round(float(value))), MIN_LEVERAGE, MAX_LEVERAGE)
    except Exception:
        return MIN_LEVERAGE


def current_identifier_from_config(cfg: dict, container_name: str) -> str:
    freqai = cfg.get('freqai') or {}
    identifier = str(freqai.get('identifier') or '').strip()
    if identifier:
        return identifier
    fallback = container_name.lower().replace('-', '_')
    return fallback


def next_identifier(current_identifier: str, target_leverage: int, now_utc: datetime) -> str:
    base = IDENTIFIER_SUFFIX_RE.sub('', current_identifier)
    stamp = now_utc.strftime('%Y%m%d_%H%M%S')
    return f'{base}_lev{target_leverage}x_{stamp}'


def model_root_from_config_path(cfg_path: Path) -> Path:
    return cfg_path.parent / 'user_data' / 'models'


def unique_backup_path(path: Path, suffix: str) -> Path:
    candidate = path.with_name(path.name + suffix)
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        retry = path.with_name(path.name + suffix + f'_{index}')
        if not retry.exists():
            return retry
        index += 1


def target_from_metric(metric: dict) -> int:
    raw = metric.get('leverage_recommended_actionable')
    if raw is None:
        return MIN_LEVERAGE
    try:
        return clamp(int(math.floor(float(raw))), MIN_LEVERAGE, MAX_LEVERAGE)
    except Exception:
        return MIN_LEVERAGE


def experience_root_from_config_path(cfg_path: Path) -> Path:
    return cfg_path.parent / 'user_data'


def experience_dir_for_leverage(cfg_path: Path, leverage: int) -> Path:
    return experience_root_from_config_path(cfg_path) / f'experience_replay_{int(leverage)}x'


def seed_model_metadata(source_dir: Path, target_dir: Path) -> list[str]:
    copied: list[str] = []
    if not source_dir.exists():
        return copied
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        'global_metadata.json',
        'historic_predictions.pkl',
        'historic_predictions.backup.pkl',
        'run_params.json',
    ):
        src = source_dir / name
        dst = target_dir / name
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())
            copied.append(name)
    pair_dictionary = target_dir / 'pair_dictionary.json'
    if not pair_dictionary.exists():
        pair_dictionary.write_text('{}\n', encoding='utf-8')
        copied.append('pair_dictionary.json(empty)')
    return copied


def _convert_experience_record(record: dict, source_leverage: int, target_leverage: int) -> dict:
    out = dict(record)
    try:
        stored_leverage = float(out.get('leverage') or source_leverage or 1.0)
    except Exception:
        stored_leverage = float(source_leverage or 1.0)
    if stored_leverage <= 0:
        stored_leverage = float(source_leverage or 1.0)
    try:
        profit_ratio = float(out.get('profit_ratio') or 0.0)
    except Exception:
        profit_ratio = 0.0
    price_change = profit_ratio / stored_leverage if stored_leverage else profit_ratio
    out['profit_ratio'] = price_change * float(target_leverage)
    out['leverage'] = float(target_leverage)
    out['migrated_from_leverage'] = stored_leverage
    return out


def _experience_record_key(record: dict) -> tuple:
    return (
        str(record.get('trade_id') or ''),
        str(record.get('pair') or ''),
        str(record.get('entry_time') or ''),
        str(record.get('exit_time') or ''),
    )


def _experience_sort_key(record: dict) -> tuple:
    return (
        str(record.get('recorded_at') or record.get('exit_time') or record.get('entry_time') or ''),
        _experience_record_key(record),
    )


def load_experience_records(path: Path) -> list[dict] | None:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    return [item for item in payload if isinstance(item, dict)]


def backup_corrupt_experience_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    backup = path.with_name(f'{path.name}.corrupt_{stamp}')
    path.replace(backup)
    return backup


def merge_experience_records(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for record in existing:
        merged[_experience_record_key(record)] = record
    for record in incoming:
        merged[_experience_record_key(record)] = record
    return sorted(merged.values(), key=_experience_sort_key)


def seed_missing_experience_files(source_dir: Path, target_dir: Path, source_leverage: int, target_leverage: int) -> list[str]:
    copied: list[str] = []
    if not source_dir.exists() or source_leverage == target_leverage:
        return copied
    target_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(source_dir.glob('*.json')):
        dst = target_dir / src.name
        records = load_experience_records(src)
        if records is None:
            continue
        converted = [
            _convert_experience_record(r, source_leverage, target_leverage)
            for r in records if isinstance(r, dict)
        ]
        if not converted:
            continue

        existing: list[dict] = []
        repaired = False
        if dst.exists():
            existing_records = load_experience_records(dst)
            if existing_records is None:
                backup_corrupt_experience_file(dst)
                repaired = True
            else:
                existing = existing_records

        merged = merge_experience_records(existing, converted)
        if existing and len(merged) == len(existing):
            continue

        dst.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        copied.append(f'{src.name}(repaired)' if repaired else src.name)
    return copied

def decide_new_leverage(current: int, desired: int) -> int:
    if desired > current:
        return min(current + 1, desired, MAX_LEVERAGE)
    if desired < current:
        return max(desired, MIN_LEVERAGE)
    return current


def apply_one(metric: dict, min_trades: int, dry_run: bool) -> dict:
    name = metric['container']
    cfg_path = resolve_config_path(CONTAINER_MAP[name])
    container = CONTAINER_MAP[name]['container']
    cfg = load_json(cfg_path)
    current = current_leverage_from_config(cfg)
    current_identifier = current_identifier_from_config(cfg, name)
    desired = target_from_metric(metric)
    closed_trades = int(metric.get('closed_trades') or 0)
    gate_blocked = closed_trades < min_trades
    reason = []
    if gate_blocked:
        desired_after_gate = current
        reason.append(f'closed_trades<{min_trades}')
    else:
        desired_after_gate = desired
        reason.append(f'recommended_floor={desired}')
    target = decide_new_leverage(current, desired_after_gate)
    leverage_changed = target != current
    if leverage_changed and desired_after_gate > current and target < desired_after_gate:
        reason.append('daily_step_limit_up=1x')
    now_utc = datetime.now(timezone.utc)
    next_freqai_identifier = current_identifier
    if leverage_changed:
        next_freqai_identifier = next_identifier(current_identifier, target, now_utc)
        reason.append('leverage_change_forces_full_retrain')
    margin_mode_before = cfg.get('margin_mode', 'cross')
    margin_mode_after = 'isolated'
    changed = leverage_changed or (margin_mode_before != margin_mode_after)

    backup_path = None
    model_backup_path = None
    target_identifier_backup_path = None
    seeded_model_metadata = []
    seeded_experience_files = []
    if changed and not dry_run:
        backup_path = str(cfg_path) + f".bak_autolev_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        Path(backup_path).write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        model_root = model_root_from_config_path(cfg_path)
        model_root.mkdir(parents=True, exist_ok=True)
        if leverage_changed:
            active_model_dir = model_root / current_identifier
            backup_dir = None
            if active_model_dir.exists():
                backup_dir = unique_backup_path(
                    active_model_dir,
                    f".bak_autolev_{now_utc.strftime('%Y%m%d_%H%M%S')}_from{current}x_to{target}x",
                )
                active_model_dir.rename(backup_dir)
                model_backup_path = str(backup_dir)
            target_model_dir = model_root / next_freqai_identifier
            if target_model_dir.exists():
                target_backup_dir = unique_backup_path(
                    target_model_dir,
                    f".bak_autolev_target_{now_utc.strftime('%Y%m%d_%H%M%S')}",
                )
                target_model_dir.rename(target_backup_dir)
                target_identifier_backup_path = str(target_backup_dir)
            if backup_dir is not None:
                seeded_model_metadata = seed_model_metadata(backup_dir, target_model_dir)
                seeded_experience_files = seed_missing_experience_files(
                    experience_dir_for_leverage(cfg_path, current),
                    experience_dir_for_leverage(cfg_path, target),
                    current,
                    target,
                )
        cfg['margin_mode'] = margin_mode_after
        cfg.setdefault('exchange', {})['leverage'] = target
        if leverage_changed:
            cfg.setdefault('freqai', {})['identifier'] = next_freqai_identifier
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        subprocess.run([*DOCKER_CMD, 'restart', container], check=True, capture_output=True, text=True, timeout=120)

    return {
        'container': name,
        'config_path': str(cfg_path),
        'closed_trades': closed_trades,
        'current_leverage_before': current,
        'recommended_actionable_raw': metric.get('leverage_recommended_actionable'),
        'desired_integer_capped': desired,
        'target_leverage_after_policy': target,
        'current_identifier_before': current_identifier,
        'target_identifier_after_policy': next_freqai_identifier,
        'leverage_changed': leverage_changed,
        'margin_mode_before': margin_mode_before,
        'margin_mode_after': margin_mode_after,
        'changed': changed,
        'restarted': changed and not dry_run,
        'dry_run': dry_run,
        'reason': reason,
        'backup_path': backup_path,
        'model_backup_path': model_backup_path,
        'target_identifier_backup_path': target_identifier_backup_path,
        'seeded_model_metadata': seeded_model_metadata,
        'seeded_experience_files': seeded_experience_files,
        'note': metric.get('note'),
    }


def render_summary(results: list[dict], min_trades: int, dry_run: bool) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    lines = []
    lines.append('Leverage Policy Update')
    lines.append(f'Time: {now} UTC')
    lines.append(f'Rules: min_trades>={min_trades}, up max step=1x/day, down direct, cap=3x, new trades only')
    if dry_run:
        lines.append('Mode: dry-run')
    lines.append('')
    for item in results:
        lines.append(f"[{item['container']}]")
        lines.append(
            f"closed={item['closed_trades']} | before={item['current_leverage_before']}x | rec={item['recommended_actionable_raw']} | int_target={item['desired_integer_capped']}x | apply={item['target_leverage_after_policy']}x"
        )
        lines.append(
            f"identifier: {item['current_identifier_before']} -> {item['target_identifier_after_policy']}"
        )
        lines.append(
            f"margin: {item['margin_mode_before']} -> {item['margin_mode_after']} | changed={item['changed']} | restarted={item['restarted']}"
        )
        lines.append('reason: ' + ', '.join(item['reason']))
        if item.get('model_backup_path'):
            lines.append(f"model_backup: {item['model_backup_path']}")
        if item.get('target_identifier_backup_path'):
            lines.append(f"target_model_backup: {item['target_identifier_backup_path']}")
        if item.get('seeded_model_metadata'):
            lines.append('seeded_model_metadata: ' + ', '.join(item['seeded_model_metadata']))
        if item.get('seeded_experience_files'):
            lines.append('seeded_experience_files: ' + ', '.join(item['seeded_experience_files']))
        if item.get('note'):
            lines.append(f"note: {item['note']}")
        lines.append('')
    return '\n'.join(lines).strip() + '\n'


def main() -> int:
    args = parse_args()
    metrics = load_json(Path(args.metrics_json))
    results = [apply_one(metric, args.min_trades, args.dry_run) for metric in metrics]
    summary = render_summary(results, args.min_trades, args.dry_run)
    print(summary, end='')
    save_json(DECISION_JSON, results)
    DECISION_TEXT.parent.mkdir(parents=True, exist_ok=True)
    DECISION_TEXT.write_text(summary, encoding='utf-8')
    if args.telegram_token and args.telegram_chat_id and not args.skip_telegram:
        try:
            send_telegram_message(args.telegram_token, args.telegram_chat_id, summary)
        except Exception as exc:
            print(f'Telegram send failed: {exc}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
