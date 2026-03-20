#!/usr/bin/env python3
"""
Freqtrade RL 容器监控报告脚本
- 直接读取宿主机 freqtrade.log，避免 docker logs --tail 造成漏报
- 报告最新推理耗时、最新训练 reward、容器运行时长、近 24h process died 次数
- 明确显示数据时间戳与新鲜度，避免“无数据/仅训练容器”误导
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import base64
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import sqlite3

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


LOCAL_TZ = ZoneInfo("Asia/Shanghai") if ZoneInfo else timezone(timedelta(hours=8))
UTC = timezone.utc
INFERENCE_STALE_MINUTES = 20
REWARD_GRACE_RATIO = 1.5
EXPERIENCE_SKEW_MINUTES = 10
RECENT_CLOSED_EXPERIENCE_MATCH_MINUTES = 30

TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - ")
INF_RE = re.compile(r"Total time spent inferencing pairlist (?P<value>[\d.]+) seconds")
REWARD_RE = re.compile(r"episode_reward=(?P<mean>-?[\d.]+) \+/- (?P<std>[\d.]+)")
ADAPTIVE_REWARD_RE = re.compile(r"\[MyTicketRL_v2\] Adaptive reward params:")
PERF_ADJUST_RE = re.compile(r"\[PerformanceTracker\] Adjusted params:")
TRAIN_PAIRLIST_RE = re.compile(r"Total time spent training pairlist (?P<value>[\d.]+) seconds")
TRAIN_COMPLETE_RE = re.compile(r"\[MyTicketRL_v2\] Training complete")
PROCESS_DIED_RE = re.compile(r"process died")
DOCKER_TS_RE = re.compile(r"^(?P<ts>\S+)\s+")
EXPERIENCE_DIR_RE = re.compile(r"^experience_replay_(?P<tag>.+)$")
VALID_LEVERAGE_TAG_RE = re.compile(r"^\d+(?:p\d+)?x$")
FLOAT_ASSIGN_RE = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?:#.*)?$")
STOPLOSS_AUDIT_TOLERANCE = 0.005
DOCKER_CMD = shlex.split(os.environ.get("FREQTRADE_DOCKER_CMD", "docker"))
DEPLOY_ROOT_CANDIDATES = [
    Path(os.environ["FREQTRADE_DEPLOY_ROOT"])
    for _ in [0]
    if os.environ.get("FREQTRADE_DEPLOY_ROOT")
]
DEPLOY_ROOT_CANDIDATES.extend([
    Path.home() / "下载" / "shipan",
    Path("/opt"),
])


@dataclass
class Point:
    ts: Optional[datetime]
    value: Optional[float]
    extra: Optional[float] = None
    label: str = ""
    kind: str = "numeric"


@dataclass
class Freshness:
    ts: Optional[datetime]
    count: int
    label: str
    name: str = ""
    bad_count: int = 0


@dataclass
class ExperienceAudit:
    current_tag: str
    current_info: Freshness
    issues: list[str]
    notes: list[str]


@dataclass
class StoplossAudit:
    checked_count: int
    label: str
    detail: str
    issues: list[str]
    notes: list[str]


CONTAINERS = [
    {
        "name": "DogeAI_NoT3_RL_WithBTC_LIVE",
        "display": "DogeAI_NoT3_RL_WithBTC_LIVE",
        "root_name": "freqtrade_dogeai_not3_rl_live",
        "config_name": "config.json",
        "db_relative_paths": ["user_data/tradesv3.sqlite"],
    },
    {
        "name": "RL_SelfEvolve_DRY",
        "display": "RL_SelfEvolve_DRY",
        "root_name": "freqtrade_rl_selfevolve",
        "config_name": "config_selfevolve.json",
        "db_relative_paths": [
            "user_data/tradesv3.sqlite",
            "user_data/tradesv3_dry.sqlite",
        ],
    },
    {
        "name": "RL_SignalFilter_LIVE",
        "display": "RL_SignalFilter_LIVE",
        "root_name": "freqtrade_rl_live",
        "config_name": "config_rl_live.json",
        "db_relative_paths": ["user_data/tradesv3.sqlite"],
    },
]


def resolve_container_root(root_name: str) -> Path:
    for base in DEPLOY_ROOT_CANDIDATES:
        candidate = base / root_name
        if candidate.exists():
            return candidate
    return DEPLOY_ROOT_CANDIDATES[0] / root_name


def read_config(config_path: str) -> dict:
    try:
        return json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_container_runtime(container_name: str) -> tuple[str, Optional[datetime]]:
    cmd = [
        *DOCKER_CMD,
        "inspect",
        container_name,
        "--format",
        "{{.State.Status}}|{{.State.StartedAt}}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True)
        raw = result.stdout.strip()
        status, started_at = raw.split("|", 1)
        started_dt = None
        if started_at and started_at != "0001-01-01T00:00:00Z":
            started_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00")).astimezone(UTC)
        return status, started_dt
    except Exception:
        return "unknown", None


def fmt_age(ts: Optional[datetime], now_utc: datetime) -> str:
    if not ts:
        return "无"
    delta = now_utc - ts
    total_minutes = max(int(delta.total_seconds() // 60), 0)
    if total_minutes < 60:
        return f"{total_minutes}m"
    hours, minutes = divmod(total_minutes, 60)
    if hours < 48:
        return f"{hours}h{minutes:02d}m"
    days, rem_hours = divmod(hours, 24)
    return f"{days}d{rem_hours}h"


def fmt_ts(ts: Optional[datetime]) -> str:
    if not ts:
        return "无"
    return ts.astimezone(LOCAL_TZ).strftime("%m-%d %H:%M")


def fmt_runtime(started_at: Optional[datetime], now_utc: datetime) -> str:
    if not started_at:
        return "未知"
    delta = now_utc - started_at
    total_minutes = max(int(delta.total_seconds() // 60), 0)
    hours, minutes = divmod(total_minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    days, rem_hours = divmod(hours, 24)
    return f"{days}d{rem_hours}h"


def related_log_files(log_path: str) -> list[Path]:
    path = Path(log_path)
    if not path.parent.exists():
        return [path]

    candidates: list[Path] = []
    try:
        for candidate in path.parent.glob(f"{path.name}*"):
            if not candidate.is_file():
                continue
            if candidate.name != path.name and not candidate.name.startswith(f"{path.name}."):
                continue
            candidates.append(candidate)
    except Exception:
        return [path]

    if not candidates:
        return [path]
    return sorted(candidates, key=lambda item: (item.stat().st_mtime, item.name))


def analyze_log(log_path: str) -> tuple[Point, Point, Point, int]:
    now_utc = datetime.now(UTC)
    cutoff_24h = now_utc - timedelta(hours=24)
    latest_inference = Point(None, None)
    latest_reward = Point(None, None, None)
    latest_training = Point(None, None, None, "无训练事件", "training")
    process_died_24h = 0
    log_files = [path for path in related_log_files(log_path) if path.exists()]
    if not log_files:
        return latest_inference, latest_reward, latest_training, process_died_24h

    for path in log_files:
        last_ts: Optional[datetime] = None
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                ts_match = TS_RE.match(line)
                if ts_match:
                    last_ts = datetime.strptime(ts_match.group("ts"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
                    if PROCESS_DIED_RE.search(line) and last_ts >= cutoff_24h:
                        process_died_24h += 1

                inf_match = INF_RE.search(line)
                if inf_match:
                    latest_inference = Point(last_ts, float(inf_match.group("value")))

                reward_match = REWARD_RE.search(line)
                if reward_match:
                    latest_reward = Point(
                        last_ts,
                        float(reward_match.group("mean")),
                        float(reward_match.group("std")),
                    )
                    continue

                pairlist_match = TRAIN_PAIRLIST_RE.search(line)
                if pairlist_match:
                    latest_training = Point(
                        last_ts,
                        float(pairlist_match.group("value")),
                        None,
                        "训练完成",
                        "training",
                    )
                    continue

                if TRAIN_COMPLETE_RE.search(line):
                    latest_training = Point(last_ts, None, None, "训练完成", "training")
                    continue

                if ADAPTIVE_REWARD_RE.search(line) or PERF_ADJUST_RE.search(line):
                    latest_training = Point(last_ts, None, None, "自适应参数更新", "adaptive")

    return latest_inference, latest_reward, latest_training, process_died_24h


def parse_docker_timestamp(raw: str) -> Optional[datetime]:
    try:
        if raw.endswith("Z"):
            if "." in raw:
                head, frac = raw[:-1].split(".", 1)
                frac = (frac[:6]).ljust(6, "0")
                raw = f"{head}.{frac}+00:00"
            else:
                raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw).astimezone(UTC)
    except Exception:
        return None


def latest_reward_from_docker(container_name: str, since_hours: int = 72) -> Point:
    cmd = [
        *DOCKER_CMD,
        "logs",
        "--since",
        f"{since_hours}h",
        "--timestamps",
        container_name,
    ]
    latest_reward = Point(None, None, None)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        combined = "\n".join([result.stdout or "", result.stderr or ""])
        for line in combined.splitlines():
            reward_match = REWARD_RE.search(line)
            if not reward_match:
                continue
            ts_match = DOCKER_TS_RE.match(line)
            ts = parse_docker_timestamp(ts_match.group("ts")) if ts_match else None
            latest_reward = Point(
                ts,
                float(reward_match.group("mean")),
                float(reward_match.group("std")),
            )
    except Exception:
        return latest_reward
    return latest_reward


def latest_training_marker_from_docker(container_name: str, since_hours: int = 72) -> Point:
    cmd = [
        *DOCKER_CMD,
        "logs",
        "--since",
        f"{since_hours}h",
        "--timestamps",
        container_name,
    ]
    latest_training = Point(None, None, None, "无训练事件", "training")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        combined = "\n".join([result.stdout or "", result.stderr or ""])
        for line in combined.splitlines():
            ts_match = DOCKER_TS_RE.match(line)
            ts = parse_docker_timestamp(ts_match.group("ts")) if ts_match else None
            pairlist_match = TRAIN_PAIRLIST_RE.search(line)
            if pairlist_match:
                latest_training = Point(
                    ts,
                    float(pairlist_match.group("value")),
                    None,
                    "训练完成",
                    "training",
                )
                continue
            if TRAIN_COMPLETE_RE.search(line):
                latest_training = Point(ts, None, None, "训练完成", "training")
                continue
            if ADAPTIVE_REWARD_RE.search(line) or PERF_ADJUST_RE.search(line):
                latest_training = Point(ts, None, None, "自适应参数更新", "adaptive")
    except Exception:
        return latest_training
    return latest_training


def reward_state(latest_reward: Point, retrain_hours: Optional[float], now_utc: datetime) -> str:
    if latest_reward.ts is None:
        return "未发现训练reward"
    if latest_reward.ts is None or retrain_hours is None:
        return "已抓到reward"

    age_hours = (now_utc - latest_reward.ts).total_seconds() / 3600.0
    if age_hours <= max(retrain_hours * REWARD_GRACE_RATIO, 2.0):
        return "周期内正常"
    return "超过训练周期"


def inference_state(latest_inference: Point, now_utc: datetime) -> str:
    if latest_inference.value is None or latest_inference.ts is None:
        return "无数据"
    age_minutes = (now_utc - latest_inference.ts).total_seconds() / 60.0
    if age_minutes <= INFERENCE_STALE_MINUTES:
        return "正常"
    return "陈旧"


def fmt_reward(point: Point) -> str:
    if point.kind != "numeric":
        return point.label or "无"
    if point.value is None:
        return "无"
    return f"{point.value:.2f} ± {point.extra:.2f}"


def pick_newer_point(left: Point, right: Point) -> Point:
    if left.ts is None:
        return right
    if right.ts is None:
        return left
    return left if left.ts >= right.ts else right


def select_reward_point(
    latest_reward: Point,
    latest_training: Point,
    retrain_hours: Optional[float],
    now_utc: datetime,
) -> Point:
    if latest_training.ts is None:
        return latest_reward
    if latest_reward.ts is None:
        return latest_training
    if latest_reward.kind != "numeric":
        return latest_training if latest_training.ts >= latest_reward.ts else latest_reward
    if retrain_hours is None:
        return latest_reward

    grace_hours = max(retrain_hours * REWARD_GRACE_RATIO, 2.0)
    reward_age_hours = (now_utc - latest_reward.ts).total_seconds() / 3600.0
    if reward_age_hours > grace_hours and latest_training.ts > latest_reward.ts:
        return latest_training
    return latest_reward


def leverage_tag(leverage: object) -> str:
    try:
        lev = float(leverage or 1.0)
    except Exception:
        lev = 1.0
    if abs(lev - round(lev)) < 1e-9:
        return f"{int(round(lev))}x"
    return f"{str(lev).replace('.', 'p')}x"


def experience_tag_sort_key(tag: str) -> tuple[float, str]:
    raw = tag[:-1] if tag.endswith("x") else tag
    try:
        return float(raw.replace("p", ".")), tag
    except Exception:
        return float("inf"), tag


def sqlite_path_from_db_url(raw: object) -> Optional[Path]:
    if not isinstance(raw, str) or not raw.startswith("sqlite:"):
        return None
    parsed = urllib.parse.urlparse(raw)
    db_path = urllib.parse.unquote(parsed.path or "")
    if not db_path:
        return None
    if db_path.startswith("//"):
        db_path = db_path[1:]
    return Path(db_path)


def resolve_db_path(cfg: dict, db_candidates: list[str]) -> Optional[Path]:
    candidates: list[Path] = []
    cfg_db = sqlite_path_from_db_url(cfg.get("db_url"))
    if cfg_db is not None:
        candidates.append(cfg_db)
    for raw in db_candidates:
        candidates.append(Path(raw))
    for path in candidates:
        if path.exists():
            return path
    return None


def open_trade_leverage_counts(db_path: Optional[Path]) -> dict[str, int]:
    if db_path is None:
        return {}

    conn = None
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT leverage FROM trades WHERE is_open = 1")
        counts: dict[str, int] = {}
        for row in cursor.fetchall():
            raw = row[0] if row else 1.0
            tag = leverage_tag(1.0 if raw in (None, "") else raw)
            counts[tag] = counts.get(tag, 0) + 1
        return counts
    except Exception:
        return {}
    finally:
        if conn is not None:
            conn.close()


def parse_db_timestamp(raw: object) -> Optional[datetime]:
    if raw in (None, ""):
        return None
    try:
        dt = datetime.fromisoformat(str(raw))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def latest_closed_trade_by_leverage(db_path: Optional[Path]) -> dict[str, tuple[datetime, str]]:
    if db_path is None:
        return {}

    conn = None
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT leverage, pair, close_date
            FROM trades
            WHERE is_open = 0
              AND close_date IS NOT NULL
            ORDER BY close_date DESC
            """
        )
        latest: dict[str, tuple[datetime, str]] = {}
        for leverage, pair, close_date in cursor.fetchall():
            ts = parse_db_timestamp(close_date)
            if ts is None:
                continue
            tag = leverage_tag(1.0 if leverage in (None, "") else leverage)
            if tag in latest:
                continue
            latest[tag] = (ts, str(pair or "unknown"))
        return latest
    except Exception:
        return {}
    finally:
        if conn is not None:
            conn.close()


def safe_float(raw: object) -> Optional[float]:
    try:
        return float(raw)
    except Exception:
        return None


def resolve_strategy_path(root_path: str, cfg: dict) -> Optional[Path]:
    strategy_name = str(cfg.get("strategy") or "").strip()
    candidates: list[Path] = []
    raw_strategy_path = cfg.get("strategy_path")
    strategy_dirs: list[Path] = []
    if isinstance(raw_strategy_path, str) and raw_strategy_path.strip():
        base = Path(raw_strategy_path.strip())
        if not base.is_absolute():
            base = Path(root_path) / base
        strategy_dirs.append(base)

    strategy_dirs.append(Path(root_path) / "user_data" / "strategies")
    strategy_dirs.append(Path(root_path) / "strategies")

    if strategy_name:
        for base in strategy_dirs:
            candidates.append(base / f"{strategy_name}.py")

    for path in candidates:
        if path.exists():
            return path

    fallback_candidates: list[Path] = []
    for base in strategy_dirs:
        if not base.exists():
            continue
        try:
            for path in sorted(base.glob("*.py")):
                name = path.name
                if ".bak" in name or name.startswith("__"):
                    continue
                fallback_candidates.append(path)
        except Exception:
            continue

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for path in fallback_candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(path)

    if len(unique_candidates) == 1:
        return unique_candidates[0]
    return None


def read_strategy_hard_stop_params(root_path: str, cfg: dict) -> tuple[Optional[float], Optional[float], Optional[str]]:
    strategy_path = resolve_strategy_path(root_path, cfg)
    if strategy_path is None:
        return None, None, "未找到策略文件，硬止损审计已降级"

    values: dict[str, float] = {}
    try:
        with strategy_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                match = FLOAT_ASSIGN_RE.match(raw_line)
                if not match:
                    continue
                name = match.group("name")
                if name not in {"hard_stoploss_price_ratio", "max_hard_stoploss"}:
                    continue
                values[name] = float(match.group("value"))
    except Exception as exc:
        return None, None, f"读取策略文件失败，硬止损审计已降级 ({exc})"

    price_ratio = values.get("hard_stoploss_price_ratio")
    max_ratio = values.get("max_hard_stoploss")
    if price_ratio is None or max_ratio is None:
        return None, None, f"策略文件缺少硬止损参数，硬止损审计已降级 ({strategy_path.name})"
    return price_ratio, max_ratio, None


def resolve_api_server(cfg: dict) -> tuple[Optional[str], str, str, Optional[str]]:
    api_cfg = cfg.get("api_server")
    if not isinstance(api_cfg, dict) or not api_cfg.get("enabled"):
        return None, "", "", "未启用 api_server，硬止损审计已降级"

    port_raw = api_cfg.get("listen_port")
    try:
        port = int(port_raw)
    except Exception:
        return None, "", "", f"api_server 端口无效 ({port_raw})，硬止损审计已降级"

    host = str(api_cfg.get("listen_ip_address") or "127.0.0.1").strip()
    if host in {"", "0.0.0.0", "::", "[::]", "localhost"}:
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    scheme = "https" if api_cfg.get("ssl_cert") else "http"
    username = str(api_cfg.get("username") or "")
    password = str(api_cfg.get("password") or "")
    return f"{scheme}://{host}:{port}", username, password, None


def request_json(url: str, username: str = "", password: str = "") -> object:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    if username or password:
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")
    with urllib.request.urlopen(request, timeout=15) as response:
        raw = response.read().decode("utf-8", errors="ignore")
    return json.loads(raw)


def extract_trade_items(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("trades", "data", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if "trade_id" in payload or "pair" in payload:
            return [payload]
    return []


def fetch_open_trades(cfg: dict) -> tuple[list[dict], Optional[str]]:
    base_url, username, password, err = resolve_api_server(cfg)
    if err is not None or base_url is None:
        return [], err

    try:
        payload = request_json(f"{base_url}/api/v1/status", username, password)
    except Exception as exc:
        return [], f"状态接口不可用，硬止损审计已降级 ({exc})"

    open_trades: list[dict] = []
    for item in extract_trade_items(payload):
        if item.get("is_open") is False:
            continue
        open_trades.append(item)
    return open_trades, None


def expected_hard_stop_ratio(leverage: float, price_ratio: float, max_ratio: float) -> float:
    lev = max(1.0, float(leverage or 1.0))
    return -min(price_ratio * lev, max_ratio)


def format_stoploss_thresholds(price_ratio: float, max_ratio: float) -> str:
    parts: list[str] = []
    for lev in (1.0, 2.0, 3.0):
        expected = abs(expected_hard_stop_ratio(lev, price_ratio, max_ratio)) * 100.0
        parts.append(f"{leverage_tag(lev)} {expected:.2f}%")
    return " / ".join(parts)


def audit_stoploss_consistency(root_path: str, cfg: dict) -> StoplossAudit:
    price_ratio, max_ratio, strategy_err = read_strategy_hard_stop_params(root_path, cfg)
    if strategy_err is not None or price_ratio is None or max_ratio is None:
        return StoplossAudit(0, "审计降级", "无", [], [strategy_err or "硬止损参数未知"])

    detail = format_stoploss_thresholds(price_ratio, max_ratio)
    trades, api_err = fetch_open_trades(cfg)
    if api_err is not None:
        return StoplossAudit(0, "审计降级", detail, [], [api_err])
    if not trades:
        return StoplossAudit(0, "无持仓", detail, [], [])

    issues: list[str] = []
    checked_count = 0
    for trade in trades:
        leverage = safe_float(trade.get("leverage")) or 1.0
        actual_ratio = safe_float(trade.get("stop_loss_ratio"))
        trade_id = trade.get("trade_id") or trade.get("id") or "?"
        pair = str(trade.get("pair") or "unknown")
        tag = leverage_tag(leverage)

        if actual_ratio is None:
            issues.append(f"{pair} #{trade_id} 为 {tag}，但状态接口未返回 stop_loss_ratio")
            continue

        checked_count += 1
        expected_ratio = expected_hard_stop_ratio(leverage, price_ratio, max_ratio)
        if actual_ratio < expected_ratio - STOPLOSS_AUDIT_TOLERANCE:
            issues.append(
                f"{pair} #{trade_id} 为 {tag}，实际止损 {abs(actual_ratio) * 100:.2f}%，应不宽于 {abs(expected_ratio) * 100:.2f}%"
            )

    label = "发现异常" if issues else "正常"
    return StoplossAudit(
        checked_count=checked_count,
        label=label,
        detail=detail,
        issues=dedupe_texts(issues),
        notes=[],
    )


def available_experience_tags(root_path: str) -> list[str]:
    tags = {"1x", "2x", "3x"}
    user_data = Path(root_path) / "user_data"
    if user_data.exists():
        for path in user_data.glob("experience_replay_*"):
            if not path.is_dir():
                continue
            match = EXPERIENCE_DIR_RE.match(path.name)
            if match:
                tag = match.group("tag")
                if VALID_LEVERAGE_TAG_RE.match(tag):
                    tags.add(tag)
    return sorted(tags, key=experience_tag_sort_key)


def dedupe_texts(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def newest_mtime_under(path: Path, pattern: str = "*") -> tuple[Optional[datetime], int, str]:
    if not path.exists():
        return None, 0, ""
    candidates = []
    try:
        candidates = [p for p in path.glob(pattern) if p.exists()]
    except Exception:
        candidates = []
    if not candidates:
        return None, 0, ""
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    ts = datetime.fromtimestamp(newest.stat().st_mtime, tz=UTC)
    return ts, len(candidates), newest.name


def parse_record_timestamp(raw: object) -> Optional[datetime]:
    if raw in (None, ""):
        return None
    try:
        text = str(raw).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def model_freshness(root_path: str, identifier: str, retrain_hours: Optional[float], now_utc: datetime) -> Freshness:
    model_dir = Path(root_path) / 'user_data' / 'models' / identifier
    ts, count, name = newest_mtime_under(model_dir, 'sub-train-*')
    if ts is None:
        return Freshness(None, count, '无模型', name)
    if retrain_hours is None:
        return Freshness(ts, count, '已更新', name)
    age_hours = (now_utc - ts).total_seconds() / 3600.0
    if age_hours <= max(retrain_hours * REWARD_GRACE_RATIO, 2.0):
        label = '正常'
    else:
        label = '偏旧'
    return Freshness(ts, count, label, name)


def experience_freshness_for_tag(root_path: str, tag: str, now_utc: datetime) -> Freshness:
    exp_dir = Path(root_path) / 'user_data' / f'experience_replay_{tag}'
    if not exp_dir.exists():
        return Freshness(None, 0, '无经验', '')

    latest_ts: Optional[datetime] = None
    latest_name = ''
    valid_files = 0
    bad_files = 0

    for fp in sorted(exp_dir.glob('*.json')):
        try:
            payload = json.loads(fp.read_text(encoding='utf-8'))
        except Exception:
            bad_files += 1
            continue
        if not isinstance(payload, list):
            bad_files += 1
            continue

        valid_files += 1
        file_latest: Optional[datetime] = None
        for item in payload:
            if not isinstance(item, dict):
                continue
            for key in ('recorded_at', 'exit_time', 'entry_time'):
                ts = parse_record_timestamp(item.get(key))
                if ts is not None:
                    if file_latest is None or ts > file_latest:
                        file_latest = ts
                    break

        if file_latest is None:
            file_latest = datetime.fromtimestamp(fp.stat().st_mtime, tz=UTC)

        if latest_ts is None or file_latest > latest_ts:
            latest_ts = file_latest
            latest_name = fp.name

    if latest_ts is None:
        if bad_files > 0:
            return Freshness(None, 0, '目录损坏', '', bad_files)
        return Freshness(None, 0, '无经验', '')

    age_hours = (now_utc - latest_ts).total_seconds() / 3600.0
    if age_hours <= 24:
        label = '近期更新'
    elif age_hours <= 72:
        label = '偏旧'
    else:
        label = '陈旧'
    return Freshness(latest_ts, valid_files, label, latest_name, bad_files)


def experience_freshness(root_path: str, leverage: float, now_utc: datetime) -> Freshness:
    return experience_freshness_for_tag(root_path, leverage_tag(leverage), now_utc)


def audit_experience_health(
    root_path: str,
    leverage: float,
    cfg: dict,
    db_candidates: list[str],
    now_utc: datetime,
) -> ExperienceAudit:
    current_tag = leverage_tag(leverage)
    tags = available_experience_tags(root_path)
    info_by_tag = {tag: experience_freshness_for_tag(root_path, tag, now_utc) for tag in tags}
    current_info = info_by_tag.get(current_tag, Freshness(None, 0, "无经验", ""))
    issues: list[str] = []
    notes: list[str] = []

    db_path = resolve_db_path(cfg, db_candidates)
    open_counts = open_trade_leverage_counts(db_path)
    latest_closed = latest_closed_trade_by_leverage(db_path)
    if db_path is None:
        notes.append("未找到交易数据库，经验交叉校验已降级")

    if current_info.bad_count > 0:
        issues.append(f"当前 {current_tag} 经验目录损坏 {current_info.bad_count} 个")
    if current_info.count == 0:
        issues.append(f"当前 {current_tag} 经验目录为空")

    current_open = open_counts.get(current_tag, 0)
    if current_info.label in {"偏旧", "陈旧"} and current_open > 0:
        notes.append(f"当前 {current_tag} 尚有未平仓 {current_open} 笔，经验需待平仓后更新")

    skew = timedelta(minutes=EXPERIENCE_SKEW_MINUTES)
    for tag in tags:
        if tag == current_tag:
            continue
        info = info_by_tag[tag]
        open_count = open_counts.get(tag, 0)

        if info.bad_count > 0:
            issues.append(f"{tag} 经验目录损坏 {info.bad_count} 个")

        if open_count > 0:
            notes.append(f"{tag} 尚有未平仓 {open_count} 笔，后续仍可能写回 {tag} 经验")

        if info.ts is None:
            continue
        if current_info.ts is not None and info.ts <= current_info.ts + skew:
            continue
        if open_count == 0:
            closed_info = latest_closed.get(tag)
            if closed_info is not None:
                closed_ts, closed_pair = closed_info
                delta_seconds = abs((info.ts - closed_ts).total_seconds())
                if delta_seconds <= RECENT_CLOSED_EXPERIENCE_MATCH_MINUTES * 60:
                    notes.append(
                        f"{tag} 最近有已平仓遗留单 {closed_pair} 于 {fmt_ts(closed_ts)} 写回经验"
                    )
                    continue
            issues.append(f"{tag} 经验比当前 {current_tag} 更新，但数据库无对应未平仓")

    return ExperienceAudit(
        current_tag=current_tag,
        current_info=current_info,
        issues=dedupe_texts(issues),
        notes=dedupe_texts(notes),
    )


def send_telegram_message(message: str, bot_token: str, chat_id: str) -> str:
    payload = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": message,
        }
    ).encode("utf-8")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    request = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read().decode("utf-8", errors="ignore")


def generate_report() -> str:
    now_utc = datetime.now(UTC)
    now_local = now_utc.astimezone(LOCAL_TZ)
    lines = ["Freqtrade RL 容器监控报告"]
    lines.append(f"时间：{now_local.strftime('%Y-%m-%d %H:%M:%S')} (Asia/Shanghai)")
    lines.append("")

    for meta in CONTAINERS:
        root_path = resolve_container_root(meta["root_name"])
        config_path = str(root_path / meta["config_name"])
        log_path = str(root_path / "user_data" / "logs" / "freqtrade.log")
        db_candidates = [str(root_path / rel) for rel in meta.get("db_relative_paths", [])]
        cfg = read_config(config_path)
        freqai = cfg.get("freqai", {})
        identifier = freqai.get("identifier", "unknown")
        retrain_hours = freqai.get("live_retrain_hours")
        try:
            retrain_hours = float(retrain_hours) if retrain_hours is not None else None
        except Exception:
            retrain_hours = None

        latest_inference, latest_reward_log, latest_training_log, process_died_24h = analyze_log(log_path)
        latest_reward_raw = latest_reward_from_docker(meta["name"])
        latest_training_raw = latest_training_marker_from_docker(meta["name"])
        latest_reward_numeric = pick_newer_point(latest_reward_log, latest_reward_raw)
        latest_training = pick_newer_point(latest_training_log, latest_training_raw)
        latest_reward = select_reward_point(latest_reward_numeric, latest_training, retrain_hours, now_utc)
        status, started_at = get_container_runtime(meta["name"])
        leverage = (cfg.get('exchange') or {}).get('leverage', 1.0)
        model_info = model_freshness(str(root_path), identifier, retrain_hours, now_utc)
        exp_audit = audit_experience_health(
            str(root_path),
            leverage,
            cfg,
            db_candidates,
            now_utc,
        )
        sl_audit = audit_stoploss_consistency(str(root_path), cfg)
        exp_info = exp_audit.current_info
        issues = dedupe_texts(exp_audit.issues + sl_audit.issues)
        notes = dedupe_texts(exp_audit.notes + sl_audit.notes)

        lines.append(f"[{meta['display']}]")
        lines.append(
            f"运行：{status}｜运行时长 {fmt_runtime(started_at, now_utc)}｜24h崩溃 {process_died_24h} 次"
        )
        lines.append(
            f"推理：{('%.2f 秒' % latest_inference.value) if latest_inference.value is not None else '无'}"
            f"｜时间 {fmt_ts(latest_inference.ts)}｜距今 {fmt_age(latest_inference.ts, now_utc)}｜{inference_state(latest_inference, now_utc)}"
        )
        lines.append(
            f"Reward：{fmt_reward(latest_reward)}"
            f"｜时间 {fmt_ts(latest_reward.ts)}｜距今 {fmt_age(latest_reward.ts, now_utc)}｜{reward_state(latest_reward, retrain_hours, now_utc)}"
        )
        lines.append(
            f"模型：时间 {fmt_ts(model_info.ts)}｜距今 {fmt_age(model_info.ts, now_utc)}｜子模型 {model_info.count} 个｜{model_info.label}"
        )
        lines.append(
            f"经验：当前 {exp_audit.current_tag}｜时间 {fmt_ts(exp_info.ts)}｜距今 {fmt_age(exp_info.ts, now_utc)}｜JSON {exp_info.count} 个"
            f"{(f'｜损坏 {exp_info.bad_count} 个' if exp_info.bad_count else '')}｜{exp_info.label}"
        )
        lines.append(
            f"止损：审计 {sl_audit.checked_count} 笔｜{sl_audit.detail}｜{sl_audit.label}"
        )
        if issues:
            lines.append(f"告警：{'；'.join(issues)}")
        if notes:
            lines.append(f"提示：{'；'.join(notes)}")
        lines.append(
            f"重训周期：{('%sh' % int(retrain_hours)) if retrain_hours is not None else '未知'}｜标识 {identifier}"
        )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", sys.argv[1] if len(sys.argv) > 1 else "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", sys.argv[2] if len(sys.argv) > 2 else "194652099")

    report = generate_report()
    print(report, end="")

    if bot_token:
        try:
            result = send_telegram_message(report, bot_token, chat_id)
            print("\n发送结果：" + result)
        except Exception as exc:
            print("\n发送失败：" + str(exc))
