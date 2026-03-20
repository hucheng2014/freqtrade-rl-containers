#!/usr/bin/env python3
import argparse
import json
import math
import os
import sqlite3
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


DEPLOY_ROOT_CANDIDATES = [
    Path(os.environ["FREQTRADE_DEPLOY_ROOT"])
    for _ in [0]
    if os.environ.get("FREQTRADE_DEPLOY_ROOT")
]
DEPLOY_ROOT_CANDIDATES.extend([
    Path.home() / "下载" / "shipan",
    Path("/opt"),
])

CONTAINERS = [
    {
        "name": "RL_SignalFilter_LIVE",
        "root_name": "freqtrade_rl_live",
        "db_relative_paths": ["user_data/tradesv3.sqlite"],
        "config_name": "config_rl_live.json",
    },
    {
        "name": "RL_SelfEvolve_DRY",
        "root_name": "freqtrade_rl_selfevolve",
        "db_relative_paths": [
            "user_data/tradesv3.sqlite",
            "user_data/tradesv3_dry.sqlite",
        ],
        "config_name": "config_selfevolve.json",
    },
    {
        "name": "DogeAI_NoT3_RL_WithBTC_LIVE",
        "root_name": "freqtrade_dogeai_not3_rl_live",
        "db_relative_paths": ["user_data/tradesv3.sqlite"],
        "config_name": "config.json",
    },
]

SHANGHAI_TZ = timezone(timedelta(hours=8))
BINANCE_FUTURES_MIN_LEVERAGE = 1.0
REGIME_SYMBOL = "BTCUSDT"
REGIME_TIMEFRAME = "1d"
REGIME_LIMIT = 420
REGIME_BREAKOUT_N = 10
REGIME_FAST_MA = 20
REGIME_SLOW_MA = 60
REGIME_BULL_THRESHOLD = 0.65
REGIME_BEAR_THRESHOLD = 0.35
REPORTS_HOME = Path(
    os.environ.get("FREQTRADE_REPORTS_HOME", str(Path.home() / "freqtrade_reports"))
)
REGIME_JSON = REPORTS_HOME / "output" / "regime_monitor_latest.json"
POST_FIX_STARTS_UTC = {
    "RL_SignalFilter_LIVE": "2026-03-05T20:52:55+00:00",
    "RL_SelfEvolve_DRY": "2026-03-03T14:58:48+00:00",
    "DogeAI_NoT3_RL_WithBTC_LIVE": "2026-03-05T20:43:05+00:00",
}


def resolve_container_root(root_name: str) -> Path:
    for base in DEPLOY_ROOT_CANDIDATES:
        candidate = base / root_name
        if candidate.exists():
            return candidate
    return DEPLOY_ROOT_CANDIDATES[0] / root_name


@dataclass
class Metrics:
    container: str
    db_path: str
    config_path: str
    db_mtime_utc: str
    lookback_days: int
    lookback_hours: Optional[int]
    bucket_hours: int
    aggregation_mode: str
    aggregation_label: str
    days_in_window: int
    buckets_in_window: int
    window_start_sh: Optional[str]
    window_end_sh: Optional[str]
    current_leverage: Optional[float]
    stake_currency: Optional[str]
    post_fix_start_utc: Optional[str]
    post_fix_closed_trades: Optional[int]
    post_fix_to_100: Optional[int]
    post_fix_to_300: Optional[int]
    closed_trades: int
    first_close_utc: Optional[str]
    last_close_utc: Optional[str]
    avg_trade_return: Optional[float]
    stdev_trade_return: Optional[float]
    variance_trade_return: Optional[float]
    win_rate: Optional[float]
    trades_per_year: Optional[float]
    sharpe_trade: Optional[float]
    sharpe_annualized: Optional[float]
    kelly_full: Optional[float]
    half_kelly_raw: Optional[float]
    half_kelly_recommended: Optional[float]
    leverage_recommended_actionable: Optional[float]
    edge_label: Optional[str]
    leverage_label: Optional[str]
    expected_growth_half_kelly: Optional[float]
    expected_growth_trade_pct: Optional[float]
    expected_growth_annual_pct: Optional[float]
    today_pnl_abs: Optional[float]
    current_bucket_label: Optional[str]
    current_bucket_pnl_abs: Optional[float]
    window_pnl_abs: Optional[float]
    bucket_pnl_abs: list[dict]
    daily_pnl_abs: list[dict]
    leverage_advice: Optional[str]
    note: Optional[str]


@dataclass
class BullBearMonitor:
    symbol: str
    timeframe: str
    candle_close_sh: Optional[str]
    latest_close: Optional[float]
    score: Optional[float]
    p_bull: Optional[float]
    state_code: int
    state_label: str
    switch_code: int
    switch_label: str
    breakout_n: int
    fast_ma: int
    slow_ma: int
    bull_threshold: float
    bear_threshold: float
    ret3: Optional[float]
    ret10: Optional[float]
    trend: Optional[float]
    vol_z: Optional[float]
    dd60: Optional[float]
    break_up: Optional[bool]
    break_down: Optional[bool]
    last_switch_sh: Optional[str]
    last_switch_label: Optional[str]
    note: Optional[str]
    error: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate trade-level Sharpe and half-Kelly metrics for Freqtrade containers."
    )
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--lookback-hours", type=int, default=0)
    parser.add_argument("--bucket-hours", type=int, default=24)
    parser.add_argument("--recent-buckets", type=int, default=6)
    parser.add_argument("--risk-free-annual", type=float, default=0.0)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--json-output", type=str, default="")
    parser.add_argument("--telegram-token", type=str, default=os.getenv("TELEGRAM_BOT_TOKEN", ""))
    parser.add_argument("--telegram-chat-id", type=str, default=os.getenv("TELEGRAM_CHAT_ID", ""))
    parser.add_argument("--policy-json", type=str, default="")
    parser.add_argument("--skip-telegram", action="store_true")
    return parser.parse_args()


def resolve_db_path(candidates: list[str]) -> Path:
    existing = [Path(path) for path in candidates if Path(path).exists()]
    if not existing:
        raise FileNotFoundError(f"No database found in candidates: {candidates}")
    return max(existing, key=lambda path: path.stat().st_mtime)


def load_current_leverage(config_path: str) -> Optional[float]:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    leverage = cfg.get("exchange", {}).get("leverage")
    return float(leverage) if leverage is not None else None


def shanghai_day_window(lookback_days: int) -> tuple[datetime, datetime]:
    now_sh = datetime.now(SHANGHAI_TZ)
    start_day = now_sh.date() - timedelta(days=max(lookback_days - 1, 0))
    start_sh = datetime.combine(start_day, datetime.min.time(), tzinfo=SHANGHAI_TZ)
    return start_sh.astimezone(timezone.utc), now_sh.astimezone(timezone.utc)


def resolve_metrics_window(lookback_days: int, lookback_hours: int) -> tuple[datetime, datetime]:
    if lookback_hours and lookback_hours > 0:
        now_sh = datetime.now(SHANGHAI_TZ)
        start_sh = now_sh - timedelta(hours=lookback_hours)
        return start_sh.astimezone(timezone.utc), now_sh.astimezone(timezone.utc)
    return shanghai_day_window(lookback_days)


def bucket_start_sh(ts_sh: datetime, bucket_hours: int) -> datetime:
    if bucket_hours <= 0:
        raise ValueError("bucket_hours must be positive")
    ts_sh = ts_sh.astimezone(SHANGHAI_TZ)
    if bucket_hours >= 24:
        return ts_sh.replace(hour=0, minute=0, second=0, microsecond=0)
    floored_hour = (ts_sh.hour // bucket_hours) * bucket_hours
    return ts_sh.replace(hour=floored_hour, minute=0, second=0, microsecond=0)


def format_bucket_label(bucket_start: datetime, bucket_hours: int) -> str:
    bucket_start = bucket_start.astimezone(SHANGHAI_TZ)
    if bucket_hours >= 24:
        return bucket_start.strftime("%m-%d")
    return bucket_start.strftime("%m-%d %H:%M")


def build_bucket_keys(window_start_sh: datetime, window_end_sh: datetime, bucket_hours: int) -> list[datetime]:
    current = bucket_start_sh(window_start_sh, bucket_hours)
    last = bucket_start_sh(window_end_sh, bucket_hours)
    keys: list[datetime] = []
    while current <= last:
        keys.append(current)
        current += timedelta(hours=bucket_hours)
    return keys


def load_closed_trades(db_path: Path, cutoff_utc: datetime) -> list[tuple[float, float, datetime, str]]:
    sql = """
        select close_profit, close_profit_abs, close_date, stake_currency
        from trades
        where is_open = 0
          and close_profit is not null
          and close_profit_abs is not null
          and close_date is not null
          and close_date >= ?
        order by close_date
    """
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(sql, (cutoff_utc.replace(tzinfo=None).isoformat(sep=" "),)).fetchall()
    conn.close()
    return [
        (
            float(close_profit),
            float(close_profit_abs),
            datetime.fromisoformat(close_date).replace(tzinfo=timezone.utc),
            str(stake_currency or "USDT"),
        )
        for close_profit, close_profit_abs, close_date, stake_currency in rows
    ]


def count_closed_trades_since(db_path: Path, start_utc: Optional[str]) -> Optional[int]:
    if not start_utc:
        return None
    start_dt = datetime.fromisoformat(start_utc).astimezone(timezone.utc)
    conn = sqlite3.connect(str(db_path))
    count = conn.execute(
        """
        select count(*)
        from trades
        where is_open = 0
          and close_date is not null
          and close_date >= ?
        """,
        (start_dt.replace(tzinfo=None).isoformat(sep=" "),),
    ).fetchone()[0]
    conn.close()
    return int(count)


def compute_metrics(
    container_name: str,
    db_path: Path,
    config_path: str,
    current_leverage: Optional[float],
    post_fix_start_utc: Optional[str],
    post_fix_closed_trades: Optional[int],
    trades: list[tuple[float, float, datetime, str]],
    lookback_days: int,
    lookback_hours: Optional[int],
    bucket_hours: int,
    window_start_utc: datetime,
    window_end_utc: datetime,
    risk_free_annual: float,
    min_trades: int,
) -> Metrics:
    db_mtime_utc = datetime.fromtimestamp(db_path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    window_start_sh = window_start_utc.astimezone(SHANGHAI_TZ)
    window_end_sh = window_end_utc.astimezone(SHANGHAI_TZ)
    days_in_window = (window_end_sh.date() - window_start_sh.date()).days + 1
    bucket_keys = build_bucket_keys(window_start_sh, window_end_sh, bucket_hours)
    bucket_returns_map: dict[datetime, float] = {key: 0.0 for key in bucket_keys}
    bucket_pnl_abs_map: dict[datetime, float] = {key: 0.0 for key in bucket_keys}

    stake_currency = "USDT"
    today_pnl_abs = 0.0
    for close_profit, close_profit_abs, close_dt, record_stake_currency in trades:
        close_sh = close_dt.astimezone(SHANGHAI_TZ)
        if close_sh.date() == window_end_sh.date():
            today_pnl_abs += close_profit_abs
        bucket_key = bucket_start_sh(close_sh, bucket_hours)
        if bucket_key in bucket_returns_map:
            bucket_returns_map[bucket_key] += close_profit
            bucket_pnl_abs_map[bucket_key] += close_profit_abs
        if record_stake_currency:
            stake_currency = record_stake_currency

    bucket_pnl_abs = [
        {
            "date": key.isoformat(timespec="seconds"),
            "label": format_bucket_label(key, bucket_hours),
            "pnl_abs": bucket_pnl_abs_map[key],
            "bucket_return": bucket_returns_map[key],
        }
        for key in bucket_keys
    ]
    daily_pnl_abs = [
        {"date": entry["date"], "label": entry["label"], "pnl_abs": entry["pnl_abs"]}
        for entry in bucket_pnl_abs
    ]
    current_bucket_label = bucket_pnl_abs[-1]["label"] if bucket_pnl_abs else None
    current_bucket_pnl_abs = bucket_pnl_abs[-1]["pnl_abs"] if bucket_pnl_abs else 0.0
    window_pnl_abs = sum(entry["pnl_abs"] for entry in bucket_pnl_abs)
    aggregation_mode = "daily_shanghai" if bucket_hours == 24 else f"{bucket_hours}h_shanghai"
    aggregation_label = "日" if bucket_hours == 24 else f"{bucket_hours}h"
    buckets_in_window = len(bucket_keys)

    if len(trades) < 2:
        return Metrics(
            container=container_name,
            db_path=str(db_path),
            config_path=config_path,
            db_mtime_utc=db_mtime_utc,
            lookback_days=lookback_days,
            lookback_hours=lookback_hours,
            bucket_hours=bucket_hours,
            aggregation_mode=aggregation_mode,
            aggregation_label=aggregation_label,
            days_in_window=days_in_window,
            buckets_in_window=buckets_in_window,
            window_start_sh=window_start_sh.isoformat(timespec="seconds"),
            window_end_sh=window_end_sh.isoformat(timespec="seconds"),
            current_leverage=current_leverage,
            stake_currency=stake_currency,
            post_fix_start_utc=post_fix_start_utc,
            post_fix_closed_trades=post_fix_closed_trades,
            post_fix_to_100=max(0, 100 - post_fix_closed_trades) if post_fix_closed_trades is not None else None,
            post_fix_to_300=max(0, 300 - post_fix_closed_trades) if post_fix_closed_trades is not None else None,
            closed_trades=len(trades),
            first_close_utc=trades[0][2].isoformat(timespec="seconds") if trades else None,
            last_close_utc=trades[-1][2].isoformat(timespec="seconds") if trades else None,
            avg_trade_return=None,
            stdev_trade_return=None,
            variance_trade_return=None,
            win_rate=None,
            trades_per_year=None,
            sharpe_trade=None,
            sharpe_annualized=None,
            kelly_full=None,
            half_kelly_raw=None,
            half_kelly_recommended=None,
            leverage_recommended_actionable=None,
            edge_label="未知",
            leverage_label="未知",
            expected_growth_half_kelly=None,
            expected_growth_trade_pct=None,
            expected_growth_annual_pct=None,
            today_pnl_abs=today_pnl_abs,
            current_bucket_label=current_bucket_label,
            current_bucket_pnl_abs=current_bucket_pnl_abs,
            window_pnl_abs=window_pnl_abs,
            bucket_pnl_abs=bucket_pnl_abs,
            daily_pnl_abs=daily_pnl_abs,
            leverage_advice="样本不足，先观察。",
            note="滚动窗口内闭单样本不足。",
        )

    returns = [item[0] for item in trades]
    dates = [item[2] for item in trades]
    closed_trades = len(returns)
    bucket_returns = [bucket_returns_map[key] for key in bucket_keys]
    mean_return = sum(bucket_returns) / len(bucket_returns)
    variance = (
        sum((value - mean_return) ** 2 for value in bucket_returns) / (len(bucket_returns) - 1)
        if len(bucket_returns) > 1
        else 0.0
    )
    stdev = math.sqrt(variance)
    win_rate = sum(1 for value in returns if value > 0) / closed_trades
    periods_per_year = (365.0 * 24.0) / float(bucket_hours)
    rf_per_bucket = risk_free_annual / periods_per_year
    excess_mean = mean_return - rf_per_bucket
    sharpe_trade = excess_mean / stdev if stdev > 0 else None
    sharpe_annualized = sharpe_trade * math.sqrt(periods_per_year) if sharpe_trade is not None else None
    kelly_full = excess_mean / variance if variance > 0 else None
    half_kelly_raw = kelly_full / 2 if kelly_full is not None else None
    half_kelly_recommended = max(0.0, half_kelly_raw) if half_kelly_raw is not None else None
    leverage_recommended_actionable = None
    if half_kelly_recommended is not None:
        leverage_recommended_actionable = max(
            BINANCE_FUTURES_MIN_LEVERAGE, half_kelly_recommended
        )
    effective_leverage_for_growth = None
    if leverage_recommended_actionable is not None:
        effective_leverage_for_growth = leverage_recommended_actionable

    expected_growth_half = None
    if effective_leverage_for_growth is not None:
        expected_growth_half = (
            rf_per_bucket
            + effective_leverage_for_growth * excess_mean
            - 0.5 * (effective_leverage_for_growth ** 2) * variance
        )
    expected_growth_trade_pct = (
        expected_growth_half * 100.0 if expected_growth_half is not None else None
    )
    expected_growth_annual_pct = (
        expected_growth_half * periods_per_year * 100.0
        if expected_growth_half is not None and periods_per_year is not None
        else None
    )

    leverage_advice = None
    edge_label = None
    leverage_label = None
    if half_kelly_recommended is None:
        edge_label = "未知"
        leverage_label = "未知"
        leverage_advice = "无法估算杠杆。"
    elif half_kelly_raw is not None and half_kelly_raw <= 0:
        edge_label = "负边际"
        leverage_label = "观察"
        if current_leverage is not None and current_leverage > BINANCE_FUTURES_MIN_LEVERAGE:
            leverage_advice = "降到1.00x，仅保留采样。"
        else:
            leverage_advice = "保留1.00x采样，继续收集样本。"
    elif half_kelly_recommended < BINANCE_FUTURES_MIN_LEVERAGE:
        edge_label = "正边际"
        leverage_label = "轻仓"
        if current_leverage is not None and current_leverage > BINANCE_FUTURES_MIN_LEVERAGE:
            leverage_advice = "降到1.00x；半凯仍低于1x。"
        else:
            leverage_advice = "维持1.00x；半凯仍低于1x。"
    elif half_kelly_recommended <= 1.5:
        edge_label = "正边际"
        leverage_label = "标准"
        if current_leverage is None:
            leverage_advice = f"目标约 {half_kelly_recommended:.2f}x。"
        elif half_kelly_recommended < current_leverage * 0.8:
            leverage_advice = f"降到 {half_kelly_recommended:.2f}x。"
        elif half_kelly_recommended > current_leverage * 1.2:
            leverage_advice = f"升到 {half_kelly_recommended:.2f}x。"
        else:
            leverage_advice = f"维持在 {half_kelly_recommended:.2f}x 附近。"
    elif half_kelly_recommended <= 2.5:
        edge_label = "正边际"
        leverage_label = "进取"
        if current_leverage is None:
            leverage_advice = f"目标约 {half_kelly_recommended:.2f}x。"
        elif half_kelly_recommended < current_leverage * 0.8:
            leverage_advice = f"降到 {half_kelly_recommended:.2f}x。"
        elif half_kelly_recommended > current_leverage * 1.2:
            leverage_advice = f"升到 {half_kelly_recommended:.2f}x。"
        else:
            leverage_advice = f"维持在 {half_kelly_recommended:.2f}x 附近。"
    else:
        edge_label = "正边际"
        leverage_label = "激进"
        if current_leverage is None:
            leverage_advice = f"目标约 {half_kelly_recommended:.2f}x。"
        elif half_kelly_recommended < current_leverage * 0.8:
            leverage_advice = f"降到 {half_kelly_recommended:.2f}x。"
        elif half_kelly_recommended > current_leverage * 1.2:
            leverage_advice = f"升到 {half_kelly_recommended:.2f}x。"
        else:
            leverage_advice = f"维持在 {half_kelly_recommended:.2f}x 附近。"

    note = None
    if closed_trades < min_trades:
        note = f"闭单数低于 min_trades={min_trades}，仅供参考。"

    return Metrics(
        container=container_name,
        db_path=str(db_path),
        config_path=config_path,
        db_mtime_utc=db_mtime_utc,
        lookback_days=lookback_days,
        lookback_hours=lookback_hours,
        bucket_hours=bucket_hours,
        aggregation_mode=aggregation_mode,
        aggregation_label=aggregation_label,
        days_in_window=days_in_window,
        buckets_in_window=buckets_in_window,
        window_start_sh=window_start_sh.isoformat(timespec="seconds"),
        window_end_sh=window_end_sh.isoformat(timespec="seconds"),
        current_leverage=current_leverage,
        stake_currency=stake_currency,
        post_fix_start_utc=post_fix_start_utc,
        post_fix_closed_trades=post_fix_closed_trades,
        post_fix_to_100=max(0, 100 - post_fix_closed_trades) if post_fix_closed_trades is not None else None,
        post_fix_to_300=max(0, 300 - post_fix_closed_trades) if post_fix_closed_trades is not None else None,
        closed_trades=closed_trades,
        first_close_utc=dates[0].isoformat(timespec="seconds"),
        last_close_utc=dates[-1].isoformat(timespec="seconds"),
        avg_trade_return=mean_return,
        stdev_trade_return=stdev,
        variance_trade_return=variance,
        win_rate=win_rate,
        trades_per_year=periods_per_year,
        sharpe_trade=sharpe_trade,
        sharpe_annualized=sharpe_annualized,
        kelly_full=kelly_full,
        half_kelly_raw=half_kelly_raw,
        half_kelly_recommended=half_kelly_recommended,
        leverage_recommended_actionable=leverage_recommended_actionable,
        edge_label=edge_label,
        leverage_label=leverage_label,
        expected_growth_half_kelly=expected_growth_half,
        expected_growth_trade_pct=expected_growth_trade_pct,
        expected_growth_annual_pct=expected_growth_annual_pct,
        today_pnl_abs=today_pnl_abs,
        current_bucket_label=current_bucket_label,
        current_bucket_pnl_abs=current_bucket_pnl_abs,
        window_pnl_abs=window_pnl_abs,
        bucket_pnl_abs=bucket_pnl_abs,
        daily_pnl_abs=daily_pnl_abs,
        leverage_advice=leverage_advice,
        note=note,
    )


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_fin(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if value < 0:
        return f"-{abs(value):.{digits}f}"
    if value > 0:
        return f"+{value:.{digits}f}"
    return f"{value:.{digits}f}"


def fmt_money(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if value < 0:
        return f"-{abs(value):.{digits}f}"
    if value > 0:
        return f"+{value:.{digits}f}"
    return f"{value:.{digits}f}"


def fmt_intish(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        return str(int(round(float(value))))
    except Exception:
        return "n/a"


def display_name(container_name: str) -> str:
    mapping = {
        "RL_SignalFilter_LIVE": "信号过滤",
        "RL_SelfEvolve_DRY": "自我进化",
        "DogeAI_NoT3_RL_WithBTC_LIVE": "DogeAI",
    }
    return mapping.get(container_name, container_name)


def fetch_binance_futures_klines(symbol: str, interval: str, limit: int) -> list[dict]:
    params = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "limit": limit})
    url = f"https://fapi.binance.com/fapi/v1/klines?{params}"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected kline payload for {symbol}: {type(payload).__name__}")

    now_utc = datetime.now(timezone.utc)
    rows: list[dict] = []
    for item in payload:
        if not isinstance(item, list) or len(item) < 7:
            continue
        close_time_utc = datetime.fromtimestamp(int(item[6]) / 1000, tz=timezone.utc)
        if close_time_utc > now_utc:
            continue
        rows.append(
            {
                "open_time_utc": datetime.fromtimestamp(int(item[0]) / 1000, tz=timezone.utc),
                "close_time_utc": close_time_utc,
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
            }
        )
    return rows


def rolling_max(values: list[float], window: int, *, exclude_current: bool = False) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(values)
    for idx in range(len(values)):
        end = idx if exclude_current else idx + 1
        start = end - window
        if start < 0:
            continue
        out[idx] = max(values[start:end])
    return out


def rolling_min(values: list[float], window: int, *, exclude_current: bool = False) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(values)
    for idx in range(len(values)):
        end = idx if exclude_current else idx + 1
        start = end - window
        if start < 0:
            continue
        out[idx] = min(values[start:end])
    return out


def rolling_mean(values: list[Optional[float]], window: int) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(values)
    for idx in range(len(values)):
        start = idx + 1 - window
        if start < 0:
            continue
        window_values = values[start : idx + 1]
        if any(value is None for value in window_values):
            continue
        out[idx] = sum(float(value) for value in window_values) / window
    return out


def rolling_std(values: list[Optional[float]], window: int) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(values)
    for idx in range(len(values)):
        start = idx + 1 - window
        if start < 0:
            continue
        window_values = values[start : idx + 1]
        if any(value is None for value in window_values):
            continue
        floats = [float(value) for value in window_values]
        mean_value = sum(floats) / window
        if window <= 1:
            out[idx] = 0.0
            continue
        variance = sum((value - mean_value) ** 2 for value in floats) / (window - 1)
        out[idx] = math.sqrt(variance)
    return out


def pct_change(values: list[float], periods: int) -> list[Optional[float]]:
    out: list[Optional[float]] = [None] * len(values)
    for idx in range(periods, len(values)):
        previous = values[idx - periods]
        if previous == 0:
            continue
        out[idx] = values[idx] / previous - 1.0
    return out


def compute_bull_bear_monitor(
    symbol: str = REGIME_SYMBOL,
    timeframe: str = REGIME_TIMEFRAME,
    limit: int = REGIME_LIMIT,
    breakout_n: int = REGIME_BREAKOUT_N,
    fast_ma: int = REGIME_FAST_MA,
    slow_ma: int = REGIME_SLOW_MA,
    bull_th: float = REGIME_BULL_THRESHOLD,
    bear_th: float = REGIME_BEAR_THRESHOLD,
) -> BullBearMonitor:
    try:
        klines = fetch_binance_futures_klines(symbol, timeframe, limit)
        if len(klines) < max(180, slow_ma + breakout_n + 10):
            raise ValueError(f"Not enough klines: {len(klines)}")

        closes = [item["close"] for item in klines]
        highs = [item["high"] for item in klines]
        lows = [item["low"] for item in klines]
        close_times = [item["close_time_utc"] for item in klines]

        ret1 = pct_change(closes, 1)
        ret3 = pct_change(closes, 3)
        ret10 = pct_change(closes, 10)
        hhv_n = rolling_max(highs, breakout_n, exclude_current=True)
        llv_n = rolling_min(lows, breakout_n, exclude_current=True)
        ma_fast = rolling_mean(closes, fast_ma)
        ma_slow = rolling_mean(closes, slow_ma)
        rolling_close_max_60 = rolling_max(closes, 60, exclude_current=False)
        vol20 = rolling_std(ret1, 20)
        vol120_mean = rolling_mean(vol20, 120)
        vol120_std = rolling_std(vol20, 120)

        trend: list[Optional[float]] = [None] * len(closes)
        vol_z: list[Optional[float]] = [None] * len(closes)
        dd60: list[Optional[float]] = [None] * len(closes)
        break_up: list[Optional[bool]] = [None] * len(closes)
        break_down: list[Optional[bool]] = [None] * len(closes)
        score: list[float] = [0.0] * len(closes)
        p_bull: list[float] = [0.5] * len(closes)
        state: list[int] = [0] * len(closes)
        switch: list[int] = [0] * len(closes)

        for idx in range(len(closes)):
            if ma_fast[idx] is not None and ma_slow[idx] not in (None, 0):
                trend[idx] = ma_fast[idx] / ma_slow[idx] - 1.0
            if vol20[idx] is not None and vol120_mean[idx] is not None and vol120_std[idx] not in (None, 0):
                vol_z[idx] = (vol20[idx] - vol120_mean[idx]) / vol120_std[idx]
            if rolling_close_max_60[idx] not in (None, 0):
                dd60[idx] = closes[idx] / rolling_close_max_60[idx] - 1.0
            if hhv_n[idx] is not None:
                break_up[idx] = closes[idx] > hhv_n[idx]
            if llv_n[idx] is not None:
                break_down[idx] = closes[idx] < llv_n[idx]

            bull_rules = (
                (1.0 if (ret3[idx] is not None and ret3[idx] >= 0.01) else 0.0)
                + (1.0 if (ret10[idx] is not None and ret10[idx] >= 0.03) else 0.0)
                + (1.0 if break_up[idx] else 0.0)
                + (1.0 if (ma_fast[idx] is not None and ma_slow[idx] is not None and ma_fast[idx] > ma_slow[idx]) else 0.0)
            )
            bear_rules = (
                (1.0 if (ret3[idx] is not None and ret3[idx] <= -0.01) else 0.0)
                + (1.0 if (ret10[idx] is not None and ret10[idx] <= -0.03) else 0.0)
                + (1.0 if break_down[idx] else 0.0)
                + (1.0 if (ma_fast[idx] is not None and ma_slow[idx] is not None and ma_fast[idx] < ma_slow[idx]) else 0.0)
            )
            score[idx] = (
                0.8 * (bull_rules - bear_rules)
                + 2.0 * math.tanh((ret10[idx] or 0.0) * 10.0)
                + 2.5 * math.tanh((trend[idx] or 0.0) * 15.0)
                - 0.8 * math.tanh((vol_z[idx] or 0.0) / 2.0)
                + 2.0 * math.tanh((dd60[idx] or 0.0) * 8.0)
            )
            p_bull[idx] = 1.0 / (1.0 + math.exp(-score[idx]))

            if idx == 0 or hhv_n[idx] is None or llv_n[idx] is None:
                continue

            previous_state = state[idx - 1]
            if previous_state <= 0 and p_bull[idx] >= bull_th and break_up[idx]:
                state[idx] = 1
            elif previous_state >= 0 and p_bull[idx] <= bear_th and break_down[idx]:
                state[idx] = -1
            else:
                state[idx] = previous_state
            switch[idx] = state[idx] - previous_state

        latest_idx = len(closes) - 1
        latest_switch = switch[latest_idx]
        latest_state = state[latest_idx]
        state_label = "牛市" if latest_state == 1 else "熊市" if latest_state == -1 else "中性"
        if latest_switch == 2:
            switch_label = "熊转牛"
        elif latest_switch == -2:
            switch_label = "牛转熊"
        elif latest_state == 1:
            switch_label = "维持牛"
        elif latest_state == -1:
            switch_label = "维持熊"
        else:
            switch_label = "中性观察"

        last_switch_idx = next((idx for idx in range(latest_idx, -1, -1) if switch[idx] in (2, -2)), None)
        last_switch_sh = None
        last_switch_label = None
        if last_switch_idx is not None:
            last_switch_sh = close_times[last_switch_idx].astimezone(SHANGHAI_TZ).strftime("%m-%d %H:%M")
            last_switch_label = "熊转牛" if switch[last_switch_idx] == 2 else "牛转熊"

        return BullBearMonitor(
            symbol=symbol,
            timeframe=timeframe,
            candle_close_sh=close_times[latest_idx].astimezone(SHANGHAI_TZ).strftime("%m-%d %H:%M"),
            latest_close=closes[latest_idx],
            score=score[latest_idx],
            p_bull=p_bull[latest_idx],
            state_code=latest_state,
            state_label=state_label,
            switch_code=latest_switch,
            switch_label=switch_label,
            breakout_n=breakout_n,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            bull_threshold=bull_th,
            bear_threshold=bear_th,
            ret3=ret3[latest_idx],
            ret10=ret10[latest_idx],
            trend=trend[latest_idx],
            vol_z=vol_z[latest_idx],
            dd60=dd60[latest_idx],
            break_up=break_up[latest_idx],
            break_down=break_down[latest_idx],
            last_switch_sh=last_switch_sh,
            last_switch_label=last_switch_label,
            note="共享监控，仅发送，不接入实盘或自动调杠。",
            error=None,
        )
    except Exception as exc:
        return BullBearMonitor(
            symbol=symbol,
            timeframe=timeframe,
            candle_close_sh=None,
            latest_close=None,
            score=None,
            p_bull=None,
            state_code=0,
            state_label="未知",
            switch_code=0,
            switch_label="获取失败",
            breakout_n=breakout_n,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            bull_threshold=bull_th,
            bear_threshold=bear_th,
            ret3=None,
            ret10=None,
            trend=None,
            vol_z=None,
            dd60=None,
            break_up=None,
            break_down=None,
            last_switch_sh=None,
            last_switch_label=None,
            note=None,
            error=str(exc),
        )


def fmt_close_time(value: Optional[str]) -> str:
    if not value:
        return "n/a"
    ts = datetime.fromisoformat(value).astimezone(SHANGHAI_TZ)
    return ts.strftime("%m-%d %H:%M")


def load_policy_map(policy_json: str) -> dict[str, dict]:
    if not policy_json:
        return {}
    path = Path(policy_json)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    result = {}
    for item in payload:
        if isinstance(item, dict) and item.get("container"):
            result[item["container"]] = item
    return result


def is_rolling_mode(args: argparse.Namespace) -> bool:
    return bool(args.lookback_hours and args.lookback_hours > 0) or args.bucket_hours != 24


def report_window_label(args: argparse.Namespace) -> str:
    if args.lookback_hours and args.lookback_hours > 0:
        return f"{args.lookback_hours}h"
    return f"{args.lookback_days}天"


def report_title(args: argparse.Namespace) -> str:
    if is_rolling_mode(args):
        return f"滚动凯利/夏普（{report_window_label(args)}窗口 / {args.bucket_hours}h桶）"
    return f"日度凯利/夏普（{args.lookback_days}天）"


def aggregation_phrase(args: argparse.Namespace) -> str:
    if args.bucket_hours == 24:
        return "北京时间按日聚合，零平仓日计0；闭单/胜率仍按逐笔统计。"
    return (
        f"北京时间按{args.bucket_hours}小时聚合，零平仓桶计0；"
        "闭单/胜率仍按逐笔统计。"
    )


def render_report(
    metrics_list: list[Metrics],
    args: argparse.Namespace,
    policy_map: dict[str, dict],
    regime_monitor: BullBearMonitor,
) -> str:
    lines = []
    now_sh = datetime.now(SHANGHAI_TZ).strftime("%m-%d %H:%M")
    total_today_abs = sum(item.today_pnl_abs or 0.0 for item in metrics_list)
    total_current_bucket_abs = sum(item.current_bucket_pnl_abs or 0.0 for item in metrics_list)
    total_window_abs = sum(item.window_pnl_abs or 0.0 for item in metrics_list)
    total_currency = next((item.stake_currency for item in metrics_list if item.stake_currency), "USDT")
    lines.append(report_title(args))
    if is_rolling_mode(args):
        lines.append(
            f"时间 {now_sh}｜rf {args.risk_free_annual:.4f}｜窗口 {report_window_label(args)}｜桶 {args.bucket_hours}h"
        )
        lines.append(f"当前桶合计(3容器已平仓) {fmt_money(total_current_bucket_abs, 4)} {total_currency}")
        lines.append(f"窗口合计(3容器已平仓) {fmt_money(total_window_abs, 4)} {total_currency}")
    else:
        lines.append(f"时间 {now_sh}｜rf {args.risk_free_annual:.4f}")
        lines.append(f"今日合计(3容器已平仓) {fmt_money(total_today_abs, 4)} {total_currency}")
    lines.append(f"口径 {aggregation_phrase(args)}")
    if policy_map:
        lines.append("说明 已先执行自动调杠，以下现杠为调后值。")
    lines.append("")
    lines.append("【共享牛熊监控】")
    if regime_monitor.error:
        lines.append(f"标的 {regime_monitor.symbol}｜周期 {regime_monitor.timeframe}｜状态 获取失败")
        lines.append(f"原因 {regime_monitor.error}")
    else:
        lines.append(f"标的 {regime_monitor.symbol}｜周期 {regime_monitor.timeframe}｜K线截止 {regime_monitor.candle_close_sh}")
        lines.append(
            f"状态 {regime_monitor.state_label}｜概率 {fmt(regime_monitor.p_bull, 3)}｜切换 {regime_monitor.switch_label}"
        )
        lines.append(
            f"收盘 {fmt(regime_monitor.latest_close, 2)}｜得分 {fmt_fin(regime_monitor.score, 3)}｜3日 {fmt_fin((regime_monitor.ret3 or 0.0) * 100.0, 2)}%｜10日 {fmt_fin((regime_monitor.ret10 or 0.0) * 100.0, 2)}%"
        )
        lines.append(
            f"突破{regime_monitor.breakout_n}日高 {'是' if regime_monitor.break_up else '否'}｜跌破{regime_monitor.breakout_n}日低 {'是' if regime_monitor.break_down else '否'}｜MA{regime_monitor.fast_ma}/{regime_monitor.slow_ma} {fmt_fin((regime_monitor.trend or 0.0) * 100.0, 2)}%"
        )
        lines.append(
            f"波动Z {fmt_fin(regime_monitor.vol_z, 3)}｜距60日高 {fmt_fin((regime_monitor.dd60 or 0.0) * 100.0, 2)}%｜上次切换 {regime_monitor.last_switch_sh or 'n/a'} {regime_monitor.last_switch_label or ''}".rstrip()
        )
        if regime_monitor.note:
            lines.append(f"说明 {regime_monitor.note}")
    lines.append("")
    for item in metrics_list:
        policy_item = policy_map.get(item.container, {})
        lines.append(f"【{display_name(item.container)}】")
        lines.append(f"闭单 {item.closed_trades}｜胜率 {fmt(item.win_rate, 3)}")
        lines.append(
            f"修后 {item.post_fix_closed_trades if item.post_fix_closed_trades is not None else 'n/a'}｜距100 {item.post_fix_to_100 if item.post_fix_to_100 is not None else 'n/a'}｜距300 {item.post_fix_to_300 if item.post_fix_to_300 is not None else 'n/a'}"
        )
        lines.append(f"平仓 {fmt_close_time(item.last_close_utc)}")
        if is_rolling_mode(args):
            lines.append(
                f"当前桶 {item.current_bucket_label or 'n/a'} {fmt_money(item.current_bucket_pnl_abs, 4)} {item.stake_currency or 'USDT'}｜窗口 {report_window_label(args)} {fmt_money(item.window_pnl_abs, 4)} {item.stake_currency or 'USDT'}"
            )
            recent_entries = item.bucket_pnl_abs[-max(args.recent_buckets, 1):]
            recent_line = "｜".join(
                f"{entry.get('label', entry.get('date', 'n/a'))} {fmt_money(float(entry['pnl_abs']), 4)}"
                for entry in recent_entries
            )
            lines.append(f"近{len(recent_entries)}桶盈亏 {recent_line}")
            lines.append(
                f"今日累计 {fmt_money(item.today_pnl_abs, 4)} {item.stake_currency or 'USDT'}"
            )
        else:
            lines.append(
                f"今盈亏 {fmt_money(item.today_pnl_abs, 4)} {item.stake_currency or 'USDT'}｜窗口 {report_window_label(args)} {fmt_money(item.window_pnl_abs, 4)} {item.stake_currency or 'USDT'}"
            )
            daily_pnl_line = "｜".join(
                f"{entry.get('label', entry.get('date', 'n/a'))} {fmt_money(float(entry['pnl_abs']), 4)}"
                for entry in item.daily_pnl_abs
            )
            lines.append(f"日盈亏 {daily_pnl_line}")
        lines.append(
            f"{item.aggregation_label}均 {fmt_fin(item.avg_trade_return, 5)}｜{item.aggregation_label}波 {fmt(item.stdev_trade_return, 5)}"
        )
        lines.append(
            f"{item.aggregation_label}夏 {fmt_fin(item.sharpe_trade)}｜年夏 {fmt_fin(item.sharpe_annualized)}"
        )
        lines.append(
            f"满凯({item.aggregation_label}) {fmt_fin(item.kelly_full)}｜半凯({item.aggregation_label}) {fmt_fin(item.half_kelly_raw)}"
        )
        lines.append(
            f"现杠 {fmt(item.current_leverage, 2)}x｜建杠 {fmt(item.leverage_recommended_actionable, 2)}x"
        )
        if policy_item:
            lines.append(
                f"调杠 前{fmt_intish(policy_item.get('current_leverage_before'))}x｜整型 {fmt_intish(policy_item.get('desired_integer_capped'))}x｜后{fmt_intish(policy_item.get('target_leverage_after_policy'))}x"
            )
            reason = ", ".join(policy_item.get('reason') or []) or 'n/a'
            lines.append(
                f"执行 {'已应用' if policy_item.get('changed') else '未改'}｜重启 {'是' if policy_item.get('restarted') else '否'}｜因 {reason}"
            )
        lines.append(
            f"边际 {item.edge_label or 'n/a'}｜状态 {item.leverage_label or 'n/a'}"
        )
        lines.append(
            f"建议 {item.leverage_advice or 'n/a'}"
        )
        lines.append(
            f"预增/{item.aggregation_label} {fmt_fin(item.expected_growth_trade_pct, 3)}%｜年 {fmt_fin(item.expected_growth_annual_pct, 2)}%"
        )
        if item.note:
            lines.append(f"备注 {item.note}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    request = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        response.read()


def main() -> int:
    args = parse_args()
    if args.bucket_hours <= 0:
        print("--bucket-hours must be positive", file=sys.stderr)
        return 2
    metrics_list: list[Metrics] = []
    regime_monitor = compute_bull_bear_monitor()
    window_start_utc, window_end_utc = resolve_metrics_window(args.lookback_days, args.lookback_hours)

    for container in CONTAINERS:
        root_path = resolve_container_root(container["root_name"])
        db_candidates = [str(root_path / rel) for rel in container["db_relative_paths"]]
        config_path = str(root_path / container["config_name"])
        db_path = resolve_db_path(db_candidates)
        current_leverage = load_current_leverage(config_path)
        post_fix_start_utc = POST_FIX_STARTS_UTC.get(container["name"])
        post_fix_closed_trades = count_closed_trades_since(db_path, post_fix_start_utc)
        trades = load_closed_trades(db_path, window_start_utc)
        metrics_list.append(
            compute_metrics(
                container_name=container["name"],
                db_path=db_path,
                config_path=config_path,
                current_leverage=current_leverage,
                post_fix_start_utc=post_fix_start_utc,
                post_fix_closed_trades=post_fix_closed_trades,
                trades=trades,
                lookback_days=args.lookback_days,
                lookback_hours=args.lookback_hours if args.lookback_hours > 0 else None,
                bucket_hours=args.bucket_hours,
                window_start_utc=window_start_utc,
                window_end_utc=window_end_utc,
                risk_free_annual=args.risk_free_annual,
                min_trades=args.min_trades,
            )
        )

    policy_map = load_policy_map(args.policy_json)
    report = render_report(metrics_list, args, policy_map, regime_monitor)
    print(report, end="")

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([asdict(item) for item in metrics_list], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        REGIME_JSON.parent.mkdir(parents=True, exist_ok=True)
        REGIME_JSON.write_text(
            json.dumps(asdict(regime_monitor), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.telegram_token and args.telegram_chat_id and not args.skip_telegram:
        try:
            send_telegram_message(args.telegram_token, args.telegram_chat_id, report)
        except Exception as exc:
            print(f"Telegram send failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
