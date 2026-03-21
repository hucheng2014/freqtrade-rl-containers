"""
import logging
from datetime import datetime, timedelta, timezone
logger = logging.getLogger(__name__)
MTF_BalancedPerformance_DogeAI_NoT3_RL

方案 B（RL 过滤）版本：
- 保留 NoT3 策略的候选信号逻辑（MTF 趋势 + 支撑阻力 + BTC 过滤）
- 通过 FreqAI RL 动作列 `&-action` 决定是否执行入场/出场
"""

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from pandas import DataFrame
import pandas as pd
import numpy as np
from freqtrade.enums import MarginMode

from MTF_BalancedPerformance_DogeAI_NoT3 import MTF_BalancedPerformance_DogeAI_NoT3

# 导入 ExperienceBuffer（位于 user_data/freqaimodels）
sys.path.insert(0, str(Path(__file__).parent.parent / "freqaimodels"))
try:
    from ExperienceBuffer import ExperienceBuffer
except ImportError:
    ExperienceBuffer = None

logger = logging.getLogger(__name__)


class MTF_BalancedPerformance_DogeAI_NoT3_RL(MTF_BalancedPerformance_DogeAI_NoT3):
    """
    DogeAI NoT3 + RL 过滤版本。
    """

    # 1x 基准硬止损为价格变动 6%，实际会按杠杆同步为 6% * leverage。
    stoploss = -0.06
    hard_stoploss_price_ratio = 0.06
    max_hard_stoploss = 0.18
    use_custom_stoploss = True
    use_exit_signal = True
    rl_exit_deadband_min = -0.0020
    rl_exit_deadband_max = 0.0035
    rl_exit_long_deadband_min = rl_exit_deadband_min
    rl_exit_long_deadband_max = rl_exit_deadband_max
    rl_exit_short_deadband_min = rl_exit_deadband_min
    rl_exit_short_deadband_max = rl_exit_deadband_max
    max_hold_hours = 46.0
    soft_stoploss_fraction = 0.75
    soft_stoploss_min_hold_minutes = 30
    reentry_loss_lookback_hours = 24
    reentry_loss_cooldown_minutes = 90
    reentry_loss_streak_cooldown_minutes = 240
    small_win_reentry_profit_threshold = 0.0035
    small_win_reentry_cooldown_minutes = 30
    futures_settings_retry_minutes = 30
    min_notional_notify_minutes = 60

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self._exp_buffers = {}
        self._experience_db_path = Path(__file__).resolve().parent.parent / "tradesv3.sqlite"
        self._last_experience_reconcile_at = None
        self._experience_reconcile_bootstrap_done = False
        self._futures_pair_seeded = False
        self._futures_pair_retry_after = None
        self._futures_pairs_ready = set()
        self._futures_pair_pending = {}
        self._futures_open_orders_pending = False
        self._last_entry_guard_log_at = {}
        if ExperienceBuffer:
            exchange_cfg = (config or {}).get("exchange") or {}
            lev = exchange_cfg.get("leverage", 1.0)
            self._get_exp_buffer(lev)
            logger.info("[Strategy] ExperienceBuffer initialized")
        else:
            logger.warning("[Strategy] ExperienceBuffer not available")

    def _hard_stoploss_for_leverage(self, leverage: Optional[float] = None) -> float:
        try:
            lev = float(leverage if leverage is not None else self._configured_leverage())
        except Exception:
            lev = 1.0
        lev = max(1.0, lev)
        return -min(self.hard_stoploss_price_ratio * lev, self.max_hard_stoploss)

    def _sync_hard_stoploss(self, leverage: Optional[float] = None) -> float:
        hard_sl = self._hard_stoploss_for_leverage(leverage)
        self.stoploss = hard_sl
        return hard_sl

    def _get_exp_buffer(self, leverage: float):
        if not ExperienceBuffer:
            return None
        try:
            lev = float(leverage or 1.0)
        except Exception:
            lev = 1.0
        if lev <= 0:
            lev = 1.0
        if abs(lev - round(lev)) < 1e-9:
            tag = f"{int(round(lev))}x"
        else:
            tag = f"{str(lev).replace('.', 'p')}x"
        storage_dir = f"user_data/experience_replay_{tag}"
        buf = self._exp_buffers.get(storage_dir)
        if not buf:
            buf = ExperienceBuffer(
                storage_dir=storage_dir,
                max_experiences=2000,
            )
            self._exp_buffers[storage_dir] = buf
        return buf

    def _experience_leverage(self, trade=None) -> float:
        configured = self._configured_leverage()
        try:
            trade_lev = float(getattr(trade, "leverage", 0.0) or 0.0)
        except Exception:
            trade_lev = 0.0
        if trade_lev > 0 and (trade_lev > 1.0 or configured <= 1.0):
            return trade_lev
        return configured

    @staticmethod
    def _to_utc_dt(value):
        if value is None:
            return None
        try:
            if hasattr(value, 'to_pydatetime'):
                value = value.to_pydatetime()
            if isinstance(value, datetime):
                dt = value
            else:
                dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _experience_recorded(self, pair: str, trade_id: int, leverage: float) -> bool:
        if trade_id <= 0:
            return False
        exp_buffer = self._get_exp_buffer(leverage)
        if not exp_buffer:
            return False
        try:
            experiences = exp_buffer.load_experiences(pair=pair, min_count=0)
        except TypeError:
            experiences = exp_buffer.load_experiences(pair=pair)
        for experience in experiences:
            try:
                if int(experience.get("trade_id", 0) or 0) == trade_id:
                    return True
            except Exception:
                continue
        return False

    def _trade_stub_from_row(self, row: sqlite3.Row):
        open_dt = self._to_utc_dt(row["open_date"]) or datetime.now(timezone.utc)
        close_dt = self._to_utc_dt(row["close_date"]) or open_dt
        open_rate = float(row["open_rate"] or 0.0)
        close_rate = float(row["close_rate"] or 0.0)
        leverage = float(row["leverage"] or self._configured_leverage() or 1.0)
        leverage = max(leverage, 1.0)
        timeframe = row["timeframe"]
        if timeframe in (None, "", 0, "0"):
            tf_digits = "".join(ch for ch in str(getattr(self, "timeframe", "15m")) if ch.isdigit())
            timeframe = int(tf_digits or 15)

        trade_stub = SimpleNamespace(
            id=int(row["id"] or 0),
            is_short=bool(row["is_short"]),
            is_open=False,
            open_date_utc=open_dt,
            close_date_utc=close_dt,
            open_rate=open_rate,
            close_rate=close_rate,
            close_rate_requested=close_rate,
            close_profit=row["close_profit"],
            stake_amount=float(row["stake_amount"] or 0.0),
            leverage=leverage,
            timeframe=int(timeframe or 15),
            exit_reason=row["exit_reason"],
        )

        def _calc_profit_ratio(rate: float, *, _open_rate=open_rate, _is_short=bool(row["is_short"]), _leverage=leverage) -> float:
            if _open_rate <= 0 or rate is None:
                return 0.0
            rate = float(rate)
            raw_profit = (_open_rate - rate) / _open_rate if _is_short else (rate - _open_rate) / _open_rate
            return raw_profit * _leverage

        trade_stub.calc_profit_ratio = _calc_profit_ratio
        return trade_stub

    def _reconcile_missing_trade_experiences(self, current_time) -> None:
        if not ExperienceBuffer or not self._experience_db_path.exists():
            return

        current_time_utc = self._to_utc_dt(current_time) or datetime.now(timezone.utc)
        cooldown = timedelta(minutes=5)
        if (
            self._last_experience_reconcile_at
            and current_time_utc - self._last_experience_reconcile_at < cooldown
        ):
            return
        self._last_experience_reconcile_at = current_time_utc

        bootstrap = not self._experience_reconcile_bootstrap_done
        lookback = timedelta(days=7 if bootstrap else 0, hours=0 if bootstrap else 6)
        row_limit = 400 if bootstrap else 80
        since_text = (current_time_utc - lookback).replace(tzinfo=None).isoformat(sep=" ")

        try:
            db_uri = f"file:{self._experience_db_path.as_posix()}?mode=ro"
            with sqlite3.connect(db_uri, uri=True, timeout=5.0) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT id, pair, is_short, open_date, close_date, open_rate, close_rate,
                           close_profit, stake_amount, leverage, exit_reason, timeframe
                    FROM trades
                    WHERE is_open = 0
                      AND close_date IS NOT NULL
                      AND close_rate IS NOT NULL
                      AND close_rate > 0
                      AND (strategy = ? OR strategy IS NULL OR strategy = '')
                      AND close_date >= ?
                    ORDER BY close_date DESC
                    LIMIT ?
                    """,
                    (self.__class__.__name__, since_text, row_limit),
                ).fetchall()
        except Exception as exc:
            logger.debug(f"[DogeAI_RL] Experience sqlite backfill query failed: {exc}")
            return

        recovered = 0
        for row in rows:
            trade_id = int(row["id"] or 0)
            pair = str(row["pair"] or "")
            leverage = float(row["leverage"] or self._configured_leverage() or 1.0)
            if not pair or trade_id <= 0 or self._experience_recorded(pair, trade_id, leverage):
                continue

            close_dt = self._to_utc_dt(row["close_date"]) or current_time_utc
            trade_stub = self._trade_stub_from_row(row)
            exit_reason = str(row["exit_reason"] or "sqlite_recovered_exit")
            self._record_trade_experience(pair, trade_stub, exit_reason, close_dt)
            recovered += 1

        self._experience_reconcile_bootstrap_done = True
        if recovered:
            logger.warning(
                f"[DogeAI_RL] Recovered {recovered} closed trades into experience buffer via sqlite backfill."
            )

    def _notify_min_notional_skip(
        self,
        pair: str,
        side: str,
        current_time,
        proposed_stake: float,
        max_stake: float,
        required_stake: float,
        rate: float,
    ) -> None:
        current_time_utc = self._to_utc_dt(current_time) or datetime.now(timezone.utc)
        sent_at = getattr(self, "_min_notional_skip_notified_at", {})
        pair_key = f"{pair}|{side or 'unknown'}"
        previous_time = sent_at.get(pair_key)
        cooldown = timedelta(minutes=float(self.min_notional_notify_minutes))
        if previous_time and current_time_utc - previous_time < cooldown:
            return

        bot_name = str((self.config or {}).get("bot_name") or self.__class__.__name__)
        message = (
            f"[{bot_name}] Skip entry {pair} ({side or 'unknown'}) due to insufficient stake: "
            f"proposed={proposed_stake:.4f} max={max_stake:.4f} "
            f"required={required_stake:.4f} rate={rate:.4f}. Pair locked 30m."
        )

        try:
            if getattr(self, "dp", None) and hasattr(self.dp, "send_msg"):
                self.dp.send_msg(message, always_send=True)
                sent_at[pair_key] = current_time_utc
                self._min_notional_skip_notified_at = sent_at
        except Exception as exc:
            logger.warning("[Strategy] Failed to send min-notional skip notification for %s: %s", pair, exc)

    def _configured_leverage(self) -> float:
        try:
            exchange_cfg = (self.config or {}).get('exchange') or {}
            lev = float(exchange_cfg.get('leverage', 1.0) or 1.0)
            return lev if lev > 0 else 1.0
        except Exception:
            return 1.0

    def _recent_pair_side_loss_state(self, pair: str, side: str, reference_time):
        exp_buffer = self._get_exp_buffer(self._configured_leverage())
        ref_utc = self._to_utc_dt(reference_time)
        if not exp_buffer or not ref_utc:
            return 0, None

        experiences = exp_buffer.load_experiences(pair=pair, min_count=0)
        if not experiences:
            return 0, None

        lookback_cutoff = ref_utc - timedelta(hours=self.reentry_loss_lookback_hours)
        filtered = []
        for exp in experiences:
            if exp.get('side') != side:
                continue
            ts = self._to_utc_dt(exp.get('exit_time') or exp.get('recorded_at'))
            if not ts or ts > ref_utc or ts < lookback_cutoff:
                continue
            try:
                profit = float(exp.get('profit_ratio', 0.0) or 0.0)
            except Exception:
                profit = 0.0
            filtered.append((ts, profit))

        if not filtered:
            return 0, None

        filtered.sort(key=lambda item: item[0], reverse=True)
        if filtered[0][1] >= 0:
            return 0, None

        streak = 0
        for _, profit in filtered:
            if profit < 0:
                streak += 1
            else:
                break
        return streak, filtered[0][0]

    def _should_block_reentry(self, pair: str, side: str, reference_time):
        streak, latest_loss_time = self._recent_pair_side_loss_state(pair, side, reference_time)
        ref_utc = self._to_utc_dt(reference_time)
        if streak <= 0 or not latest_loss_time or not ref_utc:
            return False, streak, None, 0.0

        age_minutes = max((ref_utc - latest_loss_time).total_seconds() / 60.0, 0.0)
        cooldown = float(
            self.reentry_loss_streak_cooldown_minutes
            if streak >= 2 else self.reentry_loss_cooldown_minutes
        )
        return age_minutes < cooldown, streak, age_minutes, cooldown

    def _filled_order_profit_ratio(self, trade, order) -> float | None:
        try:
            close_profit = getattr(trade, "close_profit", None)
            if close_profit is not None:
                return float(close_profit)
        except Exception:
            pass

        for attr in ("safe_price", "average", "price"):
            try:
                raw_rate = getattr(order, attr, None)
                if raw_rate not in (None, ""):
                    return float(trade.calc_profit_ratio(float(raw_rate)))
            except Exception:
                continue
        return None

    def _apply_small_win_reentry_cooldown(
        self,
        pair: str,
        trade,
        order,
        exit_reason: str,
        current_time,
    ) -> None:
        if exit_reason not in {"rl_exit_long", "rl_exit_short"}:
            return

        try:
            if hasattr(trade, "get_custom_data") and trade.get_custom_data("small_win_cooldown_applied", False):
                return
        except Exception:
            pass

        profit_ratio = self._filled_order_profit_ratio(trade, order)
        if profit_ratio is None:
            return

        threshold = float(getattr(self, "small_win_reentry_profit_threshold", 0.0035) or 0.0035)
        if profit_ratio >= threshold:
            return

        cooldown_minutes = float(getattr(self, "small_win_reentry_cooldown_minutes", 30.0) or 30.0)
        if cooldown_minutes <= 0:
            return

        lock_until = (self._to_utc_dt(current_time) or datetime.now(timezone.utc)) + timedelta(minutes=cooldown_minutes)
        try:
            self.lock_pair(pair, lock_until, reason="small_win_exit_cooldown")
            logger.info(
                f"[EntryGuard] Locked {pair} for {cooldown_minutes:.0f}m after {exit_reason}: "
                f"realized={profit_ratio:.4%} below re-entry threshold {threshold:.2%}"
            )
            if hasattr(trade, "set_custom_data"):
                trade.set_custom_data("small_win_cooldown_applied", True)
        except Exception as exc:
            logger.warning(f"[EntryGuard] Failed to lock {pair} after small-win exit: {exc}")

    def leverage(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag,
        side: str,
        **kwargs,
    ) -> float:
        """
        期货杠杆（每笔新交易）：
        使用配置中的 exchange.leverage，并同步价格止损 6% * leverage。
        """
        target = self._configured_leverage()
        if max_leverage and float(max_leverage) > 0:
            target = max(1.0, min(target, float(max_leverage)))
        else:
            target = max(1.0, target)
        self._sync_hard_stoploss(target)
        return target

    def custom_stake_amount(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        stake = float(proposed_stake or 0.0)
        rate = float(current_rate or 0.0)
        if stake <= 0 or rate <= 0:
            return stake
        if not self._guard_entry_pair(pair, current_time):
            return 0.0

        try:
            market = self.dp.market(pair) if getattr(self, 'dp', None) else None
        except Exception:
            market = None

        if not market:
            return stake

        limits = market.get('limits') or {}
        amount_limits = limits.get('amount') or {}
        market_limits = limits.get('market') or {}
        cost_limits = limits.get('cost') or {}
        precision = market.get('precision') or {}

        amount_step = float(precision.get('amount') or 0.0)
        amount_min = float(amount_limits.get('min') or market_limits.get('min') or 0.0)
        min_cost = float(cost_limits.get('min') or 0.0)
        contract_size = float(market.get('contractSize') or 1.0)
        lev = max(1.0, float(leverage or 1.0))

        if amount_step <= 0:
            amount_step = amount_min if amount_min > 0 else 0.0

        min_qty = max(amount_min, amount_step) if amount_step > 0 else amount_min
        if min_cost > 0 and contract_size > 0:
            raw_qty = min_cost / (rate * contract_size)
            if amount_step > 0:
                steps = np.ceil(raw_qty / amount_step)
                qty_for_cost = steps * amount_step
            else:
                qty_for_cost = raw_qty
            min_qty = max(min_qty, qty_for_cost)

        required_stake = (min_qty * rate * contract_size) / lev if min_qty > 0 else stake
        required_stake *= 1.01

        if required_stake <= stake + 1e-12:
            return stake

        if required_stake <= float(max_stake or 0.0) + 1e-12:
            logger.info(
                f'[Strategy] Raised stake for {pair} from {stake:.4f} to {required_stake:.4f} to satisfy exchange min notional/lot size.'
            )
            return required_stake

        try:
            lock_until = current_time + timedelta(minutes=30)
            self.lock_pair(pair, lock_until, reason='min_notional_insufficient')
        except Exception:
            pass

        logger.warning(
            f'[Strategy] Skip entry for {pair}: proposed_stake={stake:.4f} max_stake={float(max_stake or 0.0):.4f} '
            f'cannot satisfy effective minimum stake {required_stake:.4f} at rate={rate:.4f}.'
        )
        self._notify_min_notional_skip(
            pair=pair,
            side=side,
            current_time=current_time,
            proposed_stake=stake,
            max_stake=float(max_stake or 0.0),
            required_stake=required_stake,
            rate=rate,
        )
        return 0.0


    def _exchange_handle(self):
        return getattr(getattr(self, "dp", None), "_exchange", None)

    def _desired_margin_mode(self) -> MarginMode:
        margin_mode = str((self.config or {}).get("margin_mode") or "isolated").lower()
        if margin_mode == MarginMode.CROSS.value:
            return MarginMode.CROSS
        return MarginMode.ISOLATED

    def _desired_leverage(self) -> float:
        exchange_cfg = (self.config or {}).get("exchange") or {}
        try:
            leverage = float(exchange_cfg.get("leverage") or 1.0)
        except Exception:
            leverage = 1.0
        return max(1.0, leverage)

    def _entry_allowed_for_pair(self, pair: str) -> bool:
        if str((self.config or {}).get("trading_mode") or "").lower() != "futures":
            return True
        if getattr(self, "_futures_open_orders_pending", False):
            return False
        ready_pairs = getattr(self, "_futures_pairs_ready", set()) or set()
        return pair in ready_pairs

    def _ensure_futures_pair_settings(self, force: bool = False) -> None:
        if getattr(self, "_futures_pair_seeded", False) and not force:
            return

        if str((self.config or {}).get("trading_mode") or "").lower() != "futures":
            self._futures_pair_seeded = True
            return

        exchange = self._exchange_handle()
        if exchange is None or not hasattr(exchange, "_api"):
            logger.warning("[StartupGuard] Exchange handle unavailable, skip futures pair seeding.")
            self._futures_pair_seeded = False
            return

        exchange_cfg = (self.config or {}).get("exchange") or {}
        pair_whitelist = list(exchange_cfg.get("pair_whitelist") or [])
        if not pair_whitelist:
            self._futures_pair_seeded = True
            return

        desired_margin = self._desired_margin_mode()
        desired_leverage = self._desired_leverage()
        api = exchange._api
        ready_pairs = set()
        pending_pairs = {}

        try:
            if hasattr(api, 'options'):
                api.options['warnOnFetchOpenOrdersWithoutSymbol'] = False
            open_orders = api.fetch_open_orders()
        except Exception as exc:
            logger.warning(f"[StartupGuard] Could not fetch open orders before futures seed: {exc}")
            self._futures_pair_seeded = False
            self._futures_open_orders_pending = True
            return

        self._futures_open_orders_pending = bool(open_orders)
        if open_orders:
            logger.warning(
                f"[StartupGuard] Block new entries until {len(open_orders)} open orders are cleared."
            )

        updated_pairs = []

        for pair in pair_whitelist:
            try:
                market = api.market(pair)
                risk = api.fapiPrivateV2GetPositionRisk({"symbol": market["id"]})
                info = risk[0] if risk else {}
                position_amt = float(info.get("positionAmt") or 0.0)
                current_margin = str(info.get("marginType") or "").lower()
                current_leverage = float(info.get("leverage") or 0.0)

                if open_orders:
                    pending_pairs[pair] = f"open_orders={len(open_orders)}"
                    continue

                changed = False
                if abs(position_amt) <= 1e-12:
                    if current_margin != desired_margin.value:
                        exchange.set_margin_mode(pair, desired_margin, accept_fail=False)
                        changed = True
                    if abs(current_leverage - desired_leverage) > 1e-9:
                        exchange._set_leverage(desired_leverage, pair, accept_fail=False)
                        changed = True

                risk_check = api.fapiPrivateV2GetPositionRisk({"symbol": market["id"]})
                check = risk_check[0] if risk_check else {}
                verified_margin = str(check.get("marginType") or "").lower()
                verified_leverage = float(check.get("leverage") or 0.0)

                if verified_margin != desired_margin.value or abs(verified_leverage - desired_leverage) > 1e-9:
                    suffix = "open_position" if abs(position_amt) > 1e-12 else "verify_failed"
                    pending_pairs[pair] = (
                        f"{suffix}:margin={verified_margin or 'unknown'},lev={verified_leverage}"
                    )
                    continue

                ready_pairs.add(pair)
                if changed:
                    updated_pairs.append(pair)
            except Exception as exc:
                pending_pairs[pair] = str(exc)

        self._futures_pairs_ready = ready_pairs
        self._futures_pair_pending = pending_pairs
        self._futures_pair_seeded = (not self._futures_open_orders_pending) and len(pending_pairs) == 0

        if updated_pairs:
            logger.info(
                f"[StartupGuard] Seeded futures settings for {len(updated_pairs)} pairs -> margin={desired_margin.value}, leverage={desired_leverage:g}: {', '.join(updated_pairs)}"
            )
        if pending_pairs:
            preview = "; ".join(f"{pair}({reason})" for pair, reason in list(pending_pairs.items())[:8])
            logger.warning(
                f"[StartupGuard] Futures settings still pending for {len(pending_pairs)} pairs: {preview}"
            )
        elif not self._futures_open_orders_pending:
            logger.info(
                f"[StartupGuard] All {len(pair_whitelist)} futures pairs aligned -> margin={desired_margin.value}, leverage={desired_leverage:g}."
            )

    def _maybe_retry_futures_pair_settings(self, current_time) -> None:
        if str((self.config or {}).get("trading_mode") or "").lower() != "futures":
            return
        retry_after = getattr(self, "_futures_pair_retry_after", None)
        current_time_utc = self._to_utc_dt(current_time) or datetime.now(timezone.utc)
        if getattr(self, "_futures_pair_seeded", False) and not getattr(self, "_futures_open_orders_pending", False):
            return
        if retry_after and current_time_utc < retry_after:
            return
        self._ensure_futures_pair_settings(force=True)
        retry_minutes = float(getattr(self, "futures_settings_retry_minutes", 30) or 30)
        self._futures_pair_retry_after = current_time_utc + timedelta(minutes=retry_minutes)

    def _guard_entry_pair(self, pair: str, current_time) -> bool:
        self._maybe_retry_futures_pair_settings(current_time)
        if self._entry_allowed_for_pair(pair):
            return True
        current_time_utc = self._to_utc_dt(current_time) or datetime.now(timezone.utc)
        last_logs = getattr(self, "_last_entry_guard_log_at", {})
        last_logged_at = last_logs.get(pair)
        if last_logged_at and current_time_utc - last_logged_at < timedelta(minutes=5):
            return False
        if getattr(self, "_futures_open_orders_pending", False):
            reason = "open_orders_pending"
        else:
            reason = (getattr(self, "_futures_pair_pending", {}) or {}).get(pair, "futures_settings_unverified")
        logger.warning(f"[EntryGuard] Block entry for {pair}: {reason}")
        last_logs[pair] = current_time_utc
        self._last_entry_guard_log_at = last_logs
        return False

    def bot_start(self, **kwargs) -> None:
        super().bot_start(**kwargs)
        # Use the loosest base stoploss on startup.
        # Open trades may belong to older leverage buckets, and custom_stoploss
        # will tighten each trade back to its own leverage-specific hard stop.
        self.stoploss = -float(self.max_hard_stoploss)
        self._futures_pair_seeded = False
        self._futures_pair_retry_after = datetime.now(timezone.utc)
        self._futures_pairs_ready = set()
        self._futures_pair_pending = {}
        self._futures_open_orders_pending = False
        self._last_entry_guard_log_at = {}
        self._ensure_futures_pair_settings(force=True)

    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        """
        RL 需要原始 OHLC 价格列用于环境价格回放。
        """
        dataframe = super().feature_engineering_standard(dataframe, **kwargs)
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe["&-action"] = 0
        return dataframe

    @staticmethod
    def _predict_mask(dataframe: DataFrame) -> pd.Series:
        if "do_predict" in dataframe.columns:
            return dataframe["do_predict"].fillna(0).astype(float).astype(int) == 1
        return pd.Series(True, index=dataframe.index)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        btc_filter_on = int(self._num(self.btc_filter_enabled, 1)) == 1

        # 候选信号（与 NoT3 保持一致）
        candidate_long = (
            (dataframe["trend_up"])
            & (dataframe["near_support"])
            & (dataframe["volume"] > 0)
            & (dataframe["btc_trend_up"] if btc_filter_on else True)
        )
        candidate_short = (
            (dataframe["trend_down"])
            & (dataframe["near_resistance"])
            & (dataframe["volume"] > 0)
            & (dataframe["btc_trend_down"] if btc_filter_on else True)
        )

        pair = (metadata or {}).get("pair", "")
        reference_time = None
        if len(dataframe) > 0 and "date" in dataframe.columns:
            reference_time = self._to_utc_dt(dataframe["date"].iloc[-1])

        block_long, long_streak, long_age, long_cooldown = self._should_block_reentry(
            pair, "long", reference_time
        )
        block_short, short_streak, short_age, short_cooldown = self._should_block_reentry(
            pair, "short", reference_time
        )

        if block_long:
            if bool(candidate_long.iloc[-1]):
                logger.info(
                    f"[EntryGuard] Blocked {pair} long re-entry after {long_streak} recent long losses; "
                    f"last loss {long_age:.1f}m ago < cooldown {long_cooldown:.0f}m"
                )
            candidate_long = candidate_long & False

        if block_short:
            if bool(candidate_short.iloc[-1]):
                logger.info(
                    f"[EntryGuard] Blocked {pair} short re-entry after {short_streak} recent short losses; "
                    f"last loss {short_age:.1f}m ago < cooldown {short_cooldown:.0f}m"
                )
            candidate_short = candidate_short & False

        dataframe["long_signal"] = candidate_long.astype(int)
        dataframe["short_signal"] = candidate_short.astype(int)

        if "&-action" in dataframe.columns:
            predict_mask = self._predict_mask(dataframe)
            action = dataframe["&-action"].fillna(0).astype(float).astype(int)

            rl_long = candidate_long & predict_mask & (action == 1)
            rl_short = candidate_short & predict_mask & (action == 3)

            dataframe.loc[rl_long, "enter_long"] = 1
            dataframe.loc[rl_long, "enter_tag"] = "rl_long"

            dataframe.loc[rl_short, "enter_short"] = 1
            dataframe.loc[rl_short, "enter_tag"] = "rl_short"
        else:
            # 无 RL 动作列时，回退到候选信号直连
            dataframe.loc[candidate_long, "enter_long"] = 1
            dataframe.loc[candidate_long, "enter_tag"] = "rl_candidate_long"

            dataframe.loc[candidate_short, "enter_short"] = 1
            dataframe.loc[candidate_short, "enter_tag"] = "rl_candidate_short"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None

        if "&-action" in dataframe.columns:
            predict_mask = self._predict_mask(dataframe)
            action = dataframe["&-action"].fillna(0).astype(float).astype(int)

            # 允许 RL 平仓动作在 do_predict=0 时也能生效，避免重启/异常检测窗口期“死扛到硬止损”。
            exit_long = action == 2
            exit_short = action == 4

            dataframe.loc[exit_long, "exit_long"] = 1
            dataframe.loc[exit_short, "exit_short"] = 1

            # exit_tag 区分预测有效与否，便于事后审计（仅对 do_predict 存在时生效）
            if "do_predict" in dataframe.columns:
                exit_long_pred = exit_long & predict_mask
                exit_long_nopredict = exit_long & (~predict_mask)
                exit_short_pred = exit_short & predict_mask
                exit_short_nopredict = exit_short & (~predict_mask)

                dataframe.loc[exit_long_pred, "exit_tag"] = "rl_exit_long"
                dataframe.loc[exit_long_nopredict, "exit_tag"] = "rl_exit_long_nopredict"
                dataframe.loc[exit_short_pred, "exit_tag"] = "rl_exit_short"
                dataframe.loc[exit_short_nopredict, "exit_tag"] = "rl_exit_short_nopredict"
            else:
                dataframe.loc[exit_long, "exit_tag"] = "rl_exit_long"
                dataframe.loc[exit_short, "exit_tag"] = "rl_exit_short"

        return dataframe

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        """
        RL ???????? AI ???????&-s_close ?????
        ??? RL ???trailing stop?????ROI ?? 46h ?????????
        """
        try:
            open_dt = getattr(trade, 'open_date_utc', None) or getattr(trade, 'open_date', None)
            if open_dt is None or current_time is None:
                return None
            age_hours = max((current_time - open_dt).total_seconds(), 0.0) / 3600.0
            max_hold_hours = float(getattr(self, 'max_hold_hours', 46.0) or 46.0)
            if age_hours > max_hold_hours:
                logger.info(
                    f"[RiskExit] Force exiting {pair} after {age_hours:.2f}h > {max_hold_hours:.2f}h"
                )
                return 'max_hold_46h'

            soft_fraction = float(getattr(self, "soft_stoploss_fraction", 0.75) or 0.0)
            min_hold_minutes = float(getattr(self, "soft_stoploss_min_hold_minutes", 30.0) or 0.0)
            if soft_fraction > 0 and current_profit is not None and age_hours * 60.0 >= min_hold_minutes:
                trade_leverage = float(getattr(trade, "leverage", 1.0) or 1.0)
                hard_stop = abs(self._hard_stoploss_for_leverage(trade_leverage))
                soft_stop = -(hard_stop * soft_fraction)
                if current_profit <= soft_stop:
                    latest_action = None
                    latest_do_predict = None
                    try:
                        if getattr(self, "dp", None):
                            analyzed = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                            dataframe = analyzed[0] if isinstance(analyzed, tuple) else analyzed
                            if dataframe is not None and not dataframe.empty:
                                latest = dataframe.iloc[-1]
                                latest_action = latest.get("&-action")
                                latest_do_predict = latest.get("do_predict")
                    except Exception:
                        latest_action = None
                        latest_do_predict = None

                    logger.info(
                        f"[RiskExit] Soft stop exiting {pair}: "
                        f"profit={current_profit:.2%} <= {soft_stop:.2%} "
                        f"(hard={hard_stop:.2%}, leverage={trade_leverage:.1f}x, age={age_hours:.2f}h, "
                        f"action={latest_action}, do_predict={latest_do_predict})"
                    )
                    return f"soft_stop_{int(round(abs(soft_stop) * 100.0))}pct"
        except Exception as exc:
            logger.warning(f"[RiskExit] Failed to evaluate max hold exit for {pair}: {exc}")
        return None

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        """
        RL 版本不使用父类的 AI 回归预测防守（&-s_close 不存在）。
        入场完全由 RL 动作直接决定，无需退回检查回归阈值。
        """
        return True

    def custom_stoploss(self, pair: str, trade, current_time,
                        current_rate: float, current_profit: float, after_fill: bool, **kwargs):
        """
        ?????
        - ????? trailing ???????????????
        - ??????? open_profit ?? trailing??????????????
        """
        try:
            from freqtrade.strategy import stoploss_from_open
        except Exception:
            return None

        trade_leverage = float(getattr(trade, "leverage", 1.0) or 1.0)
        hard_stop = self._hard_stoploss_for_leverage(trade_leverage)
        hard_sl_rel = stoploss_from_open(
            open_relative_stop=hard_stop,
            current_profit=current_profit,
            is_short=getattr(trade, "is_short", False),
            leverage=trade_leverage,
        )

        # ?? Parameter / float ??????
        if hasattr(self, "_num"):
            trailing_positive = float(self._num(getattr(self, "trailing_stop_positive_opt", getattr(self, "trailing_stop_positive", 0.015)), 0.015))
            trailing_offset = float(self._num(getattr(self, "trailing_stop_positive_offset_opt", getattr(self, "trailing_stop_positive_offset", 0.025)), 0.025))
        else:
            tp = getattr(self, "trailing_stop_positive", 0.015)
            to = getattr(self, "trailing_stop_positive_offset", 0.025)
            trailing_positive = float(tp.value) if hasattr(tp, "value") else float(tp)
            trailing_offset = float(to.value) if hasattr(to, "value") else float(to)

        # ?? offset ? positive ???????? 0.017???????
        min_profit_gap = 0.005
        if trailing_offset <= trailing_positive + min_profit_gap:
            trailing_offset = trailing_positive + min_profit_gap

        # ????? trailing ????? stoploss?????????????
        if current_profit < trailing_offset:
            return hard_sl_rel

        try:
            target_open_profit = current_profit - trailing_positive
            if target_open_profit <= 0:
                return hard_sl_rel

            sl_rel = stoploss_from_open(
                open_relative_stop=target_open_profit,
                current_profit=current_profit,
                is_short=getattr(trade, "is_short", False),
                leverage=trade_leverage,
            )

            min_current_distance = 0.010
            if sl_rel < min_current_distance:
                sl_rel = min_current_distance
            return min(hard_sl_rel, sl_rel)
        except Exception:
            return hard_sl_rel

    def _record_trade_experience(self, pair: str, trade, exit_reason: str, current_time) -> None:
        """统一经验记录入口：成交后写入，避免 pre-fill 时 close_rate/profit 为空。"""
        try:
            lev = self._experience_leverage(trade)
            exp_buffer = self._get_exp_buffer(lev)
            if not exp_buffer:
                return
            trade_id = int(getattr(trade, "id", 0) or 0)
            if trade_id > 0 and self._experience_recorded(pair, trade_id, lev):
                logger.info(
                    f"[DogeAI_RL] Experience skip duplicate from buffer for {pair}: "
                    f"trade_id={trade_id}"
                )
                return

            dataframe = None
            if getattr(self, "dp", None):
                analyzed = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                dataframe = analyzed[0] if isinstance(analyzed, tuple) else analyzed

            exp_buffer.record_trade(
                pair=pair,
                trade=trade,
                exit_reason=exit_reason,
                current_time=current_time,
                dataframe=dataframe,
            )
            logger.info(f"[Strategy] Trade experience recorded for {pair} (leverage={lev})")
        except Exception as e:
            logger.debug(f"[DogeAI_RL] Failed to record trade experience: {e}")

    def bot_loop_start(self, current_time, **kwargs) -> None:
        parent = getattr(super(), "bot_loop_start", None)
        if callable(parent):
            try:
                parent(current_time=current_time, **kwargs)
            except TypeError:
                parent(current_time, **kwargs)
        self._reconcile_missing_trade_experiences(current_time)
        self._maybe_retry_futures_pair_settings(current_time)

    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time, **kwargs) -> bool:
        """
        仅确认出场，不在这里记录经验（此时交易未必成交，字段可能为空）。
        """
        if exit_reason in {"rl_exit_long", "rl_exit_short"}:
            current_profit = None
            try:
                current_profit = float(trade.calc_profit_ratio(rate))
            except Exception:
                try:
                    current_profit = float(kwargs.get("current_profit"))
                except Exception:
                    current_profit = None

            if current_profit is not None:
                side = 'short' if exit_reason == 'rl_exit_short' else 'long'
                deadband_min = float(
                    getattr(self, f'rl_exit_{side}_deadband_min', getattr(self, 'rl_exit_deadband_min', -0.0020))
                )
                deadband_max = float(
                    getattr(self, f'rl_exit_{side}_deadband_max', getattr(self, 'rl_exit_deadband_max', 0.0035))
                )
                if deadband_min < current_profit < deadband_max:
                    logger.info(
                        f"[Strategy] Blocked {exit_reason} for {pair}: "
                        f"profit={current_profit:.4%} inside fee deadband "
                        f"({deadband_min:.2%}, {deadband_max:.2%})"
                    )
                    return False
        return True

    def order_filled(self, pair: str, trade, order, current_time, **kwargs) -> None:
        """
        在订单成交后记录经验，确保 exit_price / profit_ratio 使用真实成交结果。
        仅在完整平仓时记录一次。
        """
        try:
            exit_side = trade.exit_side if hasattr(trade, "exit_side") else ("buy" if getattr(trade, "is_short", False) else "sell")
            ft_side = getattr(order, "ft_order_side", "")
            is_exit_order = ft_side in [exit_side, "stoploss"]
            is_trade_closed = not getattr(trade, "is_open", True)

            if not is_exit_order:
                logger.info(
                    f"[Strategy] Experience skip for {pair}: not exit order "
                    f"(ft_side={ft_side}, exit_side={exit_side})"
                )
                return

            # 实盘中 order_filled 与 trade.is_open 状态可能存在时序差，
            # 对已识别为退出订单的场景直接尝试记录经验。
            if not is_trade_closed:
                logger.info(
                    f"[Strategy] Experience record proceed though trade still open for {pair}: "
                    f"trade_id={getattr(trade, 'id', 'na')} ft_side={ft_side}"
                )

            # 去重：同一 trade 仅记录一次
            try:
                if hasattr(trade, "get_custom_data") and trade.get_custom_data("experience_recorded", False):
                    logger.info(
                        f"[Strategy] Experience skip duplicate for {pair}: "
                        f"trade_id={getattr(trade, 'id', 'na')}"
                    )
                    return
            except Exception:
                pass

            exit_reason = str(getattr(trade, "exit_reason", "") or "filled_exit")
            self._record_trade_experience(pair, trade, exit_reason, current_time)
            self._apply_small_win_reentry_cooldown(pair, trade, order, exit_reason, current_time)

            try:
                if hasattr(trade, "set_custom_data"):
                    trade.set_custom_data("experience_recorded", True)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"[Strategy] order_filled experience hook failed: {e}")
