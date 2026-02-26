"""
ExperienceBuffer — 实盘交易经验记录与回放系统
================================================
记录每笔实盘交易的完整生命周期数据，供 RL 重训练时作为
experience replay 使用。

核心设计:
1. 每笔交易关闭时，record_trade() 保存完整经验到 JSON
2. 重训练时，load_experiences() 加载所有经验
3. sample_replay_batch() 按权重采样（亏损交易权重更高）
4. cleanup_old() 定期清理过期数据
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """
    实盘交易经验缓冲区。

    数据持久化到 storage_dir 目录下的 JSON 文件，
    每个交易对一个文件，避免单文件过大。
    """

    def __init__(
        self,
        storage_dir: str = "user_data/experience_replay",
        max_experiences: int = 500,
        max_age_days: int = 90,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_experiences = max_experiences
        self.max_age_days = max_age_days
        self._cache: list[dict] = []
        self._cache_loaded = False

    def record_trade(
        self,
        pair: str,
        trade: Any,
        exit_reason: str,
        current_time: datetime,
        dataframe: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        记录一笔完成的交易经验。

        在策略的 confirm_trade_exit 中调用。
        """
        try:
            # 基础交易信息（兼容 emergency_exit / 提前回调场景）
            exit_price = 0.0
            for attr in ("close_rate", "close_rate_requested"):
                val = getattr(trade, attr, None)
                if val is None:
                    continue
                try:
                    fv = float(val)
                    if fv > 0:
                        exit_price = fv
                        break
                except Exception:
                    continue

            profit_ratio = None
            close_profit = getattr(trade, "close_profit", None)
            if close_profit is not None:
                try:
                    profit_ratio = float(close_profit)
                except Exception:
                    profit_ratio = None

            if profit_ratio is None and exit_price > 0 and hasattr(trade, "calc_profit_ratio"):
                try:
                    profit_ratio = float(trade.calc_profit_ratio(exit_price))
                except Exception:
                    profit_ratio = None

            if profit_ratio is None:
                profit_ratio = 0.0

            open_dt = getattr(trade, "open_date_utc", None) or getattr(trade, "open_date", None)
            if isinstance(current_time, datetime) and isinstance(open_dt, datetime):
                duration_seconds = int((current_time - open_dt).total_seconds())
                if duration_seconds < 0:
                    duration_seconds = 0
            else:
                duration_seconds = 0

            experience: dict[str, Any] = {
                "trade_id": int(getattr(trade, "id", 0) or 0),
                "pair": pair,
                "side": "short" if trade.is_short else "long",
                "entry_time": trade.open_date_utc.isoformat(),
                "exit_time": current_time.isoformat()
                    if isinstance(current_time, datetime)
                    else str(current_time),
                "entry_price": float(getattr(trade, "open_rate", 0.0) or 0.0),
                "exit_price": exit_price,
                "profit_ratio": profit_ratio,
                "stake_amount": float(getattr(trade, "stake_amount", 0.0) or 0.0),
                "duration_seconds": duration_seconds,
                "exit_reason": str(exit_reason),
                "leverage": float(getattr(trade, "leverage", 1.0) or 1.0),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }

            # 计算持仓K线数（优先使用 trade.timeframe，单位分钟）
            candle_minutes = int(getattr(trade, "timeframe", 15) or 15)
            candle_seconds = max(candle_minutes * 60, 60)
            experience["duration_candles"] = max(
                experience["duration_seconds"] // candle_seconds, 1
            )

            # 提取入场时的特征快照（如果 dataframe 可用）
            if dataframe is not None and len(dataframe) > 0:
                experience["features_at_entry"] = self._extract_features_snapshot(
                    dataframe, trade.open_date_utc
                )
                experience["features_during_trade"] = (
                    self._extract_trade_features_sequence(
                        dataframe, trade.open_date_utc, current_time
                    )
                )
                experience["max_drawdown"], experience["max_profit"] = (
                    self._compute_trade_extremes(
                        dataframe, trade.open_date_utc, current_time,
                        trade.open_rate, trade.is_short
                    )
                )
            else:
                experience["features_at_entry"] = {}
                experience["features_during_trade"] = []
                experience["max_drawdown"] = 0.0
                experience["max_profit"] = 0.0

            # 持久化
            self._save_experience(experience)
            self._cache_loaded = False  # 使缓存失效

            logger.info(
                f"[ExperienceBuffer] Recorded trade: {pair} {experience['side']} "
                f"profit={experience['profit_ratio']:.4f} "
                f"reason={exit_reason} "
                f"duration={experience['duration_candles']}candles"
            )

        except Exception as e:
            logger.error(f"[ExperienceBuffer] Failed to record trade: {e}")

    def load_experiences(
        self,
        pair: Optional[str] = None,
        min_count: int = 0,
    ) -> list[dict]:
        """
        加载所有（或指定交易对的）经验。

        :param pair: 如果指定，只返回该交易对的经验
        :param min_count: 最少需要的经验数，不足则返回空列表
        :return: 经验列表
        """
        if not self._cache_loaded:
            self._cache = self._load_all_experiences()
            self._cache_loaded = True

        experiences = self._cache
        if pair:
            experiences = [e for e in experiences if e.get("pair") == pair]

        if len(experiences) < min_count:
            logger.info(
                f"[ExperienceBuffer] Only {len(experiences)} experiences "
                f"(need {min_count}), skipping replay"
            )
            return []

        return experiences

    def sample_replay_batch(
        self,
        batch_size: int = 50,
        loss_weight: float = 2.0,
        recency_decay: float = 0.95,
    ) -> list[dict]:
        """
        按优先级权重采样经验批次。

        权重规则:
        - 亏损交易权重 = loss_weight (默认 2x)
        - 近期交易权重更高 (指数衰减)
        - 止损触发的交易额外加权 (识别危险模式)
        """
        experiences = self.load_experiences()
        if not experiences:
            return []

        n = len(experiences)
        batch_size = min(batch_size, n)

        # 计算权重
        weights = np.ones(n, dtype=float)
        for i, exp in enumerate(experiences):
            # 亏损交易加权
            if exp.get("profit_ratio", 0) < 0:
                weights[i] *= loss_weight
                # 止损触发额外加权
                if "stoploss" in str(exp.get("exit_reason", "")).lower():
                    weights[i] *= 1.5

            # 时间衰减: 越近的交易权重越高
            age_idx = n - 1 - i  # 假设按时间排序，越后面越新
            weights[i] *= recency_decay ** age_idx

        # 归一化为概率分布
        probs = weights / weights.sum()

        # 按权重采样
        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)
        return [experiences[i] for i in indices]

    def cleanup_old(self) -> int:
        """
        清理超过 max_age_days 的旧记录。
        返回清理的记录数。
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (
            self.max_age_days * 86400
        )
        removed = 0

        for filepath in self.storage_dir.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    records = json.load(f)

                original_count = len(records)
                records = [
                    r for r in records
                    if self._parse_timestamp(r.get("recorded_at", "")) > cutoff
                ]

                removed += original_count - len(records)

                if records:
                    with open(filepath, "w") as f:
                        json.dump(records, f, indent=2)
                else:
                    filepath.unlink()
            except Exception as e:
                logger.warning(f"[ExperienceBuffer] Cleanup error for {filepath}: {e}")

        if removed > 0:
            logger.info(f"[ExperienceBuffer] Cleaned up {removed} old experiences")
            self._cache_loaded = False

        return removed

    def get_stats(self) -> dict:
        """获取经验缓冲区的统计信息。"""
        experiences = self.load_experiences()
        if not experiences:
            return {"total": 0}

        profits = [e.get("profit_ratio", 0) for e in experiences]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        return {
            "total": len(experiences),
            "win_rate": len(wins) / len(experiences) if experiences else 0,
            "avg_profit": float(np.mean(profits)) if profits else 0,
            "avg_win": float(np.mean(wins)) if wins else 0,
            "avg_loss": float(np.mean(losses)) if losses else 0,
            "max_profit": float(max(profits)) if profits else 0,
            "max_loss": float(min(profits)) if profits else 0,
            "stoploss_count": sum(
                1 for e in experiences
                if "stoploss" in str(e.get("exit_reason", "")).lower()
            ),
        }

    # ==================== 内部方法 ====================

    @staticmethod
    def _to_utc_timestamp(value: Any) -> Optional[pd.Timestamp]:
        """将任意时间值安全转换为 UTC 时间戳。"""
        try:
            if value is None:
                return None
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts
        except Exception:
            return None

    def _extract_features_snapshot(
        self, df: pd.DataFrame, entry_time: datetime
    ) -> dict:
        """提取入场时间点的特征快照。"""
        try:
            if "date" not in df.columns:
                return {}

            # 找到最接近入场时间的行
            df_dated = df.copy()
            df_dated["date"] = pd.to_datetime(df_dated["date"], utc=True, errors="coerce")
            df_dated = df_dated.dropna(subset=["date"])
            if df_dated.empty:
                return {}

            entry_ts = self._to_utc_timestamp(entry_time)
            if entry_ts is None:
                entry_ts = df_dated["date"].iloc[-1]

            mask = df_dated["date"] <= entry_ts
            row = df_dated[mask].iloc[-1] if mask.any() else df_dated.iloc[-1]

            # 只提取特征列（以 % 开头的列）
            feature_cols = [c for c in df.columns if c.startswith("%-")]
            snapshot = {}
            for col in feature_cols[:30]:  # 限制数量防止过大
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    try:
                        snapshot[col] = float(val)
                    except Exception:
                        continue

            # 特征列缺失时，回退到核心行情/指标字段，避免空样本
            if not snapshot:
                fallback_cols = [
                    "rsi", "t3", "t3_slope", "t3_diff",
                    "close", "open", "high", "low", "volume",
                ]
                for col in fallback_cols:
                    if col not in row.index:
                        continue
                    val = row.get(col)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue
                    try:
                        snapshot[col] = float(val)
                    except Exception:
                        continue

            return snapshot
        except Exception as e:
            logger.debug(f"[ExperienceBuffer] Feature snapshot extraction failed: {e}")
            return {}

    def _extract_trade_features_sequence(
        self, df: pd.DataFrame, entry_time: datetime, exit_time: datetime
    ) -> list[dict]:
        """提取持仓期间的特征序列（采样，避免过大）。"""
        try:
            if "date" not in df.columns:
                return []

            df_dated = df.copy()
            df_dated["date"] = pd.to_datetime(df_dated["date"], utc=True, errors="coerce")
            df_dated = df_dated.dropna(subset=["date"])
            if df_dated.empty:
                return []

            entry_ts = self._to_utc_timestamp(entry_time) or df_dated["date"].iloc[0]
            exit_ts = self._to_utc_timestamp(exit_time) or df_dated["date"].iloc[-1]
            if exit_ts < entry_ts:
                entry_ts, exit_ts = exit_ts, entry_ts

            mask = (df_dated["date"] >= entry_ts) & (df_dated["date"] <= exit_ts)
            trade_df = df_dated[mask]

            if len(trade_df) == 0:
                # 兜底：若找不到严格区间，退化为最近片段，避免空序列
                trade_df = df_dated.tail(min(20, len(df_dated)))

            # 采样最多 20 个点
            if len(trade_df) > 20:
                indices = np.linspace(0, len(trade_df) - 1, 20, dtype=int)
                trade_df = trade_df.iloc[indices]

            feature_cols = [c for c in df.columns if c.startswith("%-")][:15]
            if not feature_cols:
                feature_cols = [c for c in ["rsi", "t3", "t3_slope", "t3_diff", "volume"] if c in df.columns]

            sequence = []
            for _, row in trade_df.iterrows():
                point = {
                    "close": float(row.get("close", 0) or 0),
                    "date": row.get("date").isoformat() if row.get("date") is not None else "",
                }
                for col in feature_cols:
                    val = row.get(col)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        try:
                            point[col] = float(val)
                        except Exception:
                            continue
                sequence.append(point)

            return sequence
        except Exception as e:
            logger.debug(f"[ExperienceBuffer] Feature sequence extraction failed: {e}")
            return []

    def _compute_trade_extremes(
        self,
        df: pd.DataFrame,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
        is_short: bool,
    ) -> tuple[float, float]:
        """计算交易期间的最大回撤和最大浮盈。"""
        try:
            if "date" not in df.columns or entry_price <= 0:
                return 0.0, 0.0

            df_dated = df.copy()
            df_dated["date"] = pd.to_datetime(df_dated["date"], utc=True, errors="coerce")
            df_dated = df_dated.dropna(subset=["date"])
            if df_dated.empty:
                return 0.0, 0.0

            entry_ts = self._to_utc_timestamp(entry_time) or df_dated["date"].iloc[0]
            exit_ts = self._to_utc_timestamp(exit_time) or df_dated["date"].iloc[-1]
            if exit_ts < entry_ts:
                entry_ts, exit_ts = exit_ts, entry_ts

            mask = (df_dated["date"] >= entry_ts) & (df_dated["date"] <= exit_ts)
            trade_df = df_dated[mask]

            if len(trade_df) == 0:
                trade_df = df_dated.tail(min(50, len(df_dated)))

            closes = pd.to_numeric(trade_df["close"], errors="coerce").dropna()
            if closes.empty:
                return 0.0, 0.0

            if is_short:
                profits = (entry_price - closes) / entry_price
            else:
                profits = (closes - entry_price) / entry_price

            max_dd = float(profits.min())
            max_profit = float(profits.max())
            return max_dd, max_profit

        except Exception as e:
            logger.debug(f"[ExperienceBuffer] Extreme computation failed: {e}")
            return 0.0, 0.0

    def _save_experience(self, experience: dict) -> None:
        """将经验保存到对应交易对的 JSON 文件。"""
        pair_safe = experience["pair"].replace("/", "_").replace(":", "_")
        filepath = self.storage_dir / f"{pair_safe}.json"

        records = []
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    records = json.load(f)
            except (json.JSONDecodeError, Exception):
                records = []

        records.append(experience)

        # 限制每个文件大小
        max_per_pair = self.max_experiences // 5  # 假设约5个交易对
        if len(records) > max_per_pair:
            records = records[-max_per_pair:]

        with open(filepath, "w") as f:
            json.dump(records, f, indent=2, default=str)

    def _load_all_experiences(self) -> list[dict]:
        """从所有 JSON 文件加载经验。"""
        all_experiences = []

        for filepath in sorted(self.storage_dir.glob("*.json")):
            try:
                with open(filepath, "r") as f:
                    records = json.load(f)
                all_experiences.extend(records)
            except Exception as e:
                logger.warning(f"[ExperienceBuffer] Failed to load {filepath}: {e}")

        # 按记录时间排序
        all_experiences.sort(
            key=lambda x: x.get("recorded_at", ""), reverse=False
        )

        # 限制总数
        if len(all_experiences) > self.max_experiences:
            all_experiences = all_experiences[-self.max_experiences:]

        logger.info(
            f"[ExperienceBuffer] Loaded {len(all_experiences)} experiences "
            f"from {len(list(self.storage_dir.glob('*.json')))} files"
        )
        return all_experiences

    @staticmethod
    def _parse_timestamp(ts_str: str) -> float:
        """解析 ISO 时间戳为 Unix 时间戳。"""
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return 0.0
