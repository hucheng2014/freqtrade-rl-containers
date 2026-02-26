"""
PerformanceTracker — 实盘表现跟踪与奖励参数自适应
====================================================
根据近期实盘交易表现，动态调整 RL 奖励函数参数。

核心逻辑:
- 胜率低 → 增大 win_factor 和 dd_penalty → 更保守
- 胜率高 → 轻微降低 win_factor → 适当探索
- 止损率高 → 增大 dd_penalty → 惩罚回撤
- 持仓过长 → 增大 time_penalty → 鼓励快速决策
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# 参数安全范围，防止极端漂移
PARAM_BOUNDS = {
    "win_reward_factor": (1.0, 5.0),
    "dd_penalty": (0.005, 0.1),
    "time_penalty": (0.00005, 0.001),
    "profit_aim": (0.01, 0.05),
}

# 默认基准参数
DEFAULT_PARAMS = {
    "win_reward_factor": 2.0,
    "dd_penalty": 0.02,
    "time_penalty": 0.0001,
    "profit_aim": 0.025,
}


class PerformanceTracker:
    """
    实盘表现跟踪器。

    从 ExperienceBuffer 读取交易记录，
    计算胜率/盈亏比/止损率等指标，
    输出自适应的奖励参数。
    """

    def __init__(
        self,
        storage_dir: str = "user_data/experience_replay",
        lookback_trades: int = 50,
        adjustment_rate: float = 0.1,
        history_file: str = "user_data/performance_history.json",
    ):
        self.storage_dir = Path(storage_dir)
        self.lookback_trades = lookback_trades
        self.adjustment_rate = adjustment_rate
        self.history_file = Path(history_file)
        self._adjustment_history: list[dict] = []

    def compute_adaptive_params(
        self,
        experiences: list[dict],
        base_params: Optional[dict] = None,
    ) -> dict:
        """
        根据实盘表现计算自适应奖励参数。

        :param experiences: 交易经验列表 (来自 ExperienceBuffer)
        :param base_params: 基准参数 (来自 config)
        :return: 调整后的奖励参数
        """
        if base_params is None:
            base_params = DEFAULT_PARAMS.copy()

        params = base_params.copy()

        # 取最近 N 笔交易
        recent = experiences[-self.lookback_trades:]

        if len(recent) < 10:
            logger.info(
                f"[PerformanceTracker] Only {len(recent)} trades, "
                f"need ≥10. Using base params."
            )
            return params

        # ====== 计算实盘指标 ======
        metrics = self._compute_metrics(recent)

        logger.info(
            f"[PerformanceTracker] Metrics: "
            f"win_rate={metrics['win_rate']:.2%} "
            f"avg_profit={metrics['avg_profit']:.4f} "
            f"avg_win={metrics['avg_win']:.4f} "
            f"avg_loss={metrics['avg_loss']:.4f} "
            f"stoploss_rate={metrics['stoploss_rate']:.2%} "
            f"avg_duration_h={metrics['avg_duration_hours']:.1f}"
        )

        # ====== 根据指标调整参数 ======
        adjustments = {}
        rate = self.adjustment_rate

        # 规则 1: 胜率过低 → 更保守
        if metrics["win_rate"] < 0.40:
            adj_win = 1.0 + 0.2 * rate
            adj_dd = 1.0 + 0.3 * rate
            params["win_reward_factor"] *= adj_win
            params["dd_penalty"] *= adj_dd
            adjustments["low_winrate"] = {
                "win_reward_factor": f"×{adj_win:.3f}",
                "dd_penalty": f"×{adj_dd:.3f}",
            }

        # 规则 2: 胜率较高 → 适当放松
        elif metrics["win_rate"] > 0.60:
            adj_win = 1.0 - 0.1 * rate
            params["win_reward_factor"] *= adj_win
            adjustments["high_winrate"] = {
                "win_reward_factor": f"×{adj_win:.3f}",
            }

        # 规则 3: 平均亏损 > 平均盈利 → 降低目标
        if metrics["avg_win"] > 0 and abs(metrics["avg_loss"]) > metrics["avg_win"]:
            adj_aim = 1.0 - 0.1 * rate
            params["profit_aim"] *= adj_aim
            adjustments["loss_gt_win"] = {
                "profit_aim": f"×{adj_aim:.3f}",
            }

        # 规则 4: 平均持仓过长 (> 48h)
        if metrics["avg_duration_hours"] > 48:
            adj_time = 1.0 + 0.5 * rate
            params["time_penalty"] *= adj_time
            adjustments["long_duration"] = {
                "time_penalty": f"×{adj_time:.3f}",
            }

        # 规则 5: 止损率过高 (> 30%)
        if metrics["stoploss_rate"] > 0.30:
            adj_dd = 1.0 + 0.5 * rate
            params["dd_penalty"] *= adj_dd
            adjustments["high_stoploss"] = {
                "dd_penalty": f"×{adj_dd:.3f}",
            }

        # ====== 应用安全边界 ======
        for key, (lo, hi) in PARAM_BOUNDS.items():
            if key in params:
                params[key] = float(np.clip(params[key], lo, hi))

        # ====== 记录调整历史 ======
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trade_count": len(recent),
            "metrics": metrics,
            "adjustments": adjustments,
            "final_params": {k: round(v, 6) for k, v in params.items()},
        }
        self._adjustment_history.append(record)
        self._save_history(record)

        if adjustments:
            logger.info(
                f"[PerformanceTracker] Adjusted params: "
                f"{json.dumps({k: round(v, 5) for k, v in params.items()})}"
            )
        else:
            logger.info("[PerformanceTracker] No adjustment needed, using base params.")

        return params

    def _compute_metrics(self, trades: list[dict]) -> dict:
        """计算实盘交易指标。"""
        profits = [t.get("profit_ratio", 0) for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        durations = [t.get("duration_seconds", 0) for t in trades]

        stoploss_count = sum(
            1 for t in trades
            if "stoploss" in str(t.get("exit_reason", "")).lower()
        )

        return {
            "win_rate": len(wins) / len(trades) if trades else 0,
            "avg_profit": float(np.mean(profits)) if profits else 0,
            "avg_win": float(np.mean(wins)) if wins else 0,
            "avg_loss": float(np.mean(losses)) if losses else 0,
            "max_profit": float(max(profits)) if profits else 0,
            "max_loss": float(min(profits)) if profits else 0,
            "stoploss_rate": stoploss_count / len(trades) if trades else 0,
            "stoploss_count": stoploss_count,
            "avg_duration_hours": float(np.mean(durations)) / 3600 if durations else 0,
            "total_trades": len(trades),
        }

    def _save_history(self, record: dict) -> None:
        """保存调整历史到文件。"""
        try:
            history = []
            if self.history_file.exists():
                with open(self.history_file, "r") as f:
                    history = json.load(f)

            history.append(record)

            # 只保留最近 100 条
            if len(history) > 100:
                history = history[-100:]

            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, "w") as f:
                json.dump(history, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[PerformanceTracker] Failed to save history: {e}")
