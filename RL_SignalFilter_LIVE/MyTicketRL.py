"""
MyTicketRL - 按单进化版强化学习模型 (支持经验回放)
====================================================
继承 freqtrade 内置 ReinforcementLearner，
增加 Experience Replay 自进化能力。

核心设计思路:
1. 盈利平仓 → 放大奖励 (win_reward_factor)
2. 亏损平仓 → 等比例惩罚
3. 持仓期间 → 时间惩罚 + 回撤惩罚
4. 经验回放 → 从实盘交易中学习
"""

import logging
import numpy as np
import os
import sys

try:
    from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
except ImportError:
    from freqtrade.freqai.RL.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions

# 添加 freqaimodels 目录到路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from ExperienceBuffer import ExperienceBuffer
except ImportError:
    ExperienceBuffer = None

logger = logging.getLogger(__name__)


class MyTicketRL(ReinforcementLearner):
    """
    按单（Ticket）进化版 RL 模型，支持经验回放。
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        # Experience Replay 配置
        er_config = self.rl_config.get("experience_replay", {})
        self.er_enabled = er_config.get("enabled", True)
        self.max_experiences = er_config.get("max_experiences", 500)
        self.min_experiences = er_config.get("min_experiences_to_start", 10)
        
        # 初始化经验缓冲区
        if ExperienceBuffer and self.er_enabled:
            self.exp_buffer = ExperienceBuffer(
                storage_dir="user_data/experience_replay",
                max_experiences=self.max_experiences,
            )
            logger.info(
                f"[MyTicketRL] Experience replay enabled: "
                f"max_experiences={self.max_experiences}, "
                f"min_experiences={self.min_experiences}"
            )
        else:
            self.exp_buffer = None
            logger.info("[MyTicketRL] Experience replay disabled")

    def _load_replay_experiences(self):
        """加载经验回放数据"""
        if not self.exp_buffer:
            return []
        
        experiences = self.exp_buffer.load_experiences(min_count=self.min_experiences)
        if experiences:
            stats = self.exp_buffer.get_stats()
            logger.info(
                f"[MyTicketRL] Experience buffer: "
                f"total={stats.get('total', 0)}, "
                f"win_rate={stats.get('win_rate', 0):.2%}, "
                f"avg_profit={stats.get('avg_profit', 0):.4f}"
            )
        return experiences

    class MyRLEnv(Base5ActionRLEnv):
        """
        自定义 RL 环境，覆写 calculate_reward。
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._replay_experiences = []
            self._replay_lookup = {}
            
            # 尝试从模型加载经验
            if hasattr(self, "parent_model") and hasattr(self.parent_model, "_load_replay_experiences"):
                self._replay_experiences = self.parent_model._load_replay_experiences()
                if self._replay_experiences:
                    self._build_replay_lookup()

        def _build_replay_lookup(self):
            """构建经验查找表"""
            lookup = {
                "long_win": [],
                "long_loss": [],
                "short_win": [],
                "short_loss": [],
            }
            for exp in self._replay_experiences:
                side = exp.get("side", "long")
                profit = exp.get("profit_ratio", 0)
                key = f"{side}_{'win' if profit > 0 else 'loss'}"
                lookup[key].append(exp)
            
            self._replay_lookup = lookup
            total = sum(len(v) for v in lookup.values())
            if total > 0:
                logger.info(
                    f"[MyRLEnv] Replay lookup: "
                    f"long_win={len(lookup['long_win'])}, "
                    f"long_loss={len(lookup['long_loss'])}, "
                    f"short_win={len(lookup['short_win'])}, "
                    f"short_loss={len(lookup['short_loss'])}"
                )

        def calculate_reward(self, action: int) -> float:
            """
            Ticket-style reward function with replay bonus.
            """
            rr = self.rl_config.get("model_reward_parameters", {})
            profit_aim = rr.get("profit_aim", 0.025)
            win_factor = rr.get("win_reward_factor", 2.0)
            time_pen = rr.get("time_penalty", 0.0001)
            dd_pen = rr.get("dd_penalty", 0.02)

            factor = 100.0
            pnl = self.get_unrealized_profit()

            trade_duration = 0
            if self._last_trade_tick is not None:
                trade_duration = self._current_tick - self._last_trade_tick

            # --- 空仓状态 ---
            if self._position == Positions.Neutral:
                if action == Actions.Long_enter.value or action == Actions.Short_enter.value:
                    replay_bonus = self._get_entry_replay_bonus(action)
                    return replay_bonus
                elif action == Actions.Long_exit.value or action == Actions.Short_exit.value:
                    return -1.0
                else:
                    return 0.0

            # --- 多头持仓 ---
            elif self._position == Positions.Long:
                if action == Actions.Long_exit.value:
                    base_reward = self._ticket_reward(pnl, factor, win_factor, profit_aim)
                    replay_bonus = self._get_exit_replay_bonus("long", pnl)
                    return base_reward + replay_bonus

                elif action == Actions.Short_enter.value:
                    if pnl > 0:
                        return float(pnl * factor * win_factor * 0.8)
                    else:
                        return float(pnl * factor)

                elif action == Actions.Long_enter.value:
                    return -1.0

                else:
                    reward = -time_pen * trade_duration
                    if pnl < 0:
                        leverage = getattr(self, '_leverage', 3.0)
                        price_change = pnl / leverage if leverage > 0 else pnl
                        abs_price_change = abs(price_change)

                        if abs_price_change <= 0.02:
                            weight = 0.5
                        elif abs_price_change <= 0.04:
                            weight = 1.0
                        elif abs_price_change <= 0.06:
                            weight = 2.0
                        else:
                            weight = 3.0

                        reward -= dd_pen * abs(pnl) * factor * weight
                    elif pnl > 0:
                        reward += pnl * factor * 0.01
                    return float(reward)

            # --- 空头持仓 ---
            elif self._position == Positions.Short:
                if action == Actions.Short_exit.value:
                    base_reward = self._ticket_reward(pnl, factor, win_factor, profit_aim)
                    replay_bonus = self._get_exit_replay_bonus("short", pnl)
                    return base_reward + replay_bonus

                elif action == Actions.Long_enter.value:
                    if pnl > 0:
                        return float(pnl * factor * win_factor * 0.8)
                    else:
                        return float(pnl * factor)

                elif action == Actions.Short_enter.value:
                    return -1.0

                else:
                    reward = -time_pen * trade_duration
                    if pnl < 0:
                        leverage = getattr(self, '_leverage', 3.0)
                        price_change = pnl / leverage if leverage > 0 else pnl
                        abs_price_change = abs(price_change)

                        if abs_price_change <= 0.02:
                            weight = 0.5
                        elif abs_price_change <= 0.04:
                            weight = 1.0
                        elif abs_price_change <= 0.06:
                            weight = 2.0
                        else:
                            weight = 3.0

                        reward -= dd_pen * abs(pnl) * factor * weight
                    elif pnl > 0:
                        reward += pnl * factor * 0.01
                    return float(reward)

            return 0.0

        def _ticket_reward(self, pnl: float, factor: float,
                          win_factor: float, profit_aim: float) -> float:
            """标准 Ticket-style 奖励"""
            if pnl > 0:
                reward = pnl * factor * win_factor
                if pnl >= profit_aim:
                    reward *= 1.5
                return float(reward)
            else:
                leverage = getattr(self, '_leverage', 3.0)
                price_change = pnl / leverage if leverage > 0 else pnl
                abs_price_change = abs(price_change)

                if abs_price_change <= 0.02:
                    weight = 0.5
                elif abs_price_change <= 0.04:
                    weight = 1.0
                elif abs_price_change <= 0.06:
                    weight = 2.0
                else:
                    weight = 3.0

                return float(pnl * factor * weight)

        def _get_entry_replay_bonus(self, action: int) -> float:
            """入场经验奖励"""
            if not self._replay_experiences:
                return 0.0

            is_long = action == Actions.Long_enter.value
            side = "long" if is_long else "short"

            wins = self._replay_lookup.get(f"{side}_win", [])
            losses = self._replay_lookup.get(f"{side}_loss", [])

            total = len(wins) + len(losses)
            if total < 3:
                return 0.0

            win_rate = len(wins) / total

            if win_rate < 0.35:
                return -0.5
            elif win_rate > 0.65:
                return 0.2
            else:
                return 0.0

        def _get_exit_replay_bonus(self, side: str, pnl: float) -> float:
            """平仓经验奖励"""
            if not self._replay_experiences:
                return 0.0

            if pnl < 0:
                similar_losses = self._replay_lookup.get(f"{side}_loss", [])
                if len(similar_losses) >= 3:
                    avg_loss = np.mean([abs(e.get("profit_ratio", 0)) for e in similar_losses])
                    return float(-avg_loss * 50)
                return 0.0
            else:
                similar_wins = self._replay_lookup.get(f"{side}_win", [])
                if len(similar_wins) >= 3:
                    avg_win = np.mean([e.get("profit_ratio", 0) for e in similar_wins])
                    return float(avg_win * 20)
                return 0.0
