"""
MyTicketRL - 按单进化版强化学习模型
====================================
继承 freqtrade 内置 ReinforcementLearner，
覆写 calculate_reward() 使其专注于每笔交易的盈亏结果。

核心设计思路:
1. 盈利平仓 → 放大奖励 (win_reward_factor)
2. 亏损平仓 → 等比例惩罚
3. 持仓期间 → 时间惩罚 (time_penalty) + 回撤惩罚 (dd_penalty)
4. 空仓不操作 → 轻微惩罚，鼓励模型积极寻找机会
"""

import logging
import numpy as np

try:
    from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
except ImportError:
    from freqtrade.freqai.RL.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions


logger = logging.getLogger(__name__)


class MyTicketRL(ReinforcementLearner):
    """
    按单（Ticket）进化版 RL 模型。

    与默认 ReinforcementLearner 的区别：
    - calculate_reward 以每笔交易（开→平）的盈亏作为核心奖励信号
    - 盈利交易获得放大奖励，鼓励模型学习"何时是好的入场/出场时机"
    - 持仓时间和浮亏都会产生惩罚，驱动模型学习"及时止盈止损"
    """

    class MyRLEnv(Base5ActionRLEnv):
        """
        自定义 RL 环境，覆写 calculate_reward。
        """

        def calculate_reward(self, action: int) -> float:
            """
            Ticket-style reward function.

            动作空间 (Base5ActionEnvironment):
            - 0: Neutral (不操作)
            - 1: Long_enter (开多)
            - 2: Long_exit (平多)
            - 3: Short_enter (开空)
            - 4: Short_exit (平空)

            奖励设计:
            1. 平仓盈利 → reward = pnl * factor * win_reward_factor
               达到 profit_aim 目标 → 额外 1.5x bonus
            2. 平仓亏损 → reward = pnl * factor (负值)
            3. 持仓中每 step → -time_penalty，若浮亏则额外 -dd_penalty
            4. 空仓不入场 → 0 (中性)
            5. 无效动作 (如空仓时平仓) → -1 惩罚
            """
            # 获取 reward 参数
            rr = self.rl_config.get('model_reward_parameters', {})
            profit_aim = rr.get('profit_aim', 0.025)
            win_factor = rr.get('win_reward_factor', 2.0)
            time_pen = rr.get('time_penalty', 0.0001)
            dd_pen = rr.get('dd_penalty', 0.02)

            factor = 100.0  # 放大系数，让奖励值在合理范围

            # 当前浮盈/浮亏
            pnl = self.get_unrealized_profit()

            # 持仓时长 (candle 数)
            trade_duration = 0
            if self._last_trade_tick is not None:
                trade_duration = self._current_tick - self._last_trade_tick

            # ====== 根据当前仓位状态和动作计算奖励 ======

            # --- 空仓状态 ---
            if self._position == Positions.Neutral:
                if action == Actions.Long_enter.value or action == Actions.Short_enter.value:
                    # 入场：不给即时奖励，让平仓时的 ticket reward 来回溯
                    return 0.0
                elif action == Actions.Long_exit.value or action == Actions.Short_exit.value:
                    # 无效操作：空仓时平仓
                    return -1.0
                else:
                    # 保持空仓（Neutral action）
                    return 0.0

            # --- 多头持仓 ---
            elif self._position == Positions.Long:
                if action == Actions.Long_exit.value:
                    # 平多 → Ticket reward!
                    if pnl > 0:
                        reward = pnl * factor * win_factor
                        if pnl >= profit_aim:
                            reward *= 1.5  # 达到目标的额外奖励
                        return float(reward)
                    else:
                        return float(pnl * factor)  # 亏损等比例惩罚

                elif action == Actions.Short_enter.value:
                    # 多转空：先平多（按 ticket 给分）再开空
                    if pnl > 0:
                        return float(pnl * factor * win_factor * 0.8)
                    else:
                        return float(pnl * factor)

                elif action == Actions.Long_enter.value:
                    # 无效：已在多头，重复开多
                    return -1.0

                else:
                    # 持仓中 (Neutral 或 Short_exit)
                    # 时间惩罚 + 回撤惩罚
                    reward = -time_pen * trade_duration
                    if pnl < 0:
                        reward -= dd_pen * abs(pnl) * factor
                    elif pnl > 0:
                        # 浮盈时给微小正向奖励，鼓励持盈
                        reward += pnl * factor * 0.01
                    return float(reward)

            # --- 空头持仓 ---
            elif self._position == Positions.Short:
                if action == Actions.Short_exit.value:
                    # 平空 → Ticket reward!
                    if pnl > 0:
                        reward = pnl * factor * win_factor
                        if pnl >= profit_aim:
                            reward *= 1.5
                        return float(reward)
                    else:
                        return float(pnl * factor)

                elif action == Actions.Long_enter.value:
                    # 空转多：先平空再开多
                    if pnl > 0:
                        return float(pnl * factor * win_factor * 0.8)
                    else:
                        return float(pnl * factor)

                elif action == Actions.Short_enter.value:
                    # 无效：已在空头，重复开空
                    return -1.0

                else:
                    # 持仓中
                    reward = -time_pen * trade_duration
                    if pnl < 0:
                        reward -= dd_pen * abs(pnl) * factor
                    elif pnl > 0:
                        reward += pnl * factor * 0.01
                    return float(reward)

            return 0.0
