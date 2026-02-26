"""
MyTicketRL_v2 — 自进化版强化学习模型
==========================================
继承 MyTicketRL，增加:
1. Experience Replay: 从实盘交易经验中学习
2. 自适应奖励: 根据实盘表现动态调整奖励参数
3. 混合训练: 70% 历史回放 + 30% 实盘经验回放

核心改造点:
- 覆写 fit(): 两阶段训练
- 覆写 set_train_and_eval_environments(): 注入 ReplayAwareEnv
- 内嵌 MyRLEnv: 支持 replay 模式的奖励计算
"""

import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions

# 将 freqaimodels 目录加入 sys.path 以便导入同目录模块
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from ExperienceBuffer import ExperienceBuffer
from PerformanceTracker import PerformanceTracker

# 尝试导入 MyTicketRL；如果找不到则回退到 ReinforcementLearner
try:
    from MyTicketRL import MyTicketRL as BaseRL
except ImportError:
    try:
        from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner as BaseRL
    except ImportError:
        from freqtrade.freqai.RL.BaseReinforcementLearningModel import (
            BaseReinforcementLearningModel as BaseRL,
        )

logger = logging.getLogger(__name__)


class MyTicketRL_v2(BaseRL):
    """
    自进化版 RL 模型。

    与 MyTicketRL 的关键区别:
    1. fit() 包含 experience replay 阶段
    2. 奖励参数根据实盘表现动态调整
    3. 亏损交易经验在训练中被优先回放
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Experience Replay 配置
        er_config = self.rl_config.get("experience_replay", {})
        self.er_enabled = er_config.get("enabled", True)
        self.replay_ratio = er_config.get("replay_ratio", 0.3)
        self.max_experiences = er_config.get("max_experiences", 500)
        self.loss_weight = er_config.get("loss_weight", 2.0)
        self.min_experiences = er_config.get("min_experiences_to_start", 3)

        # Adaptive Reward 配置
        ar_config = self.rl_config.get("adaptive_reward", {})
        self.ar_enabled = ar_config.get("enabled", True)
        self.ar_lookback = ar_config.get("lookback_trades", 50)
        self.ar_rate = ar_config.get("adjustment_rate", 0.1)

        # 初始化组件
        self.exp_buffer = ExperienceBuffer(
            storage_dir="user_data/experience_replay",
            max_experiences=self.max_experiences,
        )
        self.perf_tracker = PerformanceTracker(
            lookback_trades=self.ar_lookback,
            adjustment_rate=self.ar_rate,
        )

        # 运行时状态
        self._adaptive_params: Optional[dict] = None
        self._replay_experiences: list[dict] = []

        logger.info(
            f"[MyTicketRL_v2] Initialized: "
            f"experience_replay={'ON' if self.er_enabled else 'OFF'}, "
            f"adaptive_reward={'ON' if self.ar_enabled else 'OFF'}, "
            f"replay_ratio={self.replay_ratio}"
        )

    def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        """
        两阶段训练:
        阶段 A: 历史K线模拟环境训练 (标准 PPO)
        阶段 B: 实盘经验回放训练 (Experience Replay)
        """

        # ====== 步骤 1: 加载实盘经验 & 自适应奖励 ======
        if self.er_enabled or self.ar_enabled:
            self._replay_experiences = self.exp_buffer.load_experiences(
                min_count=self.min_experiences
            )

            if self._replay_experiences:
                stats = self.exp_buffer.get_stats()
                logger.info(
                    f"[MyTicketRL_v2] Experience buffer stats: "
                    f"total={stats['total']}, "
                    f"win_rate={stats['win_rate']:.2%}, "
                    f"avg_profit={stats['avg_profit']:.4f}"
                )

                # 自适应奖励参数
                if self.ar_enabled:
                    self._adaptive_params = self.perf_tracker.compute_adaptive_params(
                        self._replay_experiences,
                        base_params=self.reward_params.copy(),
                    )
                    logger.info(
                        f"[MyTicketRL_v2] Adaptive reward params: "
                        f"{self._adaptive_params}"
                    )
            else:
                logger.info(
                    f"[MyTicketRL_v2] No sufficient experiences for replay "
                    f"(need {self.min_experiences}), training with historical data only"
                )

            # 清理过期经验
            self.exp_buffer.cleanup_old()

        # ====== 步骤 2: 获取训练参数 ======
        train_cycles = self.rl_config.get("train_cycles", 20)
        total_timesteps = self.freqai_info.get(
            "model_training_parameters", {}
        ).get("n_steps", self.rl_config.get("n_steps", 1024)) * train_cycles

        # 计算两阶段的 timesteps 分配
        has_replay = bool(self._replay_experiences) and self.er_enabled
        if has_replay:
            phase_a_steps = int(total_timesteps * (1 - self.replay_ratio))
            phase_b_steps = int(total_timesteps * self.replay_ratio)
        else:
            phase_a_steps = total_timesteps
            phase_b_steps = 0

        # ====== 步骤 3: 构建/加载模型 ======
        model_training_params = self.freqai_info.get(
            "model_training_parameters", {}
        )

        if dk.pair not in self.dd.model_dictionary or not self.continual_learning:
            model = self.MODELCLASS(
                self.policy_type,
                self.train_env,
                policy_kwargs=dict(
                    net_arch=self.net_arch,
                ),
                tensorboard_log=(
                    str(Path(dk.full_path / "tensorboard" / dk.pair.split("/")[0]))
                ),
                **model_training_params,
            )
        else:
            logger.info(f"[MyTicketRL_v2] Continual learning from existing model")
            model = self.dd.model_dictionary[dk.pair]
            model.set_env(self.train_env)

        # ====== 步骤 4: 阶段 A — 历史数据训练 ======
        logger.info(
            f"[MyTicketRL_v2] Phase A: Historical training "
            f"({phase_a_steps} timesteps)"
        )
        model.learn(
            total_timesteps=phase_a_steps,
            callback=[self.eval_callback, self.tensorboard_callback],
        )

        # ====== 步骤 5: 阶段 B — 经验回放训练 ======
        if phase_b_steps > 0:
            logger.info(
                f"[MyTicketRL_v2] Phase B: Experience Replay training "
                f"({phase_b_steps} timesteps, "
                f"{len(self._replay_experiences)} experiences)"
            )

            # 构建经验回放环境
            replay_env = self._build_replay_environment(
                data_dictionary, dk
            )

            replay_usable = False
            if replay_env is not None:
                train_obs_shape = getattr(
                    getattr(self.train_env, "observation_space", None),
                    "shape",
                    None,
                )
                replay_obs_shape = getattr(
                    getattr(replay_env, "observation_space", None),
                    "shape",
                    None,
                )
                train_act_n = getattr(
                    getattr(self.train_env, "action_space", None),
                    "n",
                    None,
                )
                replay_act_n = getattr(
                    getattr(replay_env, "action_space", None),
                    "n",
                    None,
                )

                replay_usable = (
                    train_obs_shape == replay_obs_shape
                    and train_act_n == replay_act_n
                )

                if not replay_usable:
                    logger.warning(
                        "[MyTicketRL_v2] Replay env space mismatch: "
                        f"obs {replay_obs_shape} vs {train_obs_shape}, "
                        f"action {replay_act_n} vs {train_act_n}. "
                        "Fallback to Phase A for remaining steps."
                    )

            if replay_usable:
                model.set_env(replay_env)
                model.learn(
                    total_timesteps=phase_b_steps,
                    callback=[self.tensorboard_callback],
                    reset_num_timesteps=False,  # 不重置计数
                )
                logger.info("[MyTicketRL_v2] Phase B complete")

                # 恢复原始训练环境（用于后续 eval）
                model.set_env(self.train_env)
            else:
                logger.warning(
                    "[MyTicketRL_v2] Failed to build replay env, "
                    "using remaining timesteps for Phase A"
                )
                model.learn(
                    total_timesteps=phase_b_steps,
                    callback=[self.eval_callback, self.tensorboard_callback],
                    reset_num_timesteps=False,
                )

        logger.info("[MyTicketRL_v2] Training complete")
        return model

    def _build_replay_environment(
        self,
        data_dictionary: dict[str, DataFrame],
        dk: FreqaiDataKitchen,
    ) -> Any:
        """
        从实盘经验构建回放环境。

        策略: 用最近的历史训练数据作为基础环境，
        但注入实盘经验数据修改奖励信号。
        """
        try:
            train_df = data_dictionary["train_features"]
            if len(train_df) == 0:
                return None

            # 采样经验
            sampled = self.exp_buffer.sample_replay_batch(
                batch_size=min(50, len(self._replay_experiences)),
                loss_weight=self.loss_weight,
            )

            if not sampled:
                return None

            # 在标准环境的基础上构建 replay 环境
            # 使用训练数据的子集，注入经验元数据
            env_info = self.pack_env_dict(dk.pair)

            # 注入经验数据和自适应参数
            env_info["replay_experiences"] = sampled
            if self._adaptive_params:
                env_info["reward_kwargs"] = self._adaptive_params

            # 用与训练环境一致的价格结构，优先复用 train_env.prices
            prices = None
            if hasattr(self, "train_env") and hasattr(self.train_env, "prices"):
                try:
                    prices = self.train_env.prices.copy()
                except Exception:
                    prices = None

            if prices is None or prices.empty:
                rename_dict = {
                    "%-raw_open": "open",
                    "%-raw_low": "low",
                    "%-raw_high": "high",
                    "%-raw_close": "close",
                }
                prices = train_df.filter(rename_dict.keys(), axis=1).rename(
                    columns=rename_dict
                )

            if prices.empty:
                return None

            # 复用与训练环境一致的特征数据
            replay_env = self.MyRLEnv(
                df=train_df,
                prices=prices,
                **env_info,
            )

            return replay_env

        except Exception as e:
            logger.error(f"[MyTicketRL_v2] Failed to build replay env: {e}")
            return None

    # ========================================================
    # 内嵌 RL 环境 — 支持经验回放的奖励计算
    # ========================================================
    class MyRLEnv(Base5ActionRLEnv):
        """
        自进化版 RL 环境。

        在标准 Base5ActionRLEnv 的基础上:
        1. 接收实盘经验数据
        2. 自适应奖励参数
        3. 使用实盘经验增强奖励信号
        """

        def __init__(self, **kwargs):
            # 提取自定义参数
            self.replay_experiences = kwargs.pop("replay_experiences", [])
            super().__init__(**kwargs)

            # 预计算经验回放查找表
            self._replay_lookup = self._build_replay_lookup()
            self._in_replay_trade = False
            self._replay_trade_profit = 0.0

        def _build_replay_lookup(self) -> dict:
            """
            构建经验查找表。

            按 (方向, 盈亏方向) 分组，用于在训练中
            查找类似的历史经验增强奖励信号。
            """
            lookup = {
                "long_win": [],
                "long_loss": [],
                "short_win": [],
                "short_loss": [],
            }
            for exp in self.replay_experiences:
                side = exp.get("side", "long")
                profit = exp.get("profit_ratio", 0)
                key = f"{side}_{'win' if profit > 0 else 'loss'}"
                lookup[key].append(exp)

            total = sum(len(v) for v in lookup.values())
            if total > 0:
                logger.info(
                    f"[ReplayAwareEnv] Replay lookup: "
                    f"long_win={len(lookup['long_win'])}, "
                    f"long_loss={len(lookup['long_loss'])}, "
                    f"short_win={len(lookup['short_win'])}, "
                    f"short_loss={len(lookup['short_loss'])}"
                )
            return lookup

        def calculate_reward(self, action: int) -> float:
            """
            增强版奖励函数。

            在 MyTicketRL 的 ticket-style reward 基础上，
            引入实盘经验回放增强:
            - 如果当前决策模式匹配过去的亏损经验 → 额外惩罚
            - 如果当前决策模式匹配过去的盈利经验 → 额外奖励
            """
            # 获取 reward 参数 (可能已被 PerformanceTracker 调整)
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

            # ====== 空仓状态 ======
            if self._position == Positions.Neutral:
                if action == Actions.Long_enter.value or action == Actions.Short_enter.value:
                    # 入场：检查是否有经验指导
                    replay_bonus = self._get_entry_replay_bonus(action)
                    return replay_bonus
                elif action == Actions.Long_exit.value or action == Actions.Short_exit.value:
                    return -1.0  # 无效操作
                else:
                    return 0.0  # 保持空仓

            # ====== 多头持仓 ======
            elif self._position == Positions.Long:
                if action == Actions.Long_exit.value:
                    # 平多 → Ticket reward + 经验回放增强
                    base_reward = self._ticket_reward(
                        pnl, factor, win_factor, profit_aim
                    )
                    replay_bonus = self._get_exit_replay_bonus(
                        "long", pnl, trade_duration
                    )
                    return base_reward + replay_bonus

                elif action == Actions.Short_enter.value:
                    # 多转空
                    if pnl > 0:
                        return float(pnl * factor * win_factor * 0.8)
                    else:
                        return float(pnl * factor)

                elif action == Actions.Long_enter.value:
                    return -1.0  # 无效: 重复开多

                else:
                    # 持仓中: 使用基于价格变动的权重惩罚
                    reward = -time_pen * trade_duration
                    if pnl < 0:
                        # 亏损时根据价格变动区间给予不同权重惩罚
                        leverage = getattr(self, '_leverage', 3.0)
                        price_change = pnl / leverage if leverage > 0 else pnl
                        abs_price_change = abs(price_change)
                        
                        if abs_price_change <= 0.02:
                            weight = 0.5  # 0-2% 价格变动
                        elif abs_price_change <= 0.04:
                            weight = 1.0  # 2-4% 价格变动
                        elif abs_price_change <= 0.06:
                            weight = 2.0  # 4-6% 价格变动
                        else:
                            weight = 3.0  # >6% 价格变动
                        
                        reward -= dd_pen * abs(pnl) * factor * weight
                    elif pnl > 0:
                        reward += pnl * factor * 0.01
                    return float(reward)

            # ====== 空头持仓 ======
            elif self._position == Positions.Short:
                if action == Actions.Short_exit.value:
                    base_reward = self._ticket_reward(
                        pnl, factor, win_factor, profit_aim
                    )
                    replay_bonus = self._get_exit_replay_bonus(
                        "short", pnl, trade_duration
                    )
                    return base_reward + replay_bonus

                elif action == Actions.Long_enter.value:
                    if pnl > 0:
                        return float(pnl * factor * win_factor * 0.8)
                    else:
                        return float(pnl * factor)

                elif action == Actions.Short_enter.value:
                    return -1.0

                else:
                    # 持仓中: 使用基于价格变动的权重惩罚
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

        def _ticket_reward(
            self, pnl: float, factor: float,
            win_factor: float, profit_aim: float
        ) -> float:
            """
            权重形式减亏奖励计算。
            
            亏损区间权重:
            - 价格变动 0-2% (杠杆后约 0-6%): 权重 0.5
            - 价格变动 2-4% (杠杆后约 6-12%): 权重 1.0
            - 价格变动 4-6% (杠杆后约 12-18%): 权重 2.0
            
            目标: RL 在硬止损触发前(价格变动6%)积极减亏
            """
            # 获取杠杆倍数，将杠杆后盈亏转换为价格变动率
            leverage = getattr(self, '_leverage', 3.0)
            price_change = pnl / leverage if leverage > 0 else pnl
            
            if pnl > 0:
                # 盈利: 标准奖励
                reward = pnl * factor * win_factor
                if pnl >= profit_aim:
                    reward *= 1.5
                return float(reward)
            else:
                # 亏损: 基于价格变动的权重奖励
                abs_price_change = abs(price_change)
                
                if abs_price_change <= 0.02:
                    # 价格变动 0-2%: 权重 0.5 (轻微惩罚，鼓励尽早止损)
                    weight = 0.5
                elif abs_price_change <= 0.04:
                    # 价格变动 2-4%: 权重 1.0 (正常惩罚)
                    weight = 1.0
                elif abs_price_change <= 0.06:
                    # 价格变动 4-6%: 权重 2.0 (重度惩罚，即将触发硬止损)
                    weight = 2.0
                else:
                    # 超过 6% (不应发生，硬止损应该已触发)
                    weight = 3.0
                
                # 负奖励 = 亏损金额 × 因子 × 权重
                # 注意: pnl 是负数，所以结果是负奖励(惩罚)
                reward = pnl * factor * weight
                return float(reward)

        def _get_entry_replay_bonus(self, action: int) -> float:
            """
            入场时根据历史经验计算额外奖励。

            如果过去的类似方向入场大多亏损 → 给负的 bonus (劝退)
            如果过去的类似方向入场大多盈利 → 给小的正 bonus (鼓励)
            """
            if not self.replay_experiences:
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
                # 历史该方向胜率低 → 劝退
                return -0.5
            elif win_rate > 0.65:
                # 历史该方向胜率高 → 微小鼓励
                return 0.2
            else:
                return 0.0

        def _get_exit_replay_bonus(
            self, side: str, pnl: float, trade_duration: int
        ) -> float:
            """
            平仓时根据历史经验计算额外奖励。

            如果当前交易的盈亏模式在过去频繁出现:
            - 亏损模式频繁出现 → 额外惩罚 (强化记忆)
            - 盈利模式频繁出现 → 额外奖励 (巩固好习惯)
            """
            if not self.replay_experiences:
                return 0.0

            if pnl < 0:
                # 亏损: 检查是否有类似的亏损经验
                similar_losses = self._replay_lookup.get(f"{side}_loss", [])
                if len(similar_losses) >= 3:
                    # 历史上这个方向亏了很多次 → 额外惩罚
                    avg_loss = np.mean([
                        abs(e.get("profit_ratio", 0)) for e in similar_losses
                    ])
                    return float(-avg_loss * 50)  # 惩罚放大
                return 0.0

            else:
                # 盈利: 检查是否匹配历史盈利模式
                similar_wins = self._replay_lookup.get(f"{side}_win", [])
                if len(similar_wins) >= 3:
                    avg_win = np.mean([
                        e.get("profit_ratio", 0) for e in similar_wins
                    ])
                    return float(avg_win * 20)  # 额外奖励
                return 0.0
