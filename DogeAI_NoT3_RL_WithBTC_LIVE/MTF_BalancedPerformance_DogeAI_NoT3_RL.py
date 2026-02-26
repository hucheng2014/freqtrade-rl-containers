"""
import logging
logger = logging.getLogger(__name__)
MTF_BalancedPerformance_DogeAI_NoT3_RL

方案 B（RL 过滤）版本：
- 保留 NoT3 策略的候选信号逻辑（MTF 趋势 + 支撑阻力 + BTC 过滤）
- 通过 FreqAI RL 动作列 `&-action` 决定是否执行入场/出场
"""

from typing import Optional
from pandas import DataFrame
import pandas as pd

from MTF_BalancedPerformance_DogeAI_NoT3 import MTF_BalancedPerformance_DogeAI_NoT3


class MTF_BalancedPerformance_DogeAI_NoT3_RL(MTF_BalancedPerformance_DogeAI_NoT3):
    """
    DogeAI NoT3 + RL 过滤版本。
    """

    # 覆盖父类止损：价格变动6% x 3x杠杆 = 18%
    stoploss = -0.18
    use_custom_stoploss = True

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

        if "&-action" in dataframe.columns:
            predict_mask = self._predict_mask(dataframe)
            action = dataframe["&-action"].fillna(0).astype(float).astype(int)

            exit_long = predict_mask & (action == 2)
            exit_short = predict_mask & (action == 4)

            dataframe.loc[exit_long, "exit_long"] = 1
            dataframe.loc[exit_long, "exit_tag"] = "rl_exit_long"

            dataframe.loc[exit_short, "exit_short"] = 1
            dataframe.loc[exit_short, "exit_tag"] = "rl_exit_short"

        return dataframe

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        """
        RL 版本不使用父类的 AI 回归预测减亏（&-s_close 不存在）。
        出场完全由 RL 出场信号 + trailing stop + 硬止损 + ROI 控制。
        """
        return False

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
        动态止损（防 -2021: Order would immediately trigger）：
        - 未达到盈利触发阈值时，返回 -1，让硬止损（-0.18）兜底
        - 达到阈值后，按 open_profit 计算 trailing，并转换为当前价相对止损距离
        - 对交易所止损单增加最小安全距离，避免"立即触发"被拒单
        """
        # 兼容 Parameter / float 两种配置形态
        if hasattr(self, "_num"):
            trailing_positive = float(self._num(getattr(self, "trailing_stop_positive_opt", getattr(self, "trailing_stop_positive", 0.015)), 0.015))
            trailing_offset = float(self._num(getattr(self, "trailing_stop_positive_offset_opt", getattr(self, "trailing_stop_positive_offset", 0.025)), 0.025))
        else:
            tp = getattr(self, "trailing_stop_positive", 0.015)
            to = getattr(self, "trailing_stop_positive_offset", 0.025)
            trailing_positive = float(tp.value) if hasattr(tp, "value") else float(tp)
            trailing_offset = float(to.value) if hasattr(to, "value") else float(to)

        # 避免 offset 与 positive 过近（例如都等于 0.017）导致止损贴价
        min_profit_gap = 0.005
        if trailing_offset <= trailing_positive + min_profit_gap:
            trailing_offset = trailing_positive + min_profit_gap

        if current_profit >= trailing_offset:
            from freqtrade.strategy import stoploss_from_open

            target_open_profit = current_profit - trailing_positive
            if target_open_profit <= 0:
                return -1

            sl_rel = stoploss_from_open(
                open_relative_stop=target_open_profit,
                current_profit=current_profit,
                is_short=getattr(trade, "is_short", False),
                leverage=float(getattr(trade, "leverage", 1.0) or 1.0),
            )

            # 对当前价保持至少 0.6% 安全距离，降低交易所拒单概率
            min_current_distance = 0.006
            if sl_rel < min_current_distance:
                sl_rel = min_current_distance
            return sl_rel

        return -1

    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time, **kwargs) -> bool:
        """
        交易退出时记录经验到 ExperienceBuffer，用于 RL 经验回放学习。
        """
        try:
            # 获取 FreqAI 模型
            if hasattr(self, "freqai") and self.freqai and hasattr(self.freqai, "model"):
                model = self.freqai.model
                # 检查模型是否有经验缓冲区
                if hasattr(model, "exp_buffer") and model.exp_buffer:
                    # 获取当前分析后的 dataframe
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    # 记录交易经验
                    model.exp_buffer.record_trade(
                        pair=pair,
                        trade=trade,
                        exit_reason=exit_reason,
                        current_time=current_time,
                        dataframe=dataframe
                    )
        except Exception as e:
            # 记录错误但不影响交易执行
            import logging
            logging.getLogger(__name__).debug(f"[DogeAI_RL] Failed to record trade experience: {e}")
        
        return True
