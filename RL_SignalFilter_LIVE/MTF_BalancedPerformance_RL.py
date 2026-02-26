"""
RSI Adaptive T3 策略 - RL 版本 (方案 B)
============================================================
RL 作为信号过滤器版本

核心设计:
1. 策略负责特征工程和候选信号生成
2. RL 模型决定是否跟随候选信号入场
3. 保留所有现有过滤逻辑 (T3交叉 + AI风险 + BTC过滤)

动作空间:
- 0: Hold (不操作)
- 1: TakeLong (跟随多头信号)
- 2: TakeShort (跟随空头信号)
- 3: ClosePosition (平仓)
"""

import logging
import sys
from pathlib import Path

from freqtrade.strategy import (
    DecimalParameter,
    IntParameter,
    IStrategy,
    CategoricalParameter,
    BooleanParameter,
)
from pandas import DataFrame
import pandas as pd
import numpy as np
import talib.abstract as ta
from functools import reduce

# 导入 ExperienceBuffer
sys.path.insert(0, str(Path(__file__).parent.parent / "freqaimodels"))
try:
    from ExperienceBuffer import ExperienceBuffer
except ImportError:
    ExperienceBuffer = None


logger = logging.getLogger(__name__)


class MTF_BalancedPerformance_RL(IStrategy):
    """
    RL 版本策略 - 方案 B

    策略职责:
    - 特征工程 (feature_engineering_*)
    - 候选信号生成 (populate_entry_trend)
    - BTC 过滤计算

    RL 职责:
    - 决定是否跟随候选信号
    """

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        if ExperienceBuffer:
            self._exp_buffer = ExperienceBuffer(
                storage_dir="user_data/experience_replay",
                max_experiences=2000,
            )
            logger.info("[Strategy] ExperienceBuffer initialized")
        else:
            self._exp_buffer = None
            logger.warning("[Strategy] ExperienceBuffer not available")

    # 策略参数
    minimal_roi = {"0": 0.05, "60": 0.03, "120": 0.02}
    timeframe = "5m"
    can_short = True
    stoploss = -0.18  # 价格变动 6% * 3x 杠杆
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    use_custom_stoploss = True
    process_only_new_candles = True
    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}
    stake_currency = "USDT"
    stake_amount = "unlimited"
    startup_candle_count = 500
    unfilledtimeout = {"entry": 10, "exit": 10, "exit_timeout_count": 0, "unit": "minutes"}
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    exit_profit_offset = 0.0
    disable_dataframe_checks = False
    ignore_buying_expired_candle_after = 0
    position_adjustment_enable = False
    max_entry_position_adjustment = -1

    # 自定义参数
    rsiLen = IntParameter(10, 20, default=14, space="buy")
    minLen = IntParameter(5, 10, default=7, space="buy")
    maxLen = IntParameter(15, 25, default=20, space="buy")
    v = DecimalParameter(0.5, 1.0, default=0.8, space="buy")
    tp_alpha = DecimalParameter(0.05, 0.2, default=0.1, space="buy")
    tp_offset = IntParameter(1, 5, default=2, space="buy")
    btc_filter_enabled = BooleanParameter(default=True, space="buy")
    btc_pair = "BTC/USDT:USDT"
    informative_timeframes = ["1h", "4h"]


    def informative_pairs(self):
        if not self.dp:
            return []
        return [(self.btc_pair, tf) for tf in self.informative_timeframes]

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, **kwargs) -> DataFrame:
        dataframe[f"%-rsi-{period}"] = ta.RSI(dataframe["close"], timeperiod=period)
        dataframe[f"%-mfi-{period}"] = ta.MFI(
            dataframe["high"], dataframe["low"], dataframe["close"], dataframe["volume"], timeperiod=period
        )
        dataframe[f"%-adx-{period}"] = ta.ADX(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=period
        )
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        # FreqAI RL 要求价格数据必须在此方法中定义
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        # RL 模型需要的原始价格数据
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]
        dataframe["%-raw_price"] = dataframe["close"]
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe["&-action"] = 0
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get("pair", "")

        dataframe["rsi"] = ta.RSI(dataframe["close"], timeperiod=self.rsiLen.value)

        rsi_scale = 1.0 - (dataframe["rsi"] / 100.0)
        length = np.rint(self.minLen.value + (self.maxLen.value - self.minLen.value) * rsi_scale).astype(float)
        length = np.clip(length, self.minLen.value, self.maxLen.value)
        dataframe["adap_len"] = length

        src = dataframe["close"].to_numpy(dtype=float)
        L = dataframe["adap_len"].to_numpy(dtype=float)

        e1 = ema_dynamic(src, L)
        e2 = ema_dynamic(e1, L)
        e3 = ema_dynamic(e2, L)
        e4 = ema_dynamic(e3, L)
        e5 = ema_dynamic(e4, L)
        e6 = ema_dynamic(e5, L)

        v = float(self.v.value)
        c1 = -v * v * v
        c2 = 3 * v * v + 3 * v * v * v
        c3 = -6 * v * v - 3 * v - 3 * v * v * v
        c4 = 1 + 3 * v + v * v * v + 3 * v * v

        t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
        dataframe["t3"] = t3
        dataframe["t3_2"] = dataframe["t3"].shift(2)

        dataframe["t3_diff"] = dataframe["close"] - dataframe["t3"]
        dataframe["t3_slope"] = dataframe["t3"] - dataframe["t3"].shift(1)

        dataframe["trend_up"] = True
        dataframe["trend_down"] = True
        dataframe["btc_trend_up"] = True
        dataframe["btc_trend_down"] = True

        if self.dp:
            btc_filter_on = int(self.btc_filter_enabled.value) == 1
            if btc_filter_on:
                dataframe["btc_trend_up"] = False
                dataframe["btc_trend_down"] = False
                btc_votes = pd.Series(0.0, index=dataframe.index)
                btc_vote_count = 0
                for tf in self.informative_timeframes:
                    btc_inf = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe=tf)
                    if len(btc_inf) == 0:
                        continue
                    btc_close = btc_inf["close"].to_numpy(dtype=float)
                    btc_filter = two_pole_filter(btc_close, self.tp_alpha.value)
                    btc_series = pd.Series(btc_filter, index=btc_inf.index)
                    btc_inf["btc_tp_up"] = (btc_series > btc_series.shift(self.tp_offset.value)).fillna(False)
                    btc_inf = btc_inf[["date", "btc_tp_up"]].copy()
                    btc_inf = btc_inf.rename(columns={"btc_tp_up": f"btc_tp_up_{tf}"})
                    from freqtrade.strategy import merge_informative_pair
                    dataframe = merge_informative_pair(
                        dataframe, btc_inf, self.timeframe, tf, ffill=True
                    )
                    col_name = f"btc_tp_up_{tf}_{tf}"
                    if col_name in dataframe.columns:
                        dataframe[col_name] = dataframe[col_name].fillna(False).astype(bool)
                        btc_votes = btc_votes + dataframe[col_name].astype(float)
                        btc_vote_count += 1
                if btc_vote_count > 0:
                    btc_avg = btc_votes / btc_vote_count
                    # 优化：1h 与 4h 分歧(0.5)时不做双向封锁，交给 RL + 本级趋势决定方向
                    dataframe["btc_trend_up"] = btc_avg >= 0.5
                    dataframe["btc_trend_down"] = btc_avg <= 0.5
                else:
                    # 无 BTC 数据时不阻断交易，避免整段时间 0 信号
                    dataframe["btc_trend_up"] = True
                    dataframe["btc_trend_down"] = True

        # === 关键修复：添加 FreqAI 预测调用 ===
        # 在 FreqAI 配置中，模型会自动预测并添加 &-action 列
        # 这里我们确保在需要时调用 FreqAI
        if hasattr(self, 'freqai') and self.freqai:
            dataframe = self.freqai.start(dataframe, metadata, self)
        else:
            # 如果没有 FreqAI，设置默认值
            dataframe["&-action"] = 0
            dataframe["do_predict"] = 0
        
        return dataframe
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_conditions = [
            dataframe["t3_slope"] > 0,
            dataframe["t3_diff"] > 0,
            dataframe["trend_up"],
            dataframe["btc_trend_up"],
        ]
        short_conditions = [
            dataframe["t3_slope"] < 0,
            dataframe["t3_diff"] < 0,
            dataframe["trend_down"],
            dataframe["btc_trend_down"],
        ]

        candidate_long = reduce(lambda x, y: x & y, long_conditions)
        candidate_short = reduce(lambda x, y: x & y, short_conditions)

        if "&-action" in dataframe.columns:
            # 优化：do_predict==0 代表样本不够稳健，但并非无效；
            # 放宽为 >=0，避免模型大量输出被硬过滤导致长期无交易。
            if "do_predict" in dataframe.columns:
                pred_ok = dataframe["do_predict"] >= 0
            else:
                pred_ok = True

            enter_long_cond = candidate_long & pred_ok & (dataframe["&-action"] == 1)
            enter_short_cond = candidate_short & pred_ok & (dataframe["&-action"] == 3)
            dataframe.loc[enter_long_cond, "enter_long"] = 1
            dataframe.loc[enter_long_cond, "enter_tag"] = "rl_long"
            dataframe.loc[enter_short_cond, "enter_short"] = 1
            dataframe.loc[enter_short_cond, "enter_tag"] = "rl_short"
        else:
            dataframe.loc[candidate_long, "enter_long"] = 1
            dataframe.loc[candidate_long, "enter_tag"] = "rl_candidate_long"
            dataframe.loc[candidate_short, "enter_short"] = 1
            dataframe.loc[candidate_short, "enter_tag"] = "rl_candidate_short"

        return dataframe


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

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        if "&-action" in dataframe.columns:
            exit_long_cond = (dataframe["do_predict"] == 1) & (dataframe["&-action"] == 2)
            exit_short_cond = (dataframe["do_predict"] == 1) & (dataframe["&-action"] == 4)
            dataframe.loc[exit_long_cond, "exit_long"] = 1
            dataframe.loc[exit_long_cond, "exit_tag"] = "rl_exit_long"
            dataframe.loc[exit_short_cond, "exit_short"] = 1
            dataframe.loc[exit_short_cond, "exit_tag"] = "rl_exit_short"

        return dataframe

    def _record_trade_experience(self, pair: str, trade, exit_reason: str, current_time) -> None:
        """统一经验记录入口：成交后写入，避免 pre-fill 时 close_rate/profit 为空。"""
        try:
            if not self._exp_buffer:
                return

            dataframe = None
            if self.dp:
                analyzed = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                dataframe = analyzed[0] if isinstance(analyzed, tuple) else analyzed

            self._exp_buffer.record_trade(
                pair=pair,
                trade=trade,
                exit_reason=exit_reason,
                current_time=current_time,
                dataframe=dataframe,
            )
            logger.info(f"[Strategy] Trade experience recorded for {pair}")
        except Exception as e:
            logger.debug(f"[Strategy] Failed to record trade experience: {e}")

    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time, **kwargs) -> bool:
        """
        仅确认出场，不在这里记录经验（此时交易未必成交，字段可能为空）。
        """
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

            if not (is_exit_order and is_trade_closed):
                return

            # 去重：同一 trade 仅记录一次
            try:
                if hasattr(trade, "get_custom_data") and trade.get_custom_data("experience_recorded", False):
                    return
            except Exception:
                pass

            exit_reason = str(getattr(trade, "exit_reason", "") or "filled_exit")
            self._record_trade_experience(pair, trade, exit_reason, current_time)

            try:
                if hasattr(trade, "set_custom_data"):
                    trade.set_custom_data("experience_recorded", True)
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"[Strategy] order_filled experience hook failed: {e}")

# 辅助函数
def ema_dynamic(src, length):
    """动态长度EMA"""
    result = np.zeros_like(src)
    alpha = 2.0 / (length + 1)
    result[0] = src[0]
    for i in range(1, len(src)):
        result[i] = alpha[i] * src[i] + (1 - alpha[i]) * result[i-1]
    return result


def two_pole_filter(src, alpha):
    """双极点滤波器"""
    result = np.zeros_like(src)
    result[0] = src[0]
    result[1] = src[1]
    for i in range(2, len(src)):
        result[i] = (1 - alpha) * (1 - alpha) * src[i] + \
                   2 * (1 - alpha) * alpha * result[i-1] - \
                   alpha * alpha * result[i-2]
    return result
