"""
MTF_BalancedPerformance_DogeAI_NoT3

基于 MTF_BalancedPerformance_DogeAI 的无 T3 版本：
- 移除 T3 指标计算与 T3 入场条件
- 保留原有的 MTF 趋势投票、支撑阻力、BTC 过滤、FreqAI 预测/风控逻辑
"""

from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame
import pandas as pd

from MTF_BalancedPerformance_DogeAI import (
    MTF_BalancedPerformance_DogeAI,
    two_pole_filter,
)


class MTF_BalancedPerformance_DogeAI_NoT3(MTF_BalancedPerformance_DogeAI):
    """
    DogeAI 无 T3 版本。
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get("pair", "")

        sr_lookback = int(round(self._num(self.sr_lookback, 96)))
        sr_lookback = max(5, sr_lookback)
        sr_tolerance = float(self._num(self.sr_tolerance, 0.0030))
        sr_tolerance = min(max(sr_tolerance, 0.0), 0.05)

        dataframe["support_level"] = dataframe["low"].shift(1).rolling(
            window=sr_lookback, min_periods=sr_lookback
        ).min()
        dataframe["resistance_level"] = dataframe["high"].shift(1).rolling(
            window=sr_lookback, min_periods=sr_lookback
        ).max()
        dataframe["near_support"] = (
            dataframe["low"] <= dataframe["support_level"] * (1.0 + sr_tolerance)
        ).fillna(False)
        dataframe["near_resistance"] = (
            dataframe["high"] >= dataframe["resistance_level"] * (1.0 - sr_tolerance)
        ).fillna(False)

        dataframe["trend_up"] = False
        dataframe["trend_down"] = False
        dataframe["btc_trend_up"] = True
        dataframe["btc_trend_down"] = True

        if self.dp:
            pair_tf_cols: list[str] = []

            for tf in self.informative_timeframes:
                informative = self.dp.get_pair_dataframe(pair=pair, timeframe=tf)
                if len(informative) == 0:
                    continue

                close_arr = informative["close"].to_numpy(dtype=float)
                tp_filter = two_pole_filter(close_arr, self.tp_alpha)
                informative["tp_filter"] = tp_filter

                informative["tp_trend_up"] = (
                    informative["tp_filter"] > informative["tp_filter"].shift(self.tp_offset)
                ).fillna(False)

                informative = informative[["date", "tp_trend_up"]].copy()
                informative = informative.rename(columns={"tp_trend_up": f"tp_up_{tf}"})

                dataframe = merge_informative_pair(dataframe, informative, self.timeframe, tf, ffill=True)

                col_name = f"tp_up_{tf}_{tf}"
                if col_name in dataframe.columns:
                    dataframe[col_name] = dataframe[col_name].fillna(False).astype(bool)
                    pair_tf_cols.append(col_name)

            if len(pair_tf_cols) == len(self.informative_timeframes):
                mtf_up = pd.Series(True, index=dataframe.index)
                mtf_down = pd.Series(True, index=dataframe.index)
                for col_name in pair_tf_cols:
                    mtf_up = mtf_up & dataframe[col_name]
                    mtf_down = mtf_down & (~dataframe[col_name])
                dataframe["trend_up"] = mtf_up
                dataframe["trend_down"] = mtf_down
            else:
                dataframe["trend_up"] = False
                dataframe["trend_down"] = False

            btc_filter_on = int(self._num(self.btc_filter_enabled, 1)) == 1
            if btc_filter_on:
                dataframe["btc_trend_up"] = False
                dataframe["btc_trend_down"] = False
                btc_tf_cols: list[str] = []
                for tf in self.informative_timeframes:
                    btc_inf = self.dp.get_pair_dataframe(pair=self.btc_pair, timeframe=tf)
                    if len(btc_inf) == 0:
                        continue

                    btc_close = btc_inf["close"].to_numpy(dtype=float)
                    btc_filter = two_pole_filter(btc_close, self.tp_alpha)
                    btc_series = pd.Series(btc_filter, index=btc_inf.index)
                    btc_inf["btc_tp_up"] = (btc_series > btc_series.shift(self.tp_offset)).fillna(False)
                    btc_inf = btc_inf[["date", "btc_tp_up"]].copy()
                    btc_inf = btc_inf.rename(columns={"btc_tp_up": f"btc_tp_up_{tf}"})
                    dataframe = merge_informative_pair(dataframe, btc_inf, self.timeframe, tf, ffill=True)

                    col_name = f"btc_tp_up_{tf}_{tf}"
                    if col_name in dataframe.columns:
                        dataframe[col_name] = dataframe[col_name].fillna(False).astype(bool)
                        btc_tf_cols.append(col_name)

                if len(btc_tf_cols) == len(self.informative_timeframes):
                    btc_up = pd.Series(True, index=dataframe.index)
                    btc_down = pd.Series(True, index=dataframe.index)
                    for col_name in btc_tf_cols:
                        btc_up = btc_up & dataframe[col_name]
                        btc_down = btc_down & (~dataframe[col_name])
                    dataframe["btc_trend_up"] = btc_up
                    dataframe["btc_trend_down"] = btc_down

        if self.freqai_enabled and hasattr(self, "freqai"):
            dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        btc_filter_on = int(self._num(self.btc_filter_enabled, 1)) == 1

        # 无 T3 版本：仅保留 MTF 趋势 + 支撑阻力 + 成交量 + BTC 趋势过滤
        entry_long = (
            (dataframe["trend_up"])
            & (dataframe["near_support"])
            & (dataframe["volume"] > 0)
            & (dataframe["btc_trend_up"] if btc_filter_on else True)
        )
        dataframe.loc[entry_long, "enter_long"] = 1
        dataframe.loc[entry_long, "enter_tag"] = "mtf_sr_up_not3"

        entry_short = (
            (dataframe["trend_down"])
            & (dataframe["near_resistance"])
            & (dataframe["volume"] > 0)
            & (dataframe["btc_trend_down"] if btc_filter_on else True)
        )
        dataframe.loc[entry_short, "enter_short"] = 1
        dataframe.loc[entry_short, "enter_tag"] = "mtf_sr_down_not3"

        return dataframe
