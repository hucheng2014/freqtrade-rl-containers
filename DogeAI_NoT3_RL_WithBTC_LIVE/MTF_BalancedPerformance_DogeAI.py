"""
MTF_BalancedPerformance_DogeAI

Keeps the original MTF trend-voting entry logic, but replaces the old
MTF drawdown AI model with DogeGod V6.1 Conservative AI model behavior.
"""

from datetime import datetime, timezone
import logging

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy, merge_informative_pair
from pandas import DataFrame
import numpy as np
import pandas as pd
import talib.abstract as ta


logger = logging.getLogger(__name__)


def pine_round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    rounded = np.full_like(values, fill_value=np.nan, dtype=float)
    valid = ~np.isnan(values)
    rounded[valid] = np.sign(values[valid]) * np.floor(np.abs(values[valid]) + 0.5)
    return rounded


def ema_dynamic(src: np.ndarray, length: np.ndarray) -> np.ndarray:
    out = np.full_like(src, fill_value=np.nan, dtype=float)
    if len(src) == 0:
        return out

    for i in range(len(src)):
        prev = out[i - 1] if i > 0 else np.nan
        if np.isnan(prev):
            out[i] = src[i]
            continue

        alpha = 2.0 / (float(length[i]) + 1.0)
        out[i] = alpha * src[i] + (1.0 - alpha) * prev
    return out


def two_pole_filter(src: np.ndarray, alpha: float) -> np.ndarray:
    out = np.full_like(src, fill_value=np.nan, dtype=float)
    if len(src) == 0:
        return out

    a2 = alpha * alpha
    one_minus = 1.0 - alpha
    om2 = one_minus * one_minus

    out[0] = src[0]
    if len(src) > 1:
        out[1] = src[1]

    for i in range(2, len(src)):
        if np.isnan(src[i]) or np.isnan(out[i - 1]) or np.isnan(out[i - 2]):
            out[i] = out[i - 1] if not np.isnan(out[i - 1]) else src[i]
            continue
        out[i] = a2 * src[i] + 2 * one_minus * out[i - 1] - om2 * out[i - 2]

    return out


class MTF_BalancedPerformance_DogeAI(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = True
    freqai_enabled = True

    informative_timeframes = ["1h", "2h", "4h"]

    minimal_roi = {
        "0": 0.05,
        "60": 0.03,
        "120": 0.02,
    }

    # Doge Hyperopt profile base settings.
    stoploss = -0.06
    trailing_stop = False
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.07
    trailing_only_offset_is_reached = True

    buy_stoploss = DecimalParameter(
        -0.08, -0.03, default=-0.06, decimals=3, space="sell", load=True, optimize=True
    )
    buy_trailing_stop_positive = DecimalParameter(
        0.015, 0.06, default=0.03, decimals=3, space="sell", load=True, optimize=True
    )
    buy_trailing_stop_positive_offset = DecimalParameter(
        0.04, 0.12, default=0.07, decimals=3, space="sell", load=True, optimize=True
    )

    use_exit_signal = False
    exit_profit_only = False
    startup_candle_count = 500

    rsiLen = 14
    minLen = 5
    maxLen = 50
    v = 0.7

    tp_alpha = 0.10
    tp_offset = 3
    sr_lookback = IntParameter(24, 192, default=96, space="buy", load=True, optimize=False)
    sr_tolerance = DecimalParameter(0.0010, 0.0100, default=0.0030, decimals=4, space="buy", load=True, optimize=False)

    btc_pair = "BTC/USDT:USDT"
    btc_filter_enabled = IntParameter(0, 1, default=1, space="buy")

    # Doge AI thresholds.
    ai_veto_threshold = 0.008
    ai_veto_threshold_short = 0.010
    ai_early_exit_threshold = -0.008
    ai_early_exit_threshold_short = 0.010

    _btc_prediction_cache: tuple = (0, 0)
    _btc_prediction_cache_time = None
    _btc_prediction_error_cooldown = None
    _btc_prediction_cache_duration = 900

    @staticmethod
    def _num(value, default):
        try:
            return float(value.value)
        except Exception:
            try:
                return float(value)
            except Exception:
                return float(default)

    def leverage(
        self,
        pair: str,
        current_time: "datetime",
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> float:
        return 3.0

    def bot_start(self, **kwargs) -> None:
        try:
            super().bot_start(**kwargs)
        except Exception:
            pass

        if hasattr(self.buy_stoploss, "value"):
            self.stoploss = float(self.buy_stoploss.value)
        if hasattr(self.buy_trailing_stop_positive, "value"):
            self.trailing_stop_positive = float(self.buy_trailing_stop_positive.value)
        if hasattr(self.buy_trailing_stop_positive_offset, "value"):
            self.trailing_stop_positive_offset = float(self.buy_trailing_stop_positive_offset.value)

        if self.trailing_stop_positive_offset <= self.trailing_stop_positive:
            self.trailing_stop_positive_offset = self.trailing_stop_positive + 0.001

    def informative_pairs(self):
        if not self.dp:
            return []
        pairs = self.dp.current_whitelist()
        base = [(pair, tf) for pair in pairs for tf in self.informative_timeframes]
        btc = [(self.btc_pair, tf) for tf in self.informative_timeframes]
        return list(dict.fromkeys(base + btc))

    # Doge AI feature model.
    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, **kwargs) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe["%-pct_change"] = dataframe["close"].pct_change(1)
        dataframe["%-volatility"] = dataframe["close"].rolling(window=10).std() / (dataframe["close"] + 1e-8)
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe["%-ema_fast"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["%-ema_slow"] = ta.EMA(dataframe, timeperiod=55)
        dataframe["%-rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["%-atr_percent"] = ta.ATR(dataframe, timeperiod=14) / dataframe["close"]
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dataframe["&-s_close"] = dataframe["close"].shift(-6) / dataframe["close"] - 1
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get("pair", "")

        dataframe["rsi"] = ta.RSI(dataframe["close"], timeperiod=self.rsiLen)

        rsi_scale = 1.0 - (dataframe["rsi"] / 100.0)
        length_raw = (self.minLen + (self.maxLen - self.minLen) * rsi_scale).to_numpy(dtype=float)
        length = pine_round_half_away_from_zero(length_raw)
        dataframe["adap_len"] = length

        src = dataframe["close"].to_numpy(dtype=float)
        L = dataframe["adap_len"].to_numpy(dtype=float)

        e1 = ema_dynamic(src, L)
        e2 = ema_dynamic(e1, L)
        e3 = ema_dynamic(e2, L)
        e4 = ema_dynamic(e3, L)
        e5 = ema_dynamic(e4, L)
        e6 = ema_dynamic(e5, L)

        v = float(self.v)
        c1 = -v * v * v
        c2 = 3 * v * v + 3 * v * v * v
        c3 = -6 * v * v - 3 * v - 3 * v * v * v
        c4 = 1 + 3 * v + v * v * v + 3 * v * v

        t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
        dataframe["t3"] = t3
        dataframe["t3_2"] = dataframe["t3"].shift(2)
        dataframe["t3_up"] = (dataframe["t3"] > dataframe["t3_2"]).fillna(False)
        dataframe["t3_down"] = (dataframe["t3"] < dataframe["t3_2"]).fillna(False)

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

    def _get_btc_prediction(self) -> tuple:
        current_time = datetime.now(timezone.utc)

        if (
            self._btc_prediction_cache_time is not None
            and (current_time - self._btc_prediction_cache_time).total_seconds() < self._btc_prediction_cache_duration
        ):
            return self._btc_prediction_cache

        try:
            if not self.dp:
                return (0, 0)
            btc_dataframe, _ = self.dp.get_analyzed_dataframe(self.btc_pair, self.timeframe)
            if len(btc_dataframe) > 0:
                last_candle = btc_dataframe.iloc[-1]
                btc_prediction = last_candle.get("&-s_close", 0)
                btc_do_predict = last_candle.get("do_predict", 0)
                self._btc_prediction_cache = (btc_prediction, btc_do_predict)
                self._btc_prediction_cache_time = current_time
                return self._btc_prediction_cache
        except Exception as exc:
            if (
                self._btc_prediction_error_cooldown is None
                or (current_time - self._btc_prediction_error_cooldown).total_seconds() > 900
            ):
                logger.warning("MTF_DogeAI BTC prediction fetch failed: %s", exc)
                self._btc_prediction_error_cooldown = current_time

        return (0, 0)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        btc_filter_on = int(self._num(self.btc_filter_enabled, 1)) == 1

        entry_long = (
            (dataframe["t3_up"])
            & (dataframe["close"] > dataframe["t3"])
            & (dataframe["trend_up"])
            & (dataframe["near_support"])
            & (dataframe["volume"] > 0)
            & (dataframe["btc_trend_up"] if btc_filter_on else True)
        )
        dataframe.loc[entry_long, "enter_long"] = 1
        dataframe.loc[entry_long, "enter_tag"] = "mtf_sr_t3_up"

        entry_short = (
            (dataframe["t3_down"])
            & (dataframe["close"] < dataframe["t3"])
            & (dataframe["trend_down"])
            & (dataframe["near_resistance"])
            & (dataframe["volume"] > 0)
            & (dataframe["btc_trend_down"] if btc_filter_on else True)
        )
        dataframe.loc[entry_short, "enter_short"] = 1
        dataframe.loc[entry_short, "enter_tag"] = "mtf_sr_t3_down"

        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        if not self.freqai_enabled or not self.dp:
            return True

        pair_prediction = 0.0
        pair_do_predict = 0
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) > 0:
                last_candle = dataframe.iloc[-1]
                pair_prediction = float(last_candle.get("&-s_close", 0) or 0)
                pair_do_predict = int(last_candle.get("do_predict", 0) or 0)
        except Exception:
            return True

        btc_prediction, btc_do_predict = self._get_btc_prediction()
        btc_prediction = float(btc_prediction or 0)
        btc_do_predict = int(btc_do_predict or 0)

        if side == "long":
            pair_veto = pair_do_predict == 1 and pair_prediction < -self.ai_veto_threshold
            btc_veto = btc_do_predict == 1 and btc_prediction < -self.ai_veto_threshold
        else:
            pair_veto = pair_do_predict == 1 and pair_prediction > self.ai_veto_threshold_short
            btc_veto = btc_do_predict == 1 and btc_prediction > self.ai_veto_threshold_short

        if pair_veto or btc_veto:
            logger.info(
                "MTF_DogeAI AI veto: pair=%s side=%s pair_pred=%.4f btc_pred=%.4f",
                pair,
                side,
                pair_prediction,
                btc_prediction,
            )
            return False

        return True

    def _get_dynamic_ai_threshold(self, current_profit: float, pair: str) -> tuple:
        if current_profit < -0.05:
            threshold = -0.002
            threshold_short = 0.002
            logger.info("MTF_DogeAI emergency AI threshold: %s profit=%.2f%%", pair, current_profit * 100)
        elif current_profit < -0.03:
            threshold = -0.004
            threshold_short = 0.004
        elif current_profit < -0.02:
            threshold = -0.006
            threshold_short = 0.006
        else:
            threshold = self.ai_early_exit_threshold
            threshold_short = self.ai_early_exit_threshold_short
        return threshold, threshold_short

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool:
        if current_profit >= 0:
            return False

        if current_profit < self.stoploss:
            return "stoploss_6pct"

        if not self.freqai_enabled or not self.dp:
            return False

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return False

        last_candle = dataframe.iloc[-1]
        ai_prediction = last_candle.get("&-s_close", None)
        do_predict = int(last_candle.get("do_predict", 0) or 0)

        btc_prediction, btc_do_predict = self._get_btc_prediction()
        btc_prediction = float(btc_prediction or 0)
        btc_do_predict = int(btc_do_predict or 0)

        threshold, threshold_short = self._get_dynamic_ai_threshold(current_profit, pair)

        if not trade.is_short:
            pair_trigger = do_predict == 1 and ai_prediction is not None and float(ai_prediction) < threshold
            btc_trigger = btc_do_predict == 1 and btc_prediction < threshold
            if pair_trigger or btc_trigger:
                return "ai_loss_reduction_long"
        else:
            pair_trigger = do_predict == 1 and ai_prediction is not None and float(ai_prediction) > threshold_short
            btc_trigger = btc_do_predict == 1 and btc_prediction > threshold_short
            if pair_trigger or btc_trigger:
                return "ai_loss_reduction_short"

        return False

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe
