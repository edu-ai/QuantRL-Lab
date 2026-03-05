import numpy as np
import pandas as pd

from quantrl_lab.data.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register(
    name="SMA",
    required_columns={"close"},
    output_columns=["SMA"],
    description="Simple Moving Average - smooths price data by averaging over a rolling window",
)
def sma(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.DataFrame:
    """
    Add Simple Moving Average to dataframe.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Window size. Defaults to 20.
        column (str, optional): Column used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: Dataframe with SMA column added.
    """
    result = df.copy()

    # Handle multiple symbols — "Symbol" column is added by the YFinance loader
    if "Symbol" in result.columns:
        result[f"SMA_{window}"] = result.groupby("Symbol")[column].transform(lambda x: x.rolling(window=window).mean())
    else:
        result[f"SMA_{window}"] = result[column].rolling(window=window).mean()

    return result


@IndicatorRegistry.register(
    name="EMA",
    required_columns={"close"},
    output_columns=["EMA"],
    description="Exponential Moving Average - gives more weight to recent prices",
)
def ema(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.DataFrame:
    """
    Add Exponential Moving Average to dataframe.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Window size. Defaults to 20.
        column (str, optional): Column used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: Dataframe with EMA column added.
    """
    result = df.copy()

    if "Symbol" in result.columns:
        result[f"EMA_{window}"] = result.groupby("Symbol")[column].transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )
    else:
        result[f"EMA_{window}"] = result[column].ewm(span=window, adjust=False).mean()

    return result


@IndicatorRegistry.register(
    name="RSI",
    required_columns={"close"},
    output_columns=["RSI"],
    description="Relative Strength Index - momentum oscillator measuring speed and magnitude of price changes (0-100)",
)
def rsi(df: pd.DataFrame, window: int = 14, column: str = "Close") -> pd.DataFrame:
    """
    Calculate Relative Strength Index using Wilder's smoothing.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Window size. Defaults to 14.
        column (str, optional): Column used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: Dataframe with RSI column added.
    """
    result = df.copy()

    def _calculate_rsi(prices):
        prices = prices.astype(float)
        deltas = np.zeros_like(prices)
        deltas[1:] = np.diff(prices)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gains = np.full_like(prices, np.nan, dtype=float)
        avg_losses = np.full_like(prices, np.nan, dtype=float)
        rsi_values = np.full_like(prices, np.nan, dtype=float)

        if len(prices) > window:
            avg_gains[window] = np.mean(gains[1 : window + 1])  # noqa: E203
            avg_losses[window] = np.mean(losses[1 : window + 1])  # noqa: E203

            if avg_losses[window] != 0:
                rs = avg_gains[window] / avg_losses[window]
                rsi_values[window] = 100 - (100 / (1 + rs))
            else:
                rsi_values[window] = 100

            # Wilder's smoothing uses (n-1) multiplier for subsequent values
            for i in range(window + 1, len(prices)):
                avg_gains[i] = (avg_gains[i - 1] * (window - 1) + gains[i]) / window
                avg_losses[i] = (avg_losses[i - 1] * (window - 1) + losses[i]) / window

                if avg_losses[i] != 0:
                    rs = avg_gains[i] / avg_losses[i]
                    rsi_values[i] = 100 - (100 / (1 + rs))
                else:
                    rsi_values[i] = 100
        return rsi_values

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            result.loc[group.index, f"RSI_{window}"] = _calculate_rsi(group[column].values)
    else:
        result[f"RSI_{window}"] = _calculate_rsi(result[column].values)

    return result


@IndicatorRegistry.register(
    name="MACD",
    required_columns={"close"},
    output_columns=["MACD_line", "MACD_signal"],
    description="Moving Average Convergence Divergence - trend-following momentum indicator",
)
def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "Close",
) -> pd.DataFrame:
    """
    Calculate Moving Average Convergence Divergence (MACD) crossover
    indicator.

    This implementation focuses on the crossover strategy using MACD line and signal line.
    Trading signals are generated when MACD line crosses above/below the signal line.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        fast (int, optional): Short term EMA period. Defaults to 12.
        slow (int, optional): Long term EMA period. Defaults to 26.
        signal (int, optional): EMA of MACD line period. Defaults to 9.
        column (str, optional): Column used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: Dataframe with MACD line and signal line added.
    """
    result = df.copy()

    if "Symbol" in result.columns:
        for _, group in result.groupby("Symbol"):
            fast_ema = group[column].ewm(span=fast, adjust=False).mean()
            slow_ema = group[column].ewm(span=slow, adjust=False).mean()
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()

            result.loc[group.index, f"MACD_line_{fast}_{slow}"] = macd_line
            result.loc[group.index, f"MACD_signal_{signal}"] = signal_line
    else:
        fast_ema = result[column].ewm(span=fast, adjust=False).mean()
        slow_ema = result[column].ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        result[f"MACD_line_{fast}_{slow}"] = macd_line
        result[f"MACD_signal_{signal}"] = signal_line

    return result


@IndicatorRegistry.register(
    name="ATR",
    required_columns={"high", "low", "close"},
    output_columns=["ATR"],
    description="Average True Range - measures market volatility by decomposing the entire range of prices",
)
def atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR) indicator.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Window size. Defaults to 14.

    Returns:
        pd.DataFrame: Dataframe with ATR column added.
    """
    result = df.copy()

    def _calculate_atr(high, low, close):
        high_low = high - low
        high_close_prev = np.abs(high - np.append(np.nan, close[:-1]))
        low_close_prev = np.abs(low - np.append(np.nan, close[:-1]))

        tr = np.maximum(high_low, high_close_prev)
        tr = np.maximum(tr, low_close_prev)

        atr_values = np.full_like(close, np.nan, dtype=float)

        # First ATR value is the simple average of the first n periods
        if len(close) > window:
            atr_values[window - 1] = np.nanmean(tr[:window])

            # Subsequent values use Wilder's smoothing
            for i in range(window, len(close)):
                atr_values[i] = (atr_values[i - 1] * (window - 1) + tr[i]) / window

        return atr_values

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            result.loc[group.index, f"ATR_{window}"] = _calculate_atr(
                group["High"].values, group["Low"].values, group["Close"].values
            )
    else:
        result[f"ATR_{window}"] = _calculate_atr(result["High"].values, result["Low"].values, result["Close"].values)

    return result


@IndicatorRegistry.register(
    name="BB",
    required_columns={"close"},
    output_columns=["BB_middle", "BB_upper", "BB_lower", "BB_bandwidth"],
    description="Bollinger Bands - volatility bands placed above and below a moving average",
)
def bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0, column: str = "Close") -> pd.DataFrame:
    """
    Calculate Bollinger Bands indicator.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Window size for moving average. Defaults to 20.
        num_std (float, optional): Number of standard deviations. Defaults to 2.0.
        column (str, optional): Column used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: Dataframe with Bollinger Bands columns added.
    """
    result = df.copy()

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            middle_band = group[column].rolling(window=window).mean()
            std = group[column].rolling(window=window).std()
            upper_band = middle_band + (std * num_std)
            lower_band = middle_band - (std * num_std)
            bandwidth = (upper_band - lower_band) / middle_band

            result.loc[group.index, f"BB_middle_{window}"] = middle_band
            result.loc[group.index, f"BB_upper_{window}_{num_std}"] = upper_band
            result.loc[group.index, f"BB_lower_{window}_{num_std}"] = lower_band
            result.loc[group.index, f"BB_bandwidth_{window}"] = bandwidth
    else:
        middle_band = result[column].rolling(window=window).mean()
        std = result[column].rolling(window=window).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        bandwidth = (upper_band - lower_band) / middle_band

        result[f"BB_middle_{window}"] = middle_band
        result[f"BB_upper_{window}_{num_std}"] = upper_band
        result[f"BB_lower_{window}_{num_std}"] = lower_band
        result[f"BB_bandwidth_{window}"] = bandwidth

    return result


@IndicatorRegistry.register(
    name="STOCH",
    required_columns={"high", "low", "close"},
    output_columns=["STOCH_%K", "STOCH_%D"],
    description="Stochastic Oscillator - momentum indicator comparing closing price to price range over time (0-100)",
)
def stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3, smooth_k: int = 1) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        k_window (int, optional): Window for %K calculation. Defaults to 14.
        d_window (int, optional): Window for %D calculation. Defaults to 3.
        smooth_k (int, optional): Smoothing period for %K. Defaults to 1.

    Returns:
        pd.DataFrame: Dataframe with Stochastic Oscillator columns added.
    """
    result = df.copy()

    def _calculate_stochastic(high, low, close):
        lowest_low = pd.Series(low).rolling(window=k_window).min()
        highest_high = pd.Series(high).rolling(window=k_window).max()

        k_fast = 100 * ((pd.Series(close) - lowest_low) / (highest_high - lowest_low))

        if smooth_k > 1:
            k = k_fast.rolling(window=smooth_k).mean()
        else:
            k = k_fast

        d = k.rolling(window=d_window).mean()

        return k.values, d.values

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            k_values, d_values = _calculate_stochastic(group["High"].values, group["Low"].values, group["Close"].values)
            result.loc[group.index, f"STOCH_%K_{k_window}_{smooth_k}"] = k_values
            result.loc[group.index, f"STOCH_%D_{d_window}"] = d_values
    else:
        k_values, d_values = _calculate_stochastic(result["High"].values, result["Low"].values, result["Close"].values)
        result[f"STOCH_%K_{k_window}_{smooth_k}"] = k_values
        result[f"STOCH_%D_{d_window}"] = d_values

    return result


@IndicatorRegistry.register(
    name="OBV",
    required_columns={"close", "volume"},
    output_columns=["OBV"],
    description="On-Balance Volume - cumulative volume indicator showing buying/selling pressure",
)
def on_balance_volume(df: pd.DataFrame, close_col: str = "Close", volume_col: str = "Volume") -> pd.DataFrame:
    """
    Calculate On-Balance Volume (OBV) indicator.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        close_col (str, optional): Column name for close prices. Defaults to "Close".
        volume_col (str, optional): Column name for volume. Defaults to "Volume".

    Returns:
        pd.DataFrame: Dataframe with OBV column added.
    """
    result = df.copy()

    def _calculate_obv(close, volume):
        close_diff = np.diff(close, prepend=close[0])
        obv = np.zeros_like(close)
        obv[0] = volume[0]

        for i in range(1, len(close)):
            if close_diff[i] > 0:
                obv[i] = obv[i - 1] + volume[i]
            elif close_diff[i] < 0:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        return obv

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            result.loc[group.index, "OBV"] = _calculate_obv(group[close_col].values, group[volume_col].values)
    else:
        result["OBV"] = _calculate_obv(result[close_col].values, result[volume_col].values)

    return result


@IndicatorRegistry.register(
    name="WILLR",
    required_columns={"high", "low", "close"},
    output_columns=["WILLR"],
    description="Williams %R - momentum indicator measuring overbought/oversold levels (-100 to 0)",
)
def williams_r(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Williams %R indicator.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Lookback period. Defaults to 14.

    Returns:
        pd.DataFrame: Dataframe with Williams %R column added.
    """
    result = df.copy()

    def _calculate_willr(high, low, close):
        highest_high = pd.Series(high).rolling(window=window).max()
        lowest_low = pd.Series(low).rolling(window=window).min()

        willr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return willr.values

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            result.loc[group.index, f"WILLR_{window}"] = _calculate_willr(
                group["High"].values, group["Low"].values, group["Close"].values
            )
    else:
        result[f"WILLR_{window}"] = _calculate_willr(
            result["High"].values, result["Low"].values, result["Close"].values
        )

    return result


@IndicatorRegistry.register(
    name="CCI",
    required_columns={"high", "low", "close"},
    output_columns=["CCI"],
    description="Commodity Channel Index - measures deviation from average price to identify cyclical trends",
)
def cci(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate Commodity Channel Index (CCI).

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Lookback period. Defaults to 20.

    Returns:
        pd.DataFrame: Dataframe with CCI column added.
    """
    result = df.copy()

    def _calculate_cci(high, low, close):
        tp = (high + low + close) / 3
        tp_series = pd.Series(tp)

        sma_tp = tp_series.rolling(window=window).mean()

        # Rolling MAD via apply is slower but correct; pandas has no vectorized rolling MAD centered on rolling mean
        def mean_deviation(x):
            return np.mean(np.abs(x - np.mean(x)))

        mad = tp_series.rolling(window=window).apply(mean_deviation, raw=True)

        # 0.015 is the Lambert constant used in the CCI formula
        cci_val = (tp_series - sma_tp) / (0.015 * mad)

        return cci_val.values

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            result.loc[group.index, f"CCI_{window}"] = _calculate_cci(
                group["High"].values, group["Low"].values, group["Close"].values
            )
    else:
        result[f"CCI_{window}"] = _calculate_cci(result["High"].values, result["Low"].values, result["Close"].values)

    return result


@IndicatorRegistry.register(
    name="MFI",
    required_columns={"high", "low", "close", "volume"},
    output_columns=["MFI"],
    description="Money Flow Index - volume-weighted RSI measuring buying and selling pressure (0-100)",
)
def mfi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Money Flow Index (MFI).

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Lookback period. Defaults to 14.

    Returns:
        pd.DataFrame: Dataframe with MFI column added.
    """
    result = df.copy()

    def _calculate_mfi(high, low, close, volume):
        tp = (high + low + close) / 3
        rmf = tp * volume

        tp_diff = np.diff(tp, prepend=tp[0])

        pos_flow = np.where(tp_diff > 0, rmf, 0)
        neg_flow = np.where(tp_diff < 0, rmf, 0)

        pos_mf_sum = pd.Series(pos_flow).rolling(window=window).sum()
        neg_mf_sum = pd.Series(neg_flow).rolling(window=window).sum()

        money_ratio = pos_mf_sum / neg_mf_sum
        mfi_calc = 100 - (100 / (1 + money_ratio))

        # When all flow is positive (neg_mf_sum == 0), MFI is 100; when both are 0 (no volume), use 50
        mfi_calc = np.where(neg_mf_sum == 0, 100, mfi_calc)
        mfi_calc = np.where((neg_mf_sum == 0) & (pos_mf_sum == 0), 50, mfi_calc)

        return mfi_calc

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            result.loc[group.index, f"MFI_{window}"] = _calculate_mfi(
                group["High"].values, group["Low"].values, group["Close"].values, group["Volume"].values
            )
    else:
        result[f"MFI_{window}"] = _calculate_mfi(
            result["High"].values, result["Low"].values, result["Close"].values, result["Volume"].values
        )

    return result


@IndicatorRegistry.register(
    name="ADX",
    required_columns={"high", "low", "close"},
    output_columns=["ADX", "ADX_pos", "ADX_neg"],
    description="Average Directional Index - measures trend strength and direction (0-100)",
)
def adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX).

    Includes +DI and -DI.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Lookback period. Defaults to 14.

    Returns:
        pd.DataFrame: Dataframe with ADX, +DI, and -DI columns added.
    """
    result = df.copy()

    def _wilder_smooth(data, window):
        """Wilder's Smoothing (RMA): first value is SMA, subsequent are
        (prev * (n-1) + curr) / n."""
        smoothed = np.full_like(data, np.nan, dtype=float)
        if len(data) > window:
            # Use nanmean to handle initial NaNs from TR calculation
            smoothed[window - 1] = np.nanmean(data[:window])
            for i in range(window, len(data)):
                if np.isnan(smoothed[i - 1]):
                    smoothed[i] = data[i]
                else:
                    smoothed[i] = (smoothed[i - 1] * (window - 1) + data[i]) / window
        return smoothed

    def _calculate_adx(high, low, close):
        high_low = high - low
        high_close_prev = np.abs(high - np.append(np.nan, close[:-1]))
        low_close_prev = np.abs(low - np.append(np.nan, close[:-1]))
        tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))

        up_move = high - np.append(np.nan, high[:-1])
        down_move = np.append(np.nan, low[:-1]) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_smooth = _wilder_smooth(tr, window)
        plus_dm_smooth = _wilder_smooth(plus_dm, window)
        minus_dm_smooth = _wilder_smooth(minus_dm, window)

        with np.errstate(divide="ignore", invalid="ignore"):
            plus_di = 100 * (plus_dm_smooth / tr_smooth)
            minus_di = 100 * (minus_dm_smooth / tr_smooth)

        sum_di = plus_di + minus_di
        diff_di = np.abs(plus_di - minus_di)

        with np.errstate(divide="ignore", invalid="ignore"):
            dx = 100 * (diff_di / sum_di)

        adx_val = _wilder_smooth(dx, window)

        return adx_val, plus_di, minus_di

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            adx_val, p_di, m_di = _calculate_adx(group["High"].values, group["Low"].values, group["Close"].values)
            result.loc[group.index, f"ADX_{window}"] = adx_val
            result.loc[group.index, f"ADX_pos_{window}"] = p_di
            result.loc[group.index, f"ADX_neg_{window}"] = m_di
    else:
        adx_val, p_di, m_di = _calculate_adx(result["High"].values, result["Low"].values, result["Close"].values)
        result[f"ADX_{window}"] = adx_val
        result[f"ADX_pos_{window}"] = p_di
        result[f"ADX_neg_{window}"] = m_di

    return result
