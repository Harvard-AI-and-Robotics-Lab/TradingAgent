# Trading Strategy Baselines for Beginners

This document introduces the baseline trading strategies implemented in this project. These strategies serve as benchmarks for evaluating more advanced trading algorithms and provide a foundation for understanding quantitative trading concepts.

## 📊 Overview

All strategies are implemented using the [backtesting](https://kernc.github.io/backtesting.py/) package and tested on **AAPL stock** from **2024-01-01 to 2024-03-29**. Each strategy follows a systematic approach to generate buy/sell signals based on technical analysis indicators.

---

## 🛡️ 1. Buy & Hold Strategy

**File**: `buy_hold.py`

### What it does:
- Buys the stock at the beginning and holds it until the end
- No active trading - pure passive investment approach

### When to use:
- As a benchmark to compare against active strategies
- When you believe in long-term market growth
- Low maintenance investment approach

### Key Characteristics:
- **Zero trades** during the period
- **No transaction costs** after initial purchase
- **Market risk exposure** - follows market movements exactly

### Results (AAPL 2024):
- **Return**: -5.90%
- **Annualized Return**: -22.21%
- **Max Drawdown**: -13.25%

---

## 📈 2. SMA (Simple Moving Average) Strategy

**File**: `sma_strategy.py`

### What it does:
- **Golden Cross**: Buy when short-term SMA (10-day) crosses above long-term SMA (30-day)
- **Death Cross**: Sell when short-term SMA crosses below long-term SMA

### When to use:
- In trending markets
- When you want to follow momentum
- For capturing major trend changes

### Key Characteristics:
- **Trend-following** approach
- **Lag indicator** - reacts to price changes, doesn't predict them
- **Few signals** - waits for clear trend reversals

### Parameters:
- Short SMA: 10 days
- Long SMA: 30 days

### Results (AAPL 2024):
- **Return**: 0.00% (No trades executed)
- **Annualized Return**: 0.00%
- **Max Drawdown**: 0.00%

---

## ⚡ 3. MACD Strategy

**File**: `macd_strategy.py`

### What it does:
- **Buy Signal**: When MACD line crosses above the signal line
- **Sell Signal**: When MACD line crosses below the signal line

### How MACD works:
- **MACD Line** = 12-day EMA - 26-day EMA
- **Signal Line** = 9-day EMA of MACD line
- **Crossovers** indicate momentum changes

### When to use:
- For momentum-based trading
- In trending markets
- When looking for entry/exit timing

### Parameters:
- Fast EMA: 12 days
- Slow EMA: 26 days
- Signal EMA: 9 days

### Results (AAPL 2024):
- **Return**: -1.82%
- **Annualized Return**: -7.29%
- **Max Drawdown**: -4.96%

---

## 🎯 4. RSI Strategy

**File**: `rsi_strategy.py`

### What it does:
- **Buy Signal**: When RSI < 30 (oversold) and starts rising
- **Sell Signal**: When RSI > 70 (overbought) and starts falling

### How RSI works:
- **RSI** measures the speed and change of price movements
- **Range**: 0 to 100
- **Oversold**: RSI < 30 (potential buying opportunity)
- **Overbought**: RSI > 70 (potential selling opportunity)

### When to use:
- In ranging/sideways markets
- For contrarian trading (buying dips, selling peaks)
- When looking for reversal signals

### Parameters:
- RSI Period: 14 days
- Oversold Threshold: 30
- Overbought Threshold: 70

### Results (AAPL 2024):
- **Return**: +0.22% ⭐ (Best performer)
- **Annualized Return**: +0.90%
- **Max Drawdown**: -4.94%

---

## 🔄 5. KDJ Strategy

**File**: `kdj_strategy.py`

### What it does:
- **Buy Signal**: When %K crosses above %D in oversold region (< 20)
- **Sell Signal**: When %K crosses below %D in overbought region (> 80)

### How KDJ works:
- **%K Line**: Current close position relative to recent high-low range
- **%D Line**: 3-day moving average of %K
- **%J Line**: 3×%K - 2×%D (more sensitive)
- Similar to Stochastic Oscillator

### When to use:
- For timing entries in ranging markets
- When looking for reversal signals
- In combination with trend-following strategies

### Parameters:
- K Period: 14 days
- D Period: 3 days
- Oversold: < 20
- Overbought: > 80

### Results (AAPL 2024):
- **Return**: -1.04%
- **Annualized Return**: -4.23%
- **Max Drawdown**: -5.00%

---

## 📊 6. ZMR (Zero Mean Reversion) Strategy

**File**: `zmr_strategy.py`

### What it does:
- **Buy Signal**: When price is 2+ standard deviations below its 20-day mean
- **Sell Signal**: When price is 2+ standard deviations above its 20-day mean
- **Exit Signal**: When price returns within 0.5 standard deviations of mean

### How ZMR works:
- **Z-Score** = (Current Price - Rolling Mean) / Rolling Standard Deviation
- **Mean Reversion**: Assumes prices return to their average over time
- **Statistical Approach**: Uses standard deviation to identify extremes

### When to use:
- In ranging/sideways markets
- When prices tend to revert to mean
- For contrarian trading strategies

### Parameters:
- Lookback Period: 20 days
- Entry Threshold: ±2.0 standard deviations
- Exit Threshold: ±0.5 standard deviations

### Results (AAPL 2024):
- **Return**: +0.22% ⭐ (Tied for best)
- **Annualized Return**: +0.90%
- **Max Drawdown**: -4.94%

---

## 📋 Strategy Comparison Summary

| Strategy   | Return% | AR%     | Max DD% | Trades | Type           |
|------------|---------|---------|---------|--------|----------------|
| **RSI**    | +0.22   | +0.90   | -4.94   | 0      | Mean Reversion |
| **ZMR**    | +0.22   | +0.90   | -4.94   | 0      | Mean Reversion |
| **SMA**    | 0.00    | 0.00    | 0.00    | 0      | Trend Following|
| **KDJ**    | -1.04   | -4.23   | -5.00   | 0      | Mean Reversion |
| **MACD**   | -1.82   | -7.29   | -4.96   | 0      | Momentum      |
| **Buy&Hold**| -5.90   | -22.21  | -13.25  | 0      | Passive       |

---

## 🎓 Key Learning Points

### 1. **Strategy Categories**:
- **Trend Following**: SMA, MACD (follow market direction)
- **Mean Reversion**: RSI, KDJ, ZMR (expect price reversals)
- **Passive**: Buy & Hold (no active trading)

### 2. **Risk Management**:
- **Max Drawdown** shows maximum loss from peak
- **Lower drawdown** = better risk management
- **Active strategies** can reduce drawdown vs. buy & hold

### 3. **Market Conditions Matter**:
- **Trending Markets**: Favor trend-following strategies
- **Ranging Markets**: Favor mean-reversion strategies
- **Volatile Markets**: May trigger more false signals

### 4. **No Trading ≠ No Strategy**:
- Several strategies showed 0 trades in this period
- This indicates disciplined waiting for proper signals
- Sometimes not trading is the best action

---

## 🚀 Getting Started

### Running Individual Strategies:
```bash
python buy_hold.py      # Run Buy & Hold
python sma_strategy.py   # Run SMA strategy
python rsi_strategy.py   # Run RSI strategy
# ... etc
```

### Running All Strategies:
```bash
python run_all_strategies.py
```

### Output Files:
- **HTML Plots**: Interactive charts for each strategy
- **Results Files**: Detailed metrics and statistics
- **Comparison Chart**: Side-by-side strategy comparison
- **All saved in**: `results/AAPL/` directory

---

## 📚 Next Steps

1. **Experiment** with different parameters
2. **Combine** strategies for better performance
3. **Add** risk management rules (stop-losses, position sizing)
4. **Test** on different time periods and stocks
5. **Study** market conditions where each strategy performs best

---

## ⚠️ Important Disclaimers

- **Past performance** does not guarantee future results
- **Backtesting** may not reflect real trading conditions
- **Transaction costs** and **slippage** can significantly impact returns
- Always **paper trade** before using real money
- Consider **market regime changes** and **black swan events**

---

*Happy Trading! 📈*