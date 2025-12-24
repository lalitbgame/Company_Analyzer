import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from typing import Tuple, Dict

# Check for yfinance, define fallback if missing (for VM execution)
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# Set style and seed
sns.set_style('whitegrid')
np.random.seed(42)

class MarketRiskEngine:
    def __init__(self, tickers: list, start_date: str, end_date: str = None):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
        if HAS_YFINANCE:
            print(f"Fetching data for {tickers} from Yahoo Finance...")
            try:
                raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if 'Close' in raw_data:
                    self.data = raw_data['Close']
                else:
                    self.data = raw_data
                
                if isinstance(self.data, pd.Series):
                    self.data = self.data.to_frame()
                    self.data.columns = tickers
                
                self.data.dropna(inplace=True)
            except Exception as e:
                print(f"YFinance failed: {e}. Switching to Synthetic Data.")
                self._generate_synthetic_data()
        else:
            print("YFinance not installed. Using Synthetic Data.")
            self._generate_synthetic_data()

        # Log returns
        self.log_returns = np.log(self.data / self.data.shift(1)).dropna()

    def _generate_synthetic_data(self):
        # Fallback for VM without internet/yfinance
        dates = pd.date_range(start=self.start_date, end='2024-01-01', freq='B')
        data_dict = {}
        for ticker in self.tickers:
            T = len(dates)
            S0 = 150.0
            mu = 0.10
            sigma = 0.30
            dt = 1/252
            rets = np.random.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt), T)
            prices = S0 * np.exp(np.cumsum(rets))
            data_dict[ticker] = prices
        self.data = pd.DataFrame(data_dict, index=dates)

    def get_stats(self, ticker: str) -> Tuple[float, float, float]:
        mu = 0 
        sigma = self.log_returns[ticker].std() * np.sqrt(252)
        S0 = self.data[ticker].iloc[-1]
        return mu, sigma, S0

    def simulate_gbm(self, ticker: str, days: int, simulations: int) -> np.ndarray:
        mu, sigma, S0 = self.get_stats(ticker)
        dt = 1/252
        Z = np.random.normal(0, 1, (simulations, days))
        drift_term = (mu - 0.5 * sigma**2) * dt
        diffusion_term = sigma * np.sqrt(dt) * Z
        daily_log_returns = drift_term + diffusion_term
        cumulative_log_returns = np.cumsum(daily_log_returns, axis=1)
        price_paths = S0 * np.exp(cumulative_log_returns)
        S0_vec = np.full((simulations, 1), S0)
        return np.hstack([S0_vec, price_paths])

    def simulate_historical_bootstrap(self, ticker: str, days: int, simulations: int) -> np.ndarray:
        S0 = self.data[ticker].iloc[-1]
        hist_rets = self.log_returns[ticker].values
        random_rets = np.random.choice(hist_rets, size=(simulations, days))
        cumulative_log_returns = np.cumsum(random_rets, axis=1)
        price_paths = S0 * np.exp(cumulative_log_returns)
        S0_vec = np.full((simulations, 1), S0)
        return np.hstack([S0_vec, price_paths])

    def compute_risk_metrics(self, price_paths: np.ndarray, alpha: float = 0.01) -> Dict:
        S0 = price_paths[0, 0]
        terminal_prices = price_paths[:, -1]
        pnl = terminal_prices - S0
        var = np.percentile(pnl, alpha * 100)
        cvar = pnl[pnl <= var].mean()
        return {
            "Start Price": S0,
            "Mean Final Price": terminal_prices.mean(),
            "5th Percentile": np.percentile(terminal_prices, 5),
            "95th Percentile": np.percentile(terminal_prices, 95),
            f"VaR ({1-alpha:.0%})": var,
            f"CVaR ({1-alpha:.0%})": cvar
        }

    def calculate_ma(self, ticker, sma_window=10, ema_window=10):
        series = self.data[ticker]
        sma = series.rolling(window=sma_window).mean()
        ema = series.ewm(span=ema_window, adjust=False).mean()
        return sma, ema

    def forecast_ma(self, series, days_forecast=30):
        # Linear extrapolation of the last 10 days
        y = series.dropna().iloc[-10:].values
        x = np.arange(len(y))
        slope, intercept, _, _, _ = linregress(x, y)
        
        future_x = np.arange(len(y), len(y) + days_forecast)
        future_y = slope * future_x + intercept
        return future_y

# --- EXECUTION ---
target_stock = 'NVDA'
engine = MarketRiskEngine([target_stock], start_date='2020-01-01')

days_forecast = 30
n_sims = 100

# Simulations
gbm_paths = engine.simulate_gbm(target_stock, days_forecast, simulations=n_sims)
boot_paths = engine.simulate_historical_bootstrap(target_stock, days_forecast, simulations=n_sims)

# Metrics
gbm_metrics = engine.compute_risk_metrics(gbm_paths)
boot_metrics = engine.compute_risk_metrics(boot_paths)

# Report
print(f"\nResults (Current Price: ${gbm_metrics['Start Price']:.2f})")
print("-" * 65)
print(f"{'Metric':<25} | {'GBM (Normal)':<18} | {'Bootstrap (History)':<18}")
print("-" * 65)
print(f"{'Range (5th-95th)':<25} | ${gbm_metrics['5th Percentile']:.1f} - ${gbm_metrics['95th Percentile']:.1f}   | ${boot_metrics['5th Percentile']:.1f} - ${boot_metrics['95th Percentile']:.1f}")
print(f"{'Mean Forecast':<25} | ${gbm_metrics['Mean Final Price']:.2f}{'':<12} | ${boot_metrics['Mean Final Price']:.2f}")
print(f"{'VaR (99% Risk)':<25} | ${gbm_metrics['VaR (99%)']:.2f}{'':<12} | ${boot_metrics['VaR (99%)']:.2f}")
print("-" * 65)

# --- VISUALIZATION ---
# Prepare MA Data
sma, ema = engine.calculate_ma(target_stock, sma_window=50, ema_window=20)
sma_forecast = engine.forecast_ma(sma, days_forecast)
ema_forecast = engine.forecast_ma(ema, days_forecast)

# Plot 1: History (Last 2 Years)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
# Slice last 2 years (approx 504 trading days)
history_window = 504
hist_dates = engine.data.index[-history_window:]
hist_price = engine.data[target_stock].iloc[-history_window:]
hist_sma = sma.iloc[-history_window:]
hist_ema = ema.iloc[-history_window:]

plt.plot(hist_dates, hist_price, label='Close Price', color='black', alpha=0.7)
plt.plot(hist_dates, hist_sma, label='SMA (50)', color='blue', linestyle='--')
plt.plot(hist_dates, hist_ema, label='EMA (50)', color='green', linestyle='--')
plt.title(f"2-Year History: {target_stock}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)

# Plot 2: Forecast (Next 30 Days)
plt.subplot(1, 2, 2)
future_dates = pd.date_range(start=engine.data.index[-1], periods=days_forecast+1, freq='B')

# MC Paths (GBM)
for i in range(min(50, n_sims)):
    plt.plot(future_dates, gbm_paths[i, :], color='grey', alpha=0.1)
plt.plot(future_dates, gbm_paths.mean(axis=0), color='red', linewidth=2, label='MC Mean (GBM)')

# MA Forecasts (Align: Start at T+1)
# Need to append last historical point to connect lines
last_sma = sma.iloc[-1]
last_ema = ema.iloc[-1]
plot_sma_forecast = np.concatenate(([last_sma], sma_forecast))
plot_ema_forecast = np.concatenate(([last_ema], ema_forecast))

plt.plot(future_dates, plot_sma_forecast, color='blue', linestyle=':', linewidth=2, label='SMA Forecast (Linear)')
plt.plot(future_dates, plot_ema_forecast, color='green', linestyle=':', linewidth=2, label='EMA Forecast (Linear)')

plt.title(f"30-Day Forecast: {target_stock}")
plt.xlabel("Date")
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('history_and_forecast.png')
plt.show()