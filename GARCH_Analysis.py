import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

# Check for arch library, else use fallback
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("WARNING: 'arch' library not found. Using SciPy fallback.")

class MarketRiskEngineV2:
    def __init__(self, tickers: list, start_date: str, end_date: str = None):
        self.tickers = tickers
        print(f"Fetching data for {tickers}...")
        self.data = yf.download(tickers, start=start_date, end=end_date)['Close']
        self.data.dropna(inplace=True)
        # Log returns are required for GARCH
        self.log_returns = np.log(self.data / self.data.shift(1)).dropna()

    def _fit_garch_scipy(self, returns: np.ndarray) -> tuple:
        """
        Dependency-free GARCH(1,1) estimation using Maximum Likelihood.
        """
        # Scale returns to percentage for numerical stability
        returns_s = returns * 100
        
        def neg_log_likelihood(params, r):
            omega, alpha, beta = params
            sigma2 = np.zeros(len(r))
            sigma2[0] = np.var(r)
            for t in range(1, len(r)):
                sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
            sigma2 = np.maximum(sigma2, 1e-10) # Safety floor
            return 0.5 * (np.sum(np.log(sigma2)) + np.sum(r**2 / sigma2))

        # Constraints: alpha + beta < 1 (Stationarity)
        cons = ({'type': 'ineq', 'fun': lambda x: 0.999 - (x[1] + x[2])})
        bounds = ((1e-6, None), (1e-4, 1), (1e-4, 1))
        
        # Initial guess: Omega ~ Variance * (1 - alpha - beta)
        res = minimize(neg_log_likelihood, [0.01, 0.1, 0.85], args=(returns_s,),
                       bounds=bounds, constraints=cons, method='SLSQP')
        
        # Unscale omega
        omega = res.x[0] / 10000
        alpha, beta = res.x[1], res.x[2]
        return omega, alpha, beta

    def forecast_volatility(self, ticker: str, horizon: int) -> np.ndarray:
        """
        Forecasts variance vector sigma^2 for the next N days.
        """
        series = self.log_returns[ticker]
        
        if HAS_ARCH:
            # PROFESSIONAL METHOD
            am = arch_model(series * 100, vol='Garch', p=1, o=0, q=1, dist='Normal')
            res = am.fit(disp='off')
            forecasts = res.forecast(horizon=horizon)
            # Rescale variance back to decimal
            return np.sqrt(forecasts.variance.values[-1, :] / 10000)
        
        else:
            # FALLBACK METHOD (Manual Recursion)
            returns = series.values
            omega, alpha, beta = self._fit_garch_scipy(returns)
            
            # Forecast
            forecast_var = np.zeros(horizon)
            # Estimate current variance (last step)
            current_var = np.var(returns) # Simplified initialization
            # E[sigma^2_{t+k}] -> Mean reversion
            long_run_var = omega / (1 - alpha - beta)
            persistence = alpha + beta
            
            # Recursion for T+1
            forecast_var[0] = omega + alpha * returns[-1]**2 + beta * current_var
            
            for k in range(1, horizon):
                forecast_var[k] = long_run_var + (persistence**k) * (forecast_var[0] - long_run_var)
                
            return np.sqrt(forecast_var)

    def simulate_gbm_garch(self, ticker: str, days: int, simulations: int = 10000):
        """
        Monte Carlo with GARCH Volatility Term Structure.
        """
        S0 = self.data[ticker].iloc[-1]
        
        # 1. Get Volatility Path (Vector of length `days`)
        sigma_path = self.forecast_volatility(ticker, days)
        
        # 2. Setup Vectorized Simulation
        dt = 1/252
        Z = np.random.normal(0, 1, (simulations, days))
        
        # 3. Apply Dynamic Volatility
        # sigma_path is 1D (days,), we broadcast to (simulations, days)
        drift_matrix = -0.5 * (sigma_path ** 2) * dt  # Assuming mu=0 for risk neutral
        diffusion_matrix = sigma_path * np.sqrt(dt) * Z
        
        daily_log_returns = drift_matrix + diffusion_matrix
        cumulative_log_returns = np.cumsum(daily_log_returns, axis=1)
        
        price_paths = S0 * np.exp(cumulative_log_returns)
        return price_paths, sigma_path

# --- EXECUTION EXAMPLE ---

engine = MarketRiskEngineV2(['AAPL', 'GOOGL'], start_date='2020-01-01')

# 1. Simulate with GARCH
horizon = 20
paths, vol_term_structure = engine.simulate_gbm_garch('GOOGL', horizon)

# 2. Risk Metrics
S0 = paths[0,0] # Actually S0 is not in paths, paths starts at T+1. 
# Correction: paths should implicitly start from S0 or we reference S0.
final_prices = paths[:, -1]
VaR_99 = np.percentile(final_prices, 1) - engine.data['GOOGL'].iloc[-1]

print(f"GARCH VaR (99%): ${VaR_99:.2f}")

# 3. Visualize Volatility Forecast
plt.figure(figsize=(10,5))
plt.plot(vol_term_structure, marker='o', label='GARCH Forecast')
plt.axhline(engine.log_returns['GOOGL'].std(), color='r', linestyle='--', label='Historical Avg')
plt.title(f"Volatility Term Structure: GOOGL ({horizon} Days)")
plt.legend()
plt.show()