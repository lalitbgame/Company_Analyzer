
import pandas as pd

def _last_n_years(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.sort_index()
    # yfinance indexes are often Timestamp year-end dates
    return df.iloc[-n:]

def _safe_div(a, b):
    try:
        if b is None or b == 0:
            return None
        return a / b
    except Exception:
        return None

def score_company_health(
    ticker: str,
    balance_sheet: pd.DataFrame,
    financials: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> dict:
    """
    Returns a dict with:
      - score (0-100)
      - rating (Strong / Okay / Weak / Risky)
      - metric_scores: per-metric numeric scores
      - flags: list[str]
      - snapshot: latest-year values for key fields
    Assumes inputs are transposed (index=time, columns=line items) as produced by the project functions.
    """
    flags = []
    metric_scores = {}
    snapshot = {}

    # --- Pick latest year row ---
    bs = balance_sheet.sort_index() if balance_sheet is not None else pd.DataFrame()
    fs = financials.sort_index() if financials is not None else pd.DataFrame()
    cf = cashflow.sort_index() if cashflow is not None else pd.DataFrame()

    if bs.empty or fs.empty or cf.empty:
        return {
            "score": 0,
            "rating": "No Data",
            "metric_scores": {},
            "flags": ["Missing statements from Yahoo Finance for this ticker."],
            "snapshot": {}
        }

    latest_bs = bs.iloc[-1]
    latest_fs = fs.iloc[-1]
    latest_cf = cf.iloc[-1]

    # --- Core values (best-effort) ---
    cur_assets = latest_bs.get("Current Assets")
    cur_liab = latest_bs.get("Current Liabilities")
    tot_assets = latest_bs.get("Total Assets")
    tot_liab = latest_bs.get("Total Liabilities Net Minority Interest")
    equity = latest_bs.get("Total Equity Gross Minority Interest")
    cash = latest_bs.get("Cash And Cash Equivalents")

    revenue = latest_fs.get("Total Revenue")
    ebit = latest_fs.get("EBIT")  # yfinance provides EBIT
    gross_profit = latest_fs.get("Gross Profit")
    net_income = latest_fs.get("Net Income")

    ocf = latest_cf.get("Operating Cash Flow")
    fcf = latest_cf.get("Free Cash Flow")
    capex = latest_cf.get("Capital Expenditure")

    # Store snapshot
    snapshot.update({
        "Current Assets": cur_assets,
        "Current Liabilities": cur_liab,
        "Total Assets": tot_assets,
        "Total Liabilities": tot_liab,
        "Total Equity": equity,
        "Cash & Equivalents": cash,
        "Revenue": revenue,
        "EBIT": ebit,
        "Gross Profit": gross_profit,
        "Net Income": net_income,
        "Operating Cash Flow": ocf,
        "Free Cash Flow": fcf,
        "Capex": capex,
    })

    # --- Helper: last 3-year trends (working-capital stress, leverage trend, cash conversion trend) ---
    bs3 = _last_n_years(bs, 3)
    fs3 = _last_n_years(fs, 3)
    cf3 = _last_n_years(cf, 3)

    def _trend_pct(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) < 2:
            return None
        first, last = s.iloc[0], s.iloc[-1]
        if first == 0 or pd.isna(first) or pd.isna(last):
            return None
        return (last - first) / abs(first)

    # --- Metrics & scoring (0..10 each, total scaled to 100) ---
    # 1) Liquidity: current ratio
    current_ratio = _safe_div(cur_assets, cur_liab)
    snapshot["Current Ratio"] = current_ratio
    if current_ratio is None:
        metric_scores["Liquidity (Current Ratio)"] = 0
        flags.append("Could not compute Current Ratio (missing current assets/liabilities).")
    elif current_ratio >= 1.5:
        metric_scores["Liquidity (Current Ratio)"] = 10
    elif current_ratio >= 1.0:
        metric_scores["Liquidity (Current Ratio)"] = 7
    elif current_ratio >= 0.8:
        metric_scores["Liquidity (Current Ratio)"] = 4
        flags.append("Current Ratio is below 1.0 (tight short-term liquidity).")
    else:
        metric_scores["Liquidity (Current Ratio)"] = 1
        flags.append("Current Ratio is very low (high short-term liquidity risk).")

    # 2) Leverage: debt-to-equity (using total liabilities as proxy)
    debt_to_equity = _safe_div(tot_liab, equity)
    snapshot["Debt to Equity (Liab/Equity)"] = debt_to_equity
    if debt_to_equity is None:
        metric_scores["Leverage (Debt/Equity proxy)"] = 0
        flags.append("Could not compute leverage ratio (missing liabilities/equity).")
    elif debt_to_equity <= 1.0:
        metric_scores["Leverage (Debt/Equity proxy)"] = 10
    elif debt_to_equity <= 2.0:
        metric_scores["Leverage (Debt/Equity proxy)"] = 7
    elif debt_to_equity <= 3.0:
        metric_scores["Leverage (Debt/Equity proxy)"] = 4
        flags.append("High leverage (liabilities materially exceed equity).")
    else:
        metric_scores["Leverage (Debt/Equity proxy)"] = 1
        flags.append("Very high leverage (risk increases sharply in downturns).")

    # 3) Solvency: debt-to-assets (liabilities/assets)
    debt_to_assets = _safe_div(tot_liab, tot_assets)
    snapshot["Debt to Assets (Liab/Assets)"] = debt_to_assets
    if debt_to_assets is None:
        metric_scores["Solvency (Liab/Assets)"] = 0
    elif debt_to_assets <= 0.5:
        metric_scores["Solvency (Liab/Assets)"] = 10
    elif debt_to_assets <= 0.7:
        metric_scores["Solvency (Liab/Assets)"] = 7
    elif debt_to_assets <= 0.85:
        metric_scores["Solvency (Liab/Assets)"] = 4
        flags.append("Liabilities are a large share of assets (solvency weaker).")
    else:
        metric_scores["Solvency (Liab/Assets)"] = 1
        flags.append("Liabilities dominate the asset base (solvency risk).")

    # 4) Profitability: operating margin (EBIT / Revenue)
    op_margin = _safe_div(ebit, revenue)
    snapshot["Operating Margin (EBIT/Revenue)"] = op_margin
    if op_margin is None:
        metric_scores["Profitability (Operating Margin)"] = 0
        flags.append("Could not compute operating margin (missing EBIT/revenue).")
    elif op_margin >= 0.20:
        metric_scores["Profitability (Operating Margin)"] = 10
    elif op_margin >= 0.12:
        metric_scores["Profitability (Operating Margin)"] = 7
    elif op_margin >= 0.06:
        metric_scores["Profitability (Operating Margin)"] = 4
        flags.append("Thin operating margin (profits can vanish in a slowdown).")
    else:
        metric_scores["Profitability (Operating Margin)"] = 1
        flags.append("Very low operating margin (business model looks fragile).")

    # 5) Unit economics: gross margin (Gross Profit / Revenue)
    gross_margin = _safe_div(gross_profit, revenue)
    snapshot["Gross Margin (Gross Profit/Revenue)"] = gross_margin
    if gross_margin is None:
        metric_scores["Unit Economics (Gross Margin)"] = 0
    elif gross_margin >= 0.40:
        metric_scores["Unit Economics (Gross Margin)"] = 10
    elif gross_margin >= 0.25:
        metric_scores["Unit Economics (Gross Margin)"] = 7
    elif gross_margin >= 0.15:
        metric_scores["Unit Economics (Gross Margin)"] = 4
        flags.append("Low gross margin (limited pricing power or high input costs).")
    else:
        metric_scores["Unit Economics (Gross Margin)"] = 1
        flags.append("Very low gross margin (watch competitive pressure).")

    # 6) Earnings quality: OCF vs Net Income
    cash_to_income = _safe_div(ocf, net_income)
    snapshot["OCF / Net Income"] = cash_to_income
    if cash_to_income is None:
        metric_scores["Earnings Quality (OCF/Net Income)"] = 0
        flags.append("Could not compute OCF/Net Income (missing OCF or Net Income).")
    elif cash_to_income >= 1.2:
        metric_scores["Earnings Quality (OCF/Net Income)"] = 10
    elif cash_to_income >= 0.9:
        metric_scores["Earnings Quality (OCF/Net Income)"] = 7
    elif cash_to_income >= 0.6:
        metric_scores["Earnings Quality (OCF/Net Income)"] = 4
        flags.append("Cash conversion is weak (profits not turning into cash).")
    else:
        metric_scores["Earnings Quality (OCF/Net Income)"] = 1
        flags.append("Very weak cash conversion (potential accrual risk).")

    # 7) Free cash flow health: FCF positive + capex intensity proxy
    if pd.isna(fcf):
        metric_scores["Free Cash Flow (FCF)"] = 0
        flags.append("Free Cash Flow missing from Yahoo Finance for this ticker.")
    else:
        if fcf > 0:
            metric_scores["Free Cash Flow (FCF)"] = 10
        elif fcf > -0.05 * (revenue if revenue else 1):
            metric_scores["Free Cash Flow (FCF)"] = 6
            flags.append("FCF slightly negative (may be investment-heavy period).")
        else:
            metric_scores["Free Cash Flow (FCF)"] = 2
            flags.append("FCF materially negative (funding needs can rise).")

    # 8) Trend sanity checks (3-year)
    liab_trend = _trend_pct(bs3.get("Total Liabilities Net Minority Interest"))
    rev_trend = _trend_pct(fs3.get("Total Revenue"))
    ni_trend = _trend_pct(fs3.get("Net Income"))
    rec_trend = _trend_pct(bs3.get("Accounts Receivable"))
    inv_trend = _trend_pct(bs3.get("Inventory"))

    trend_points = 10
    if liab_trend is not None and liab_trend > 0.5:
        trend_points -= 3
        flags.append("Liabilities grew >50% over last ~3 reported years (leverage trending up).")
    if rev_trend is not None and rev_trend < 0:
        trend_points -= 3
        flags.append("Revenue trend is negative over last ~3 reported years.")
    if ni_trend is not None and ni_trend < 0:
        trend_points -= 3
        flags.append("Net income trend is negative over last ~3 reported years.")
    if rec_trend is not None and rev_trend is not None and rec_trend > rev_trend + 0.25:
        trend_points -= 2
        flags.append("Receivables rising much faster than revenue (collection risk).")
    if inv_trend is not None and rev_trend is not None and inv_trend > rev_trend + 0.25:
        trend_points -= 2
        flags.append("Inventory rising much faster than revenue (demand/stock risk).")

    metric_scores["Trend Checks (3y)"] = max(trend_points, 0)

    # --- Total score ---
    total = sum(metric_scores.values())
    score = round((total / (10 * len(metric_scores))) * 100) if metric_scores else 0

    if score >= 80:
        rating = "Strong"
    elif score >= 60:
        rating = "Okay"
    elif score >= 40:
        rating = "Weak"
    else:
        rating = "Risky"

    return {
        "score": score,
        "rating": rating,
        "metric_scores": metric_scores,
        "flags": flags,
        "snapshot": snapshot
    }
