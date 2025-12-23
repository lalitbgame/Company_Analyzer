
import re
import streamlit as st
import pandas as pd

from financialstatementfunctions_p import (
    get_MultipleBalanceSheet,
    get_WholeRatio,
    convert_df,
    get_MultipleFinancial,
    get_MultipleCashFlow,
    generate_tabs,
    get_MultipleProfitabilityRatios,
    get_MultipleLiquidityRatios,
    get_MultipleEfficiencyRatios,
    extract_balance_sheet,
    get_Financial,
    get_CashFLow,
)

from health import score_company_health

# ----------------------------
# Page config + styling
# ----------------------------
st.set_page_config(page_title="Aadivira — Fundamentals & Health", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
      .fv-title { font-size: 1.65rem; font-weight: 700; margin-bottom: .15rem; }
      .fv-sub { color: #6b7280; margin-top: 0; }
      .fv-card {
        border: 1px solid rgba(120,120,120,.20);
        border-radius: 16px;
        padding: 14px 16px;
        background: rgba(255,255,255,.55);
      }
      .fv-muted { color: #6b7280; font-size: .92rem; }
      .fv-badge {
        display: inline-block; padding: 4px 10px; border-radius: 999px;
        border: 1px solid rgba(120,120,120,.25); font-size: .85rem;
      }
      .fv-footer { color: #9ca3af; font-size: .85rem; }
      .stMetric { background: transparent !important; }
      div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers
# ----------------------------
def _dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _clean_ticker_list(text: str):
    if not text or not text.strip():
        return []
    parts = re.split(r"[,\s]+", text.strip())
    return [p.upper() for p in parts if p]

def _fmt_num(x):
    try:
        if x is None or pd.isna(x):
            return "—"
        x = float(x)
        ax = abs(x)
        if ax >= 1e12:
            return f"{x/1e12:,.2f}T"
        if ax >= 1e9:
            return f"{x/1e9:,.2f}B"
        if ax >= 1e6:
            return f"{x/1e6:,.2f}M"
        if ax >= 1e3:
            return f"{x:,.0f}"
        return f"{x:,.2f}"
    except Exception:
        return str(x)

def _rating_badge(rating: str):
    # Keep minimal, no hard-coded colors; rely on default theme.
    return f"<span class='fv-badge'>{rating}</span>"

@st.cache_data(ttl=60 * 60, show_spinner=False)
def _load_statements(tickers: tuple[str, ...]):
    # These functions are from your existing module.
    bs = get_MultipleBalanceSheet(list(tickers))
    fs = get_MultipleFinancial(list(tickers))
    cf = get_MultipleCashFlow(list(tickers))
    return bs, fs, cf

@st.cache_data(ttl=60 * 60, show_spinner=False)
def _load_ratios(tickers: tuple[str, ...]):
    prof = get_MultipleProfitabilityRatios(list(tickers))
    liq = get_MultipleLiquidityRatios(list(tickers))
    eff = get_MultipleEfficiencyRatios(list(tickers))
    allr = get_WholeRatio(list(tickers))
    return prof, liq, eff, allr

def _latest_snapshot(df: pd.DataFrame, tickers: list[str], cols: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        d = df[df["Company"] == t].sort_index()
        if d.empty:
            continue
        row = d.iloc[-1]
        rows.append([t] + [row.get(c) for c in cols])
    return pd.DataFrame(rows, columns=["Company"] + cols)

def _build_health_for_ticker(ticker: str):
    # Uses your single-ticker helpers to avoid shape issues.
    current_Assets, non_current_Assets, total_Assets, current_Liabilities, non_current_Liabilities, total_Liabilities, equity = extract_balance_sheet(ticker)
    bs_single = pd.concat(
        [current_Assets, non_current_Assets, total_Assets, current_Liabilities, non_current_Liabilities, total_Liabilities, equity],
        axis=1
    )
    fs_single = get_Financial(ticker)
    cf_single = get_CashFLow(ticker)
    return score_company_health(ticker, bs_single, fs_single, cf_single)

# ----------------------------
# Sidebar: professional controls
# ----------------------------
st.sidebar.markdown("### Aadivira")
st.sidebar.caption("Fundamentals & health scoring from Yahoo Finance data.")

market = st.sidebar.selectbox("Market", ["US / Global", "India — NSE (.NS)", "India — BSE (.BO)"], index=0)
suffix = "" if market == "US / Global" else (".NS" if "NSE" in market else ".BO")

st.sidebar.markdown("#### Quick pick")
global_presets = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Nvidia": "NVDA",
    "Google": "GOOG",
    "Amazon": "AMZN",
}
india_presets = {
    "Reliance": f"RELIANCE{suffix}",
    "TCS": f"TCS{suffix}",
    "HDFC Bank": f"HDFCBANK{suffix}",
    "Infosys": f"INFY{suffix}",
    "ITC": f"ITC{suffix}",
    "L&T": f"LT{suffix}",
    "SBI": f"SBIN{suffix}",
    "Airtel": f"BHARTIARTL{suffix}",
    "Asian Paints": f"ASIANPAINT{suffix}",
    "HUL": f"HINDUNILVR{suffix}",
}

presets = global_presets if market == "US / Global" else india_presets
picked = st.sidebar.multiselect("Presets", list(presets.keys()), default=list(presets.keys())[:2])

st.sidebar.markdown("#### Add tickers")
manual = st.sidebar.text_input("Comma/space-separated", value="")

base_manual = _clean_ticker_list(manual)
manual_full = []
for t in base_manual:
    if suffix and not t.endswith((".NS", ".BO")):
        manual_full.append(t + suffix)
    else:
        manual_full.append(t)

tickers = _dedup_keep_order([presets[n] for n in picked] + manual_full)

st.sidebar.markdown("#### Options")
show_raw = st.sidebar.checkbox("Show raw snapshots", value=False)
compact = st.sidebar.checkbox("Compact tables", value=True)

st.sidebar.divider()
st.sidebar.caption("Tip: use this as a screener. Verify on filings before investing.")

if not tickers:
    st.warning("Select at least one ticker to start.")
    st.stop()

# ----------------------------
# Header
# ----------------------------
st.markdown("<div class='fv-title'>Aadivira — Fundamentals & Health</div>", unsafe_allow_html=True)
st.markdown("<div class='fv-sub'>Compare financial statements, ratios, and a rule-based company health score.</div>", unsafe_allow_html=True)
st.divider()

# ----------------------------
# Navigation
# ----------------------------
page = st.tabs(["Dashboard", "Statements", "Ratios", "Health"])


# ----------------------------
# Dashboard
# ----------------------------
with page[0]:
    st.subheader("Dashboard")

    with st.spinner("Loading statements and ratios..."):
        bs, fs, cf = _load_statements(tuple(tickers))
        _, _, _, allr = _load_ratios(tuple(tickers))

    # KPI row
    c1, c2, c3, c4 = st.columns(4)

    # Health scores
    health_rows = []
    for t in tickers:
        try:
            h = _build_health_for_ticker(t)
            health_rows.append({"Company": t, "Score": h["score"], "Rating": h["rating"]})
        except Exception:
            health_rows.append({"Company": t, "Score": None, "Rating": "No Data"})

    health_df = pd.DataFrame(health_rows).sort_values(by="Score", ascending=False, na_position="last")

    top = health_df.iloc[0].to_dict() if not health_df.empty else None
    with c1:
        st.metric("Companies", len(tickers))
    with c2:
        st.metric("Top Score", f"{int(top['Score'])}/100" if top and pd.notna(top.get("Score")) else "—")
    with c3:
        st.metric("Top Company", top["Company"] if top else "—")
    with c4:
        st.markdown(_rating_badge(top["Rating"] if top else "—"), unsafe_allow_html=True)

    st.markdown("##### Health score ranking")
    st.dataframe(health_df, use_container_width=True, hide_index=True)

    # Quick snapshots
    st.markdown("##### Latest snapshots (latest reported year)")
    snapA, snapB = st.columns(2)

    with snapA:
        st.markdown("<div class='fv-card'><b>Balance Sheet</b><div class='fv-muted'>Core strength & liquidity</div></div>", unsafe_allow_html=True)
        bs_cols = [c for c in ["Total Assets", "Total Liabilities Net Minority Interest", "Total Equity Gross Minority Interest", "Cash And Cash Equivalents", "Current Assets", "Current Liabilities"] if c in bs.columns]
        snap_bs = _latest_snapshot(bs, tickers, bs_cols)
        if compact:
            st.dataframe(snap_bs, use_container_width=True, hide_index=True)
        else:
            st.dataframe(snap_bs, use_container_width=True)

    with snapB:
        st.markdown("<div class='fv-card'><b>Income & Cash</b><div class='fv-muted'>Profitability & cash quality</div></div>", unsafe_allow_html=True)
        fs_cols = [c for c in ["Total Revenue", "EBIT", "Net Income", "Gross Profit"] if c in fs.columns]
        cf_cols = [c for c in ["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure"] if c in cf.columns]
        snap_fs = _latest_snapshot(fs, tickers, fs_cols)
        snap_cf = _latest_snapshot(cf, tickers, cf_cols)
        st.write("Income Statement")
        st.dataframe(snap_fs, use_container_width=True, hide_index=True)
        st.write("Cash Flow")
        st.dataframe(snap_cf, use_container_width=True, hide_index=True)

    st.markdown("##### Quick ratio view (selected)")
    if not allr.empty:
        # Let user choose which ratios to show
        ratio_cols = [c for c in allr.columns if c != "Company"]
        default_cols = ratio_cols[:6] if len(ratio_cols) >= 6 else ratio_cols
        selected = st.multiselect("Select ratios", ratio_cols, default=default_cols)
        if selected:
            view = allr[["Company"] + selected].copy()
            st.dataframe(view, use_container_width=True, hide_index=True)

    st.caption("Data source: Yahoo Finance via yfinance. Some tickers may have missing statement fields.")


# ----------------------------
# Statements
# ----------------------------
with page[1]:
    st.subheader("Statements")
    st.write("Explore full statements per company. (Uses your existing statement explorer UI.)")
    generate_tabs(tickers)


# ----------------------------
# Ratios
# ----------------------------
with page[2]:
    st.subheader("Ratios")
    with st.spinner("Loading ratios..."):
        prof, liq, eff, allr = _load_ratios(tuple(tickers))

    st.markdown("##### Complete ratio sheet")
    st.dataframe(allr, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download ratios CSV",
        data=convert_df(allr),
        file_name="FinVista_Ratios.csv",
        mime="text/csv",
    )

    st.markdown("##### Ratio explorer")
    ratio_cols = [c for c in allr.columns if c != "Company"]
    chosen = st.multiselect("Choose ratios to plot", ratio_cols, default=ratio_cols[:3] if ratio_cols else [])
    if chosen:
        # Build a simple line chart from the "allr" sheet if it contains time index.
        # If your ratio sheet isn't time-indexed, we just show bar-like comparison.
        comp = allr[["Company"] + chosen].copy()
        comp = comp.set_index("Company")
        st.line_chart(comp)  # Streamlit defaults for colors/style


# ----------------------------
# Health
# ----------------------------
with page[3]:
    st.subheader("Health")
    st.write("Rule-based screening score. Treat as **first-pass filtering**, then validate in annual reports/filings.")

    rows = []
    snapshots = []
    metric_tables = {}

    with st.spinner("Scoring companies..."):
        for t in tickers:
            try:
                h = _build_health_for_ticker(t)
                rows.append({
                    "Company": t,
                    "Score": h["score"],
                    "Rating": h["rating"],
                    "Top flags": " | ".join(h["flags"][:3]) + (" ..." if len(h["flags"]) > 3 else ""),
                })
                snap = h["snapshot"].copy()
                snap["Company"] = t
                snapshots.append(snap)
                metric_tables[t] = pd.DataFrame(
                    {"Metric": list(h["metric_scores"].keys()), "Score (0-10)": list(h["metric_scores"].values())}
                )
            except Exception as e:
                rows.append({"Company": t, "Score": None, "Rating": "No Data", "Top flags": "Unable to score (missing Yahoo fields)."})
                metric_tables[t] = pd.DataFrame({"Metric": [], "Score (0-10)": []})

    scorecard = pd.DataFrame(rows).sort_values(by="Score", ascending=False, na_position="last")
    st.dataframe(scorecard, use_container_width=True, hide_index=True)

    left, right = st.columns([1, 1])
    with left:
        pick = st.selectbox("Inspect company", tickers, index=0)
    with right:
        show_metric = st.checkbox("Show metric breakdown", value=True)

    if pick:
        try:
            h = _build_health_for_ticker(pick)
            a, b, c = st.columns(3)
            a.metric("Health Score", f"{h['score']}/100")
            b.metric("Rating", h["rating"])
            cr = h["snapshot"].get("Current Ratio")
            c.metric("Current Ratio", f"{cr:.2f}" if isinstance(cr, (int, float)) and cr == cr else "—")

            if h["flags"]:
                st.warning("Key flags:\n- " + "\n- ".join(h["flags"]))

            if show_metric:
                st.markdown("##### Metric breakdown")
                st.dataframe(metric_tables[pick], use_container_width=True, hide_index=True)

            if show_raw:
                st.markdown("##### Raw snapshot (latest year)")
                snap = pd.DataFrame([h["snapshot"]]).T.reset_index()
                snap.columns = ["Line Item", "Value"]
                snap["Value"] = snap["Value"].map(_fmt_num)
                st.dataframe(snap, use_container_width=True, hide_index=True)

        except Exception:
            st.error("Could not build a detailed view for this ticker (Yahoo Finance statement fields missing).")

    st.markdown("<div class='fv-footer'>FinVista is a screening tool. Always cross-check with official filings before taking risk.</div>", unsafe_allow_html=True)
