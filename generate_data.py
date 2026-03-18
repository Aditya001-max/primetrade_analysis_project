"""
Synthetic data generator – mirrors the real Hyperliquid + Fear/Greed schema.
Replace load_data() in the notebook with pd.read_csv() on the real files.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# ── 1. Fear / Greed index ────────────────────────────────────────────────────
def make_fear_greed(start="2024-01-01", end="2024-12-31"):
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    # Markov-like transitions so sentiment clusters realistically
    states, s = [], "Fear"
    for _ in range(n):
        states.append(s)
        s = np.random.choice(["Fear","Greed"], p=[0.45,0.55] if s=="Greed" else [0.55,0.45])
    return pd.DataFrame({"Date": dates, "Classification": states})

# ── 2. Trader / trade-level data ─────────────────────────────────────────────
SYMBOLS   = ["BTC","ETH","SOL","ARB","DOGE"]
N_TRADERS = 120
N_TRADES  = 55_000

def make_trader_data(fear_greed_df: pd.DataFrame):
    fg = fear_greed_df.set_index("Date")["Classification"]
    dates_range = fear_greed_df["Date"].values

    accounts   = [f"0x{i:04x}" for i in range(N_TRADERS)]
    # Assign stable archetypes
    archetype  = {a: np.random.choice(
                    ["high_lev","low_lev","frequent","infrequent","winner","inconsistent"],
                    p=[.17,.17,.17,.17,.16,.16])
                  for a in accounts}

    rows = []
    for _ in range(N_TRADES):
        acc  = np.random.choice(accounts)
        arch = archetype[acc]
        date = pd.Timestamp(np.random.choice(dates_range))
        sent = fg.get(date.normalize(), "Unknown")

        # Archetype-driven parameters
        lev_mu  = {"high_lev":18,"low_lev":3,"frequent":8,"infrequent":6,"winner":7,"inconsistent":12}[arch]
        size_mu = {"high_lev":3000,"low_lev":800,"frequent":500,"infrequent":4000,"winner":1500,"inconsistent":2000}[arch]
        freq_w  = {"high_lev":1,"low_lev":1,"frequent":4,"infrequent":0.3,"winner":1.5,"inconsistent":1}[arch]

        leverage = max(1, np.random.normal(lev_mu, lev_mu*0.3))
        size     = max(10, np.random.lognormal(np.log(size_mu), 0.6))
        side     = np.random.choice(["BUY","SELL"],
                      p=[0.58,0.42] if sent=="Greed" else [0.42,0.58])

        # PnL model: winners win more, high-lev lose more on Fear
        base_pnl = np.random.normal(0, size * 0.02)
        if arch == "winner":      base_pnl += abs(base_pnl) * 0.3
        if arch == "inconsistent":base_pnl *= np.random.choice([2,-2])
        if arch == "high_lev" and sent=="Fear": base_pnl -= size * 0.015
        if sent == "Greed": base_pnl *= 1.1

        rows.append(dict(
            account    = acc,
            symbol     = np.random.choice(SYMBOLS, p=[.45,.25,.15,.10,.05]),
            execution_price = np.random.uniform(20000,70000) if "BTC" else np.random.uniform(1000,4000),
            size       = round(size, 2),
            side       = side,
            time       = date + pd.Timedelta(hours=np.random.uniform(0,24)),
            start_position = round(np.random.normal(0, size*0.5), 2),
            event      = np.random.choice(["FILL","PARTIAL_FILL"], p=[.85,.15]),
            closedPnL  = round(base_pnl, 4),
            leverage   = round(leverage, 1),
            archetype  = arch,   # kept for validation; drop before submission
        ))

    return pd.DataFrame(rows)

if __name__ == "__main__":
    fg = make_fear_greed()
    td = make_trader_data(fg)
    fg.to_csv("data/fear_greed.csv", index=False)
    td.to_csv("data/trader_data.csv", index=False)
    print(f"Fear/Greed: {fg.shape}  |  Trader data: {td.shape}")
