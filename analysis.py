"""
Primetrade.ai – Trader Performance vs Market Sentiment
Round-0 Data Science Intern Assignment
Author: Candidate
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  IMPORTS & STYLE
# ═══════════════════════════════════════════════════════════════════════════════
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# ── Palette ──────────────────────────────────────────────────────────────────
FEAR_C   = "#E74C3C"
GREED_C  = "#2ECC71"
BG       = "#0F1117"
CARD_BG  = "#1A1D26"
TEXT     = "#E8E8E8"
ACCENT   = "#F39C12"
BLUE     = "#3498DB"

def set_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": CARD_BG,
        "axes.edgecolor": "#333", "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "text.color": TEXT, "grid.color": "#2A2A3A",
        "grid.linestyle": "--", "grid.alpha": 0.5,
        "font.family": "DejaVu Sans", "font.size": 11,
        "legend.facecolor": CARD_BG, "legend.edgecolor": "#444",
    })

set_style()
os.makedirs("charts", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PART A-1 — LOAD & DOCUMENT
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART A — DATA PREPARATION")
print("=" * 70)

# ── To use REAL data, replace these two lines: ───────────────────────────────
#   fg_raw = pd.read_csv("path/to/fear_greed.csv")
#   td_raw = pd.read_csv("path/to/trader_data.csv")
# ─────────────────────────────────────────────────────────────────────────────
fg_raw = pd.read_csv("data/fear_greed.csv")
td_raw = pd.read_csv("data/trader_data.csv")

print("\n── Fear/Greed Dataset ──────────────────────────────────────")
print(f"  Rows: {fg_raw.shape[0]:,}   Cols: {fg_raw.shape[1]}")
print(f"  Columns: {list(fg_raw.columns)}")
print(f"  Missing values:\n{fg_raw.isnull().sum().to_string()}")
print(f"  Duplicate rows: {fg_raw.duplicated().sum()}")
print(f"  Sentiment split:\n{fg_raw['Classification'].value_counts().to_string()}")

print("\n── Trader Dataset ──────────────────────────────────────────")
print(f"  Rows: {td_raw.shape[0]:,}   Cols: {td_raw.shape[1]}")
print(f"  Columns: {list(td_raw.columns)}")
print(f"  Missing values:\n{td_raw.isnull().sum().to_string()}")
print(f"  Duplicate rows: {td_raw.duplicated().sum()}")
print(f"  Date range: {pd.to_datetime(td_raw['time']).min()} → {pd.to_datetime(td_raw['time']).max()}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART A-2 — CLEAN & ALIGN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Cleaning & Merging ──────────────────────────────────────")

fg = fg_raw.copy()
fg["Date"] = pd.to_datetime(fg["Date"]).dt.normalize()
fg = fg.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)

td = td_raw.copy()
td["time"]    = pd.to_datetime(td["time"])
td["date"]    = td["time"].dt.normalize()
td["closedPnL"] = pd.to_numeric(td["closedPnL"], errors="coerce")
td["leverage"]  = pd.to_numeric(td["leverage"],  errors="coerce")
td["size"]      = pd.to_numeric(td["size"],      errors="coerce")
td = td.dropna(subset=["closedPnL","leverage","size"])

# Merge sentiment
td = td.merge(fg.rename(columns={"Date":"date","Classification":"sentiment"}),
              on="date", how="left")

unmapped = td["sentiment"].isna().sum()
print(f"  Trades with no matching sentiment date: {unmapped} ({unmapped/len(td)*100:.1f}%)")
td = td.dropna(subset=["sentiment"])

# Feature flags
td["is_long"]    = (td["side"] == "BUY").astype(int)
td["is_winner"]  = (td["closedPnL"] > 0).astype(int)
td["abs_pnl"]    = td["closedPnL"].abs()
td["lev_bucket"] = pd.cut(td["leverage"], bins=[0,5,15,100],
                           labels=["Low (≤5x)","Mid (5-15x)","High (>15x)"])

print(f"  Final merged rows: {len(td):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART A-3 — DAILY METRICS
# ═══════════════════════════════════════════════════════════════════════════════
daily = (td.groupby(["date","sentiment","account"])
           .agg(
               pnl        = ("closedPnL","sum"),
               trades     = ("closedPnL","count"),
               wins       = ("is_winner","sum"),
               avg_size   = ("size","mean"),
               avg_lev    = ("leverage","mean"),
               long_cnt   = ("is_long","sum"),
           ).reset_index())

daily["win_rate"]   = daily["wins"] / daily["trades"]
daily["long_ratio"] = daily["long_cnt"] / daily["trades"]

# Drawdown proxy per trader (rolling 7-day min of cumulative PnL)
td_sorted = td.sort_values(["account","date"])
td_sorted["cum_pnl"] = td_sorted.groupby("account")["closedPnL"].cumsum()
td_sorted["roll_min"] = (td_sorted.groupby("account")["cum_pnl"]
                           .transform(lambda x: x.rolling(7, min_periods=1).min()))
td_sorted["drawdown_proxy"] = td_sorted["cum_pnl"] - td_sorted["roll_min"]

daily_dd = (td_sorted.groupby(["date","sentiment","account"])
                     ["drawdown_proxy"].min().reset_index()
                     .rename(columns={"drawdown_proxy":"max_drawdown_proxy"}))
daily = daily.merge(daily_dd, on=["date","sentiment","account"], how="left")

# Trader-level summaries
trader = (daily.groupby("account")
               .agg(
                   total_pnl   = ("pnl","sum"),
                   avg_pnl     = ("pnl","mean"),
                   win_rate    = ("win_rate","mean"),
                   avg_trades  = ("trades","mean"),
                   avg_lev     = ("avg_lev","mean"),
                   avg_size    = ("avg_size","mean"),
                   avg_lr      = ("long_ratio","mean"),
               ).reset_index())

trader["freq_bucket"] = pd.qcut(trader["avg_trades"], q=3,
                                 labels=["Infrequent","Moderate","Frequent"])
trader["lev_bucket"]  = pd.cut(trader["avg_lev"], bins=[0,5,15,100],
                                labels=["Low (≤5x)","Mid (5-15x)","High (>15x)"])
trader["winner_bucket"] = pd.cut(trader["win_rate"], bins=[0,.4,.6,1.0],
                                  labels=["Inconsistent","Moderate","Consistent Winner"])

print("\nTrader summary sample:")
print(trader.describe().round(2).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# PART B — ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART B — ANALYSIS")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# B1 — Performance: Fear vs Greed
# ─────────────────────────────────────────────────────────────────────────────
print("\n── B1: Performance by Sentiment ────────────────────────────")
perf = daily.groupby("sentiment").agg(
    median_pnl  = ("pnl","median"),
    mean_pnl    = ("pnl","mean"),
    mean_wr     = ("win_rate","mean"),
    mean_dd     = ("max_drawdown_proxy","mean"),
    n_obs       = ("pnl","count"),
).round(3)
print(perf.to_string())

fear_pnl  = daily.loc[daily["sentiment"]=="Fear",  "pnl"]
greed_pnl = daily.loc[daily["sentiment"]=="Greed", "pnl"]
t, p = stats.ttest_ind(fear_pnl, greed_pnl)
print(f"\nWelch t-test PnL (Fear vs Greed): t={t:.3f}, p={p:.4f} {'*significant*' if p<0.05 else '(not significant)'}")

# CHART 1 — PnL distribution by sentiment
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Chart 1 — Trader PnL & Win-Rate by Market Sentiment",
             fontsize=14, color=TEXT, fontweight="bold", y=1.02)

# 1a Violin
ax = axes[0]
data_plot = [fear_pnl.clip(-3000,3000), greed_pnl.clip(-3000,3000)]
parts = ax.violinplot(data_plot, positions=[1,2], widths=0.6, showmedians=True)
parts["cmedians"].set_color(ACCENT)
for pc, c in zip(parts["bodies"], [FEAR_C, GREED_C]):
    pc.set_facecolor(c); pc.set_alpha(0.7)
ax.set_xticks([1,2]); ax.set_xticklabels(["Fear","Greed"], fontsize=12)
ax.set_ylabel("Daily PnL (USD)"); ax.set_title("PnL Distribution")
ax.axhline(0, color="#555", lw=1)

# 1b Win rate bars
ax = axes[1]
wr = daily.groupby("sentiment")["win_rate"].mean()
bars = ax.bar(wr.index, wr.values, color=[FEAR_C, GREED_C], width=0.4)
for bar, val in zip(bars, wr.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{val:.1%}", ha="center", color=TEXT, fontsize=11, fontweight="bold")
ax.set_ylim(0, 0.8); ax.set_ylabel("Win Rate"); ax.set_title("Average Win Rate")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

# 1c Drawdown proxy
ax = axes[2]
dd = daily.groupby("sentiment")["max_drawdown_proxy"].mean()
bars = ax.bar(dd.index, dd.abs().values, color=[FEAR_C, GREED_C], width=0.4, alpha=0.85)
for bar, val in zip(bars, dd.abs().values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
            f"${val:,.0f}", ha="center", color=TEXT, fontsize=11)
ax.set_ylabel("Avg Drawdown Proxy (USD)"); ax.set_title("Drawdown Proxy")

plt.tight_layout()
plt.savefig("charts/chart1_performance_by_sentiment.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
plt.close()
print("  → Saved charts/chart1_performance_by_sentiment.png")

# ─────────────────────────────────────────────────────────────────────────────
# B2 — Behavior: Fear vs Greed
# ─────────────────────────────────────────────────────────────────────────────
print("\n── B2: Behavior Change by Sentiment ───────────────────────")
behav = daily.groupby("sentiment").agg(
    avg_trades    = ("trades","mean"),
    avg_leverage  = ("avg_lev","mean"),
    long_ratio    = ("long_ratio","mean"),
    avg_size      = ("avg_size","mean"),
).round(3)
print(behav.to_string())

# CHART 2 — Behavior radar + scatter
fig = plt.figure(figsize=(16, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Chart 2 — Trading Behavior by Sentiment",
             fontsize=14, color=TEXT, fontweight="bold")
gs = gridspec.GridSpec(1, 3, figure=fig)

# 2a Trade frequency
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(CARD_BG)
tf = daily.groupby(["date","sentiment"])["trades"].sum().reset_index()
for sent, c in [("Fear", FEAR_C), ("Greed", GREED_C)]:
    sub = tf[tf["sentiment"]==sent]
    ax1.scatter(sub["date"], sub["trades"], color=c, alpha=0.3, s=10)
    roll = sub.set_index("date")["trades"].rolling(7).mean()
    ax1.plot(roll.index, roll.values, color=c, lw=2, label=sent)
ax1.legend(); ax1.set_title("Daily Trade Count (7d MA)")
ax1.set_xlabel(""); ax1.set_ylabel("Trades")
ax1.tick_params(axis="x", rotation=30)

# 2b Average leverage
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor(CARD_BG)
lv = daily.groupby("sentiment")["avg_lev"].mean()
bars = ax2.bar(lv.index, lv.values, color=[FEAR_C, GREED_C], width=0.4)
for bar, val in zip(bars, lv.values):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
             f"{val:.1f}x", ha="center", color=TEXT, fontweight="bold")
ax2.set_title("Avg Leverage"); ax2.set_ylabel("Leverage (x)")

# 2c Long/Short ratio
ax3 = fig.add_subplot(gs[2])
ax3.set_facecolor(CARD_BG)
lr = daily.groupby("sentiment")["long_ratio"].mean()
short_r = 1 - lr
x = np.arange(2)
ax3.bar(x, lr.values, label="Long", color=GREED_C, width=0.5)
ax3.bar(x, short_r.values, bottom=lr.values, label="Short", color=FEAR_C, width=0.5)
ax3.set_xticks(x); ax3.set_xticklabels(lr.index)
ax3.set_ylabel("Ratio"); ax3.set_title("Long / Short Bias")
ax3.legend()
ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

plt.tight_layout()
plt.savefig("charts/chart2_behavior_by_sentiment.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
plt.close()
print("  → Saved charts/chart2_behavior_by_sentiment.png")

# ─────────────────────────────────────────────────────────────────────────────
# B3 — Segments
# ─────────────────────────────────────────────────────────────────────────────
print("\n── B3: Segment Analysis ────────────────────────────────────")

# ── Segment 1: High vs Low Leverage ─────────────────────────────────────────
seg1 = td.merge(trader[["account","lev_bucket"]].rename(columns={"lev_bucket":"trader_lev_bucket"}), on="account", how="left")
seg1_perf = (seg1.groupby(["trader_lev_bucket","sentiment"])
               .agg(median_pnl=("closedPnL","median"),
                    win_rate  =("is_winner","mean"),
                    n         =("closedPnL","count"))
               .reset_index())
print("\nSegment 1 – Leverage bucket × Sentiment:")
print(seg1_perf.to_string(index=False))

# ── Segment 2: Trade Frequency ───────────────────────────────────────────────
seg2 = td.merge(trader[["account","freq_bucket"]], on="account", how="left")
seg2_perf = (seg2.groupby(["freq_bucket","sentiment"])
               .agg(median_pnl=("closedPnL","median"),
                    win_rate  =("is_winner","mean"))
               .reset_index())
print("\nSegment 2 – Frequency bucket × Sentiment:")
print(seg2_perf.to_string(index=False))

# ── Segment 3: Winner consistency ───────────────────────────────────────────
seg3 = td.merge(trader[["account","winner_bucket"]], on="account", how="left")
seg3_perf = (seg3.groupby(["winner_bucket","sentiment"])
               .agg(median_pnl=("closedPnL","median"),
                    avg_lev   =("leverage","mean"))
               .reset_index())
print("\nSegment 3 – Winner bucket × Sentiment:")
print(seg3_perf.to_string(index=False))

# CHART 3 — Segment heatmaps
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Chart 3 — Segment Performance (Median PnL) by Sentiment",
             fontsize=14, color=TEXT, fontweight="bold")

for ax, df, seg_col, title in [
    (axes[0], seg1_perf, "trader_lev_bucket", "Leverage Segment"),
    (axes[1], seg2_perf, "freq_bucket",   "Frequency Segment"),
    (axes[2], seg3_perf, "winner_bucket", "Winner Consistency"),
]:
    pivot = df.pivot(index=seg_col, columns="sentiment", values="median_pnl")
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", cmap="RdYlGn",
                center=0, linewidths=0.5, cbar_kws={"label":"Median PnL"})
    ax.set_title(title, color=TEXT); ax.set_xlabel(""); ax.set_ylabel("")

plt.tight_layout()
plt.savefig("charts/chart3_segment_heatmaps.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
plt.close()
print("  → Saved charts/chart3_segment_heatmaps.png")

# CHART 4 — Detailed segment breakdown
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Chart 4 — Leverage vs PnL & Win Rate by Segment",
             fontsize=14, color=TEXT, fontweight="bold")

# 4a Scatter leverage vs total PnL (trader level)
ax = axes[0]
ax.set_facecolor(CARD_BG)
colors = {"Low (≤5x)": GREED_C, "Mid (5-15x)": ACCENT, "High (>15x)": FEAR_C}
for lb in trader["lev_bucket"].dropna().unique():
    sub = trader[trader["lev_bucket"] == lb]
    ax.scatter(sub["avg_lev"], sub["total_pnl"].clip(-20000, 30000),
               label=lb, color=colors.get(str(lb), BLUE), alpha=0.6, s=40)
ax.axhline(0, color="#555", lw=1, ls="--")
ax.set_xlabel("Avg Leverage (x)"); ax.set_ylabel("Total PnL (USD)")
ax.set_title("Leverage vs Total PnL"); ax.legend()

# 4b Win-rate box by winner bucket
ax = axes[1]
ax.set_facecolor(CARD_BG)
buckets = ["Inconsistent","Moderate","Consistent Winner"]
data_box = [trader.loc[trader["winner_bucket"]==b, "win_rate"].dropna() for b in buckets]
bplot = ax.boxplot(data_box, patch_artist=True, widths=0.4,
                   medianprops=dict(color=ACCENT, lw=2))
c_list = [FEAR_C, BLUE, GREED_C]
for patch, c in zip(bplot["boxes"], c_list):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_xticklabels(buckets, rotation=10)
ax.set_ylabel("Win Rate"); ax.set_title("Win Rate Distribution by Consistency Segment")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

plt.tight_layout()
plt.savefig("charts/chart4_segment_detail.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
plt.close()
print("  → Saved charts/chart4_segment_detail.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — Sentiment timeline + cumulative PnL
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.patch.set_facecolor(BG)
fig.suptitle("Chart 5 — Market Sentiment & Aggregate Trader Metrics Over Time",
             fontsize=14, color=TEXT, fontweight="bold")

# 5a Sentiment strips
ax = axes[0]; ax.set_facecolor(CARD_BG)
for _, row in fg.iterrows():
    c = FEAR_C if row["Classification"]=="Fear" else GREED_C
    ax.axvspan(row["Date"], row["Date"]+pd.Timedelta(days=1), color=c, alpha=0.5)
ax.set_yticks([]); ax.set_ylabel("Sentiment", color=TEXT)

# 5b Daily agg PnL
ax = axes[1]; ax.set_facecolor(CARD_BG)
daily_agg = daily.groupby("date")["pnl"].median().reset_index()
daily_agg["roll"] = daily_agg["pnl"].rolling(7).mean()
ax.bar(daily_agg["date"], daily_agg["pnl"].clip(-5000,5000),
       color=[GREED_C if v>0 else FEAR_C for v in daily_agg["pnl"]], alpha=0.4, width=0.9)
ax.plot(daily_agg["date"], daily_agg["roll"], color=ACCENT, lw=2, label="7-day MA")
ax.axhline(0, color="#555", lw=1)
ax.set_ylabel("Median PnL (USD)"); ax.legend()

# 5c Trade volume
ax = axes[2]; ax.set_facecolor(CARD_BG)
vol = td.groupby("date")["size"].sum().reset_index()
ax.fill_between(vol["date"], vol["size"], color=BLUE, alpha=0.5)
ax.set_ylabel("Total Volume (USD)"); ax.set_xlabel("Date")

plt.tight_layout()
plt.savefig("charts/chart5_timeline.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("  → Saved charts/chart5_timeline.png")

# ═══════════════════════════════════════════════════════════════════════════════
# BONUS — CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BONUS — K-Means Clustering of Trader Archetypes")
print("=" * 70)

feat_cols = ["avg_pnl","win_rate","avg_trades","avg_lev","avg_size","avg_lr"]
X = trader[feat_cols].fillna(0)
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

inertias = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_sc)
    inertias.append(km.inertia_)

km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
trader["cluster"] = km4.fit_predict(X_sc)

cluster_summary = trader.groupby("cluster")[feat_cols].mean().round(3)
print("\nCluster centroids:")
print(cluster_summary.to_string())

# CHART 6 — Cluster scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Chart 6 — Behavioral Archetypes via K-Means Clustering",
             fontsize=14, color=TEXT, fontweight="bold")

cluster_labels = {0:"Passive / Low Risk", 1:"Aggressive / High Lev",
                  2:"Consistent Winners", 3:"Volatile / Inconsistent"}
cluster_colors = [BLUE, FEAR_C, GREED_C, ACCENT]

ax = axes[0]; ax.set_facecolor(CARD_BG)
for c in range(4):
    sub = trader[trader["cluster"]==c]
    ax.scatter(sub["avg_lev"], sub["total_pnl"].clip(-20000,40000),
               label=cluster_labels[c], color=cluster_colors[c], alpha=0.7, s=60)
ax.set_xlabel("Avg Leverage"); ax.set_ylabel("Total PnL")
ax.set_title("Leverage vs PnL (by cluster)"); ax.legend(fontsize=8)
ax.axhline(0, color="#555", lw=1, ls="--")

ax = axes[1]; ax.set_facecolor(CARD_BG)
for c in range(4):
    sub = trader[trader["cluster"]==c]
    ax.scatter(sub["avg_trades"], sub["win_rate"],
               label=cluster_labels[c], color=cluster_colors[c], alpha=0.7, s=60)
ax.set_xlabel("Avg Daily Trades"); ax.set_ylabel("Win Rate")
ax.set_title("Trade Frequency vs Win Rate (by cluster)"); ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

plt.tight_layout()
plt.savefig("charts/chart6_clusters.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("  → Saved charts/chart6_clusters.png")

# ═══════════════════════════════════════════════════════════════════════════════
# BONUS — PREDICTIVE MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BONUS — Predictive Model: Next-Day Profitability")
print("=" * 70)

# Build daily-trader feature table
model_df = daily.copy()
model_df = model_df.sort_values(["account","date"])
model_df["sent_enc"]   = (model_df["sentiment"]=="Greed").astype(int)
model_df["target"]     = (model_df["pnl"] > 0).astype(int)  # next-day profitable
# Lag features
for lag in [1, 3]:
    model_df[f"pnl_lag{lag}"]    = model_df.groupby("account")["pnl"].shift(lag)
    model_df[f"wr_lag{lag}"]     = model_df.groupby("account")["win_rate"].shift(lag)
    model_df[f"trades_lag{lag}"] = model_df.groupby("account")["trades"].shift(lag)

feat_model = ["sent_enc","avg_lev","avg_size","long_ratio","trades",
              "pnl_lag1","wr_lag1","pnl_lag3","wr_lag3","trades_lag3"]
model_df_clean = model_df.dropna(subset=feat_model + ["target"])

X_m = model_df_clean[feat_model]
y_m = model_df_clean["target"]

X_tr, X_te, y_tr, y_te = train_test_split(X_m, y_m, test_size=0.2, random_state=42,
                                            stratify=y_m)
clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  random_state=42)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
y_prob = clf.predict_proba(X_te)[:,1]

print(f"\nTest AUC-ROC: {roc_auc_score(y_te, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_te, y_pred))

# Feature importance
fi = pd.Series(clf.feature_importances_, index=feat_model).sort_values(ascending=False)
print("\nTop Feature Importances:")
print(fi.round(4).to_string())

# CHART 7 — Feature importance
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG); ax.set_facecolor(CARD_BG)
fi.plot(kind="bar", ax=ax, color=[GREED_C if i==0 else BLUE for i in range(len(fi))])
ax.set_title("Chart 7 — Feature Importances: Next-Day Profitability Model",
             fontsize=13, color=TEXT, fontweight="bold")
ax.set_ylabel("Importance"); ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("charts/chart7_feature_importance.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
plt.close()
print("  → Saved charts/chart7_feature_importance.png")

print("\n" + "=" * 70)
print("All charts saved to charts/")
print("=" * 70)
