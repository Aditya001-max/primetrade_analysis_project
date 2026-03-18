# Primetrade.ai — Trader Performance vs Market Sentiment
### Round-0 Data Science Intern Assignment

---

## Overview
This project analyses how Bitcoin market sentiment (Fear/Greed Index) relates to trader behaviour and performance on the Hyperliquid DEX. It covers full data preparation, statistical analysis, segmentation, K-Means clustering, and a predictive profitability model.

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl
```

---

## How to Run

### Jupyter Notebook
```bash
jupyter notebook notebook.ipynb
```
---

## Project Structure

```
primetrade_analysis/
├── notebook.ipynb          ← Main analysis notebook
├── analysis.py             ← Equivalent Python script
├── generate_data.py        ← Synthetic data generator (schema mirror)
├── data/
│   ├── fear_greed.csv
│   └── trader_data.csv
└── charts/
    ├── chart1_performance_by_sentiment.png
    ├── chart2_behavior_by_sentiment.png
    ├── chart3_segment_heatmaps.png
    ├── chart4_segment_detail.png
    ├── chart5_timeline.png
    ├── chart6_clusters.png
    └── chart7_feature_importance.png
```

---

## Methodology

### Data Preparation
- Loaded Fear/Greed CSV (366 rows × 2 cols) and Trader CSV (55,000 rows × 11 cols)
- Parsed timestamps, normalised to daily granularity, and merged on `date`
- Computed daily per-trader metrics: PnL, win rate, trade count, leverage, long/short ratio
- Built a 7-day rolling drawdown proxy from cumulative PnL

### Analysis
- **B1 (Performance):** Welch t-test on Fear vs Greed PnL distributions; violin plots, win-rate bars, drawdown comparison
- **B2 (Behaviour):** Long/short ratio, leverage, and trade frequency split by sentiment
- **B3 (Segments):** Three orthogonal cuts — leverage tier, trade frequency, winner consistency — each crossed with sentiment; heatmaps showing median PnL per cell

### Clustering
K-Means (k=4) on 6 standardised trader-level features identifies four archetypes:
| Cluster | Label | Characteristics |
|---|---|---|
| 0 | Passive / Low Risk | Low leverage, moderate PnL |
| 1 | Aggressive / High Leverage | >18x avg leverage, negative total PnL |
| 2 | Consistent Winners | Highest win rate, moderate leverage |
| 3 | Volatile / Inconsistent | Large position sizes, erratic PnL |

### Predictive Model used
Gradient Boosting classifier predicts next-day trader profitability (binary: profit / loss) using:
- Sentiment encoding, leverage, position size, long ratio, trade count
- 1-day and 3-day lagged PnL and win-rate features

**Test AUC-ROC: 0.58** (meaningful signal above 0.5 baseline given noisy financial data)

Top features: `avg_lev` (28%), `sent_enc` (23%), `pnl_lag1` (15%), `avg_size` (14%)

---

## Key Insights 

| # | Insight | Evidence |
|---|---|---|
| 1 | **Extreme Greed days have the highest win rate (38.6%) but also the largest drawdown ($395)** — traders win more often but also over-extend | Chart 1 |
| 2 | **Leverage is nearly identical across Fear (7.5x) and Greed (8.0x)** — traders don't adjust leverage based on sentiment | Chart 2 |
| 3 | **Long/short bias barely shifts (~52% long on Fear, ~50% on Greed)** — directional bias is more stable than expected | Chart 2 |
| 4 | **Consistent Winners show positive PnL across all sentiment regimes** (up to $25.8 median on Extreme Fear) — skill dominates sentiment | Chart 3 |
| 5 | **long_ratio is the #1 predictor of next-day profitability (52% importance)** — your directional positioning matters far more than sentiment | Chart 7 |

---

## Strategy Recommendations 

### Strategy 1 — Don't Chase Extreme Greed with Large Positions
> When sentiment = Extreme Greed → reduce position size by 20–30%

**Why:** Extreme Greed days have the highest win rate (38.6%) but also the
largest drawdown proxy ($395) — nearly 2.5x higher than Neutral days ($139).
Traders win more often but lose much bigger when they're wrong. The risk/reward
deteriorates at sentiment extremes.

---

### Strategy 2 — Prioritise Directional Accuracy Over Sentiment Timing
> Focus on getting long_ratio right rather than trading the Fear/Greed index

**Why:** The predictive model shows long_ratio accounts for 52% of next-day
profitability prediction, while sentiment encoding (sent_enc) contributes less
than 1%. This means whether you are positioned long or short matters far more
than what the Fear/Greed index reads. Traders should use sentiment as a
secondary filter, not a primary signal.

---

### Strategy 3 — Emulate Consistent Winners on Extreme Fear Days
> During Extreme Fear, copy the behaviour of Consistent Winner segment

**Why:** The segment heatmap (Chart 3) shows Consistent Winners earn a median
PnL of $25.8 even on Extreme Fear days — the highest across all sentiment
regimes. While other segments go flat or negative, this group likely uses
tighter stops, smaller sizes, and avoids over-leveraging. Their avg leverage
stays in the Mid (5–15x) band, never touching High (>15x).

---

