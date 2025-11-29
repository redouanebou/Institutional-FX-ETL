\# Institutional FX Data Pipeline: Hybrid Tick/OHLC Fusion 🧬



!\[Python](https://img.shields.io/badge/Python-3.9%2B-blue)

!\[Data Integrity](https://img.shields.io/badge/Data%20Integrity-Critical-red)

!\[Performance](https://img.shields.io/badge/Performance-Vectorized-green)



\## 📉 The Problem: "Dirty Data" in Financial Markets



In algorithmic trading, \*\*Data Quality > Model Complexity\*\*. 

While preparing a 15-year dataset for a Deep Learning model (GBPUSD), I performed a forensic audit on the available M1 OHLC history provided by standard brokers.



The findings were catastrophic:

\* \*\*Missing Data:\*\* The standard dataset contained \*\*2,383,999 missing minutes\*\* (Gaps).

\* \*\*Data Loss:\*\* \*\*29.03%\*\* of the timeline was non-existent.

\* \*\*No Spread:\*\* Historical M1 data lacked Bid/Ask spread information, making realistic backtesting impossible.



Training a Machine Learning model on this dataset would result in \*\*Look-Ahead Bias\*\* and \*\*Regime Hallucination\*\* due to the broken time grid.



\## 🛠️ The Solution: Hybrid ETL Pipeline



I engineered a custom ETL (Extract, Transform, Load) engine `HybridDataMerger` that fuses legacy M1 history with high-resolution Tick Data to reconstruct a statistically valid timeline.



\### Key Features



1\.  \*\*Tick-Derived Volume Reconstruction:\*\* \* \*Challenge:\* Raw tick data often contains `Volume=0` in standard aggregated feeds.

&nbsp;   \* \*Solution:\* The pipeline ignores the metadata volume and calculates \*\*True Tick Volume\*\* by counting the actual `Ask` updates per minute. This transforms "dead" columns into high-signal volatility features.



2\.  \*\*Hybrid Fusion (The "Overwrite" Strategy):\*\*

&nbsp;   \* The engine prioritizes high-precision Tick Data (with Spread \& Microstructure info).

&nbsp;   \* It falls back to legacy M1 data \*only\* when ticks are unavailable (older history).

&nbsp;   \* Result: A seamless 15-year dataset where the recent years have institutional-grade precision.



3\.  \*\*Immutable Time Grid (Gap Bridging):\*\*

&nbsp;   \* To satisfy the input requirements of Time-Series models (Transformers/LSTMs), the pipeline enforces a continuous time grid.

&nbsp;   \* It identifies the ~2.3 million gaps and creates "Flat Candles" (Forward Filled) with `Volume=0` and a specific `is\_flat=1` flag.

&nbsp;   \* This allows the ML model to explicitly learn "Market Inactivity" rather than seeing disjointed time jumps.



\## 📊 Performance Metrics



The pipeline processes massive tick datasets using \*\*Chunking (50M rows/chunk)\*\* and \*\*Vectorized Pandas Operations\*\* to manage memory efficiency.



\*\*Audit Results (GBPUSD 2010-2025):\*\*

| Metric | Value |

| :--- | :--- |

| \*\*Total Rows Generated\*\* | \*\*8,213,517\*\* |

| \*\*Gaps Detected \& Bridged\*\* | \*\*2,383,999 (29.03%)\*\* |

| \*\*Volatility Anomalies Dropped\*\* | \*\*243\*\* |

| \*\*Spread Recovery\*\* | \*\*100%\*\* (via median imputation for legacy data) |



\## 💻 Usage



```python

from src.hybrid\_merger import HybridDataMerger



\# Initialize the pipeline

merger = HybridDataMerger(

&nbsp;   m1\_path="data/raw/GBPUSD.csv",       # Legacy History

&nbsp;   tick\_path="data/raw/GBPUSD\_Ticks.csv", # High-Res Ticks

&nbsp;   output\_path="data/processed/GBPUSD\_Hybrid.csv",

&nbsp;   chunk\_size=50\_000\_000

)



\# Run the ETL Process

merger.run()

