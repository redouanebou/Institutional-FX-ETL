<div align="center">

# 🧬 Institutional FX Data Pipeline
### Hybrid Tick/OHLC Fusion Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Vectorized-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production%20Grade-success?style=for-the-badge)

<p align="center">
  <em>An industrial-grade ETL solution to eliminate Look-Ahead Bias and reconstruct incomplete Forex history.</em>
</p>

</div>

---

## 📉 The Problem: "Dirty Data" in Financial Markets

In algorithmic trading, **Data Quality > Model Complexity**. 

While preparing a 15-year dataset for a Deep Learning model (GBPUSD), I performed a forensic audit on standard M1 OHLC history provided by brokers. The findings were **catastrophic**:

| Metric | Findings | Impact |
| :--- | :--- | :--- |
| **Missing Data** | `2,383,999` minutes gaps | Model fails to learn time-decay |
| **Data Loss** | **29.03%** of timeline | Blind spots in backtesting |
| **No Spread** | Missing Bid/Ask spread | Unrealistic PnL simulations |

> ⚠️ **Critical Risk:** Training on this data guarantees **Regime Hallucination** and **Look-Ahead Bias**.

---

## 🛠️ The Solution: Hybrid ETL Pipeline

I engineered `HybridDataMerger`, a custom ETL engine that prioritizes high-resolution **Tick Data** and fuses it with legacy history to create a statistically immutable time grid.

### 🔄 Pipeline Architecture

```mermaid
graph LR
    A[Legacy M1 Data] -->|Load & Audit| C(Hybrid Merger Engine)
    B[High-Res Tick Data] -->|Resample & Spread Calc| C
    C -->|Priority Overwrite| D{Gap Detection}
    D -->|Found Gap| E[Bridge with Flat Candle]
    D -->|Continuous| F[Validate Integrity]
    E --> F
    F -->|Output| G[clean_dataset.csv]
