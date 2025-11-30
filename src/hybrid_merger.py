import pandas as pd
import numpy as np
import os
import gc
import sys

class HybridDataMerger:
    def __init__(self, m1_path, tick_path, output_path, spike_threshold=0.005, chunk_size=50_000_000, smooth_spread=True):
        self.m1_path = m1_path
        self.tick_path = tick_path
        self.output_path = output_path
        self.audit_path = output_path.replace('.csv', '_AUDIT.csv')
        self.spike_threshold = spike_threshold
        self.chunk_size = chunk_size 
        self.smooth_spread = smooth_spread
        self.audit_log = []

    def log_events(self, df_subset, reason):
        """Logs anomaly events for forensic audit."""
        if df_subset.empty: return
        logs = df_subset.copy()
        logs['reason'] = reason
        if 'datetime' not in logs.columns and isinstance(logs.index, pd.DatetimeIndex):
            logs = logs.reset_index()
        cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'reason']
        for c in cols:
            if c not in logs.columns: logs[c] = np.nan
        self.audit_log.append(logs[cols])

    def save_audit_log(self):
        """Exports the audit log to CSV."""
        if not self.audit_log: return
        full_log = pd.concat(self.audit_log, ignore_index=True)
        if 'datetime' in full_log.columns: full_log['datetime'] = pd.to_datetime(full_log['datetime'])
        full_log.to_csv(self.audit_path, index=False)
        print(f"📝 Audit Log Saved: {self.audit_path}")

    def load_m1_history(self):
        """Loads legacy M1 data to serve as the skeletal framework."""
        print(f"📖 Loading Base M1 History: {self.m1_path}")
        try:
            df = pd.read_csv(
                self.m1_path,
                names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol'],
                header=0, 
                dtype={'Open': 'float32', 'High': 'float32', 'Low': 'float32', 'Close': 'float32', 'TickVol': 'float32'}
            )
            df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S', utc=True)
            df.set_index('datetime', inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'TickVol']]
            df.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'TickVol':'volume'}, inplace=True)
            df['spread'] = np.nan
            print(f"   🔹 Base M1 Rows: {len(df)}")
            return df
        except Exception as e:
            print(f"❌ M1 Load Error: {e}")
            return pd.DataFrame()

    def process_ticks_chunked(self):
        """Processes high-res ticks in chunks to prevent RAM overflow."""
        print(f"💎 Processing High-Res Ticks (Chunked): {self.tick_path}")
        if not os.path.exists(self.tick_path):
            print("❌ Tick file not found!")
            return pd.DataFrame()

        chunk_iter = pd.read_csv(
            self.tick_path,
            names=['datetime', 'ask', 'bid', 'av', 'bv'],
            header=0,
            usecols=['datetime', 'ask', 'bid'],
            dtype={'ask': 'float32', 'bid': 'float32'},
            chunksize=self.chunk_size,
            on_bad_lines='skip' 
        )

        resampled_chunks = []
        total_rows_processed = 0

        try:
            for i, chunk in enumerate(chunk_iter):
                chunk['datetime'] = pd.to_datetime(chunk['datetime'], utc=True)
                chunk.set_index('datetime', inplace=True)
                chunk.sort_index(inplace=True)
                
                bad_ticks = (chunk['ask'] < 0.0001) | (chunk['bid'] < 0.0001)
                if bad_ticks.any():
                    chunk = chunk[~bad_ticks]

                mask_weekend = (chunk.index.dayofweek == 5) | \
                               ((chunk.index.dayofweek == 4) & (chunk.index.hour >= 22)) | \
                               ((chunk.index.dayofweek == 6) & (chunk.index.hour < 21))
                
                chunk = chunk[~mask_weekend]

                if chunk.empty: continue

                chunk['mid'] = (chunk['ask'] + chunk['bid']) / 2
                chunk['spread'] = chunk['ask'] - chunk['bid']

                resampler = chunk.resample('1min', label='left', closed='left')
                mid_ohlc = resampler['mid'].ohlc() 
                spread_mean = resampler['spread'].mean()
                tick_vol = resampler['ask'].count()
                
                chunk_agg = pd.DataFrame({
                    'open': mid_ohlc['open'],
                    'high': mid_ohlc['high'],
                    'low': mid_ohlc['low'],
                    'close': mid_ohlc['close'],
                    'volume': tick_vol,
                    'spread': spread_mean
                })
                
                if chunk_agg.empty: continue
                chunk_agg = chunk_agg.dropna(subset=['close'])
                
                resampled_chunks.append(chunk_agg)
                count = len(chunk)
                total_rows_processed += count
                print(f"   Processed Chunk {i+1} ({count} valid ticks) ...")
                del chunk
                gc.collect()

            print("   🔨 Consolidating Chunks...")
            if not resampled_chunks: return pd.DataFrame()

            full_agg = pd.concat(resampled_chunks)
            full_agg.sort_index(inplace=True)

            final_ohlc = full_agg.groupby(level=0).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                'volume': 'sum', 'spread': 'mean'
            })
            final_ohlc = final_ohlc.dropna(subset=['close'])
            
            print(f"   🔹 Generated {len(final_ohlc)} Superior Candles from {total_rows_processed} Ticks")
            return final_ohlc

        except Exception as e:
            print(f"❌ Tick Chunk Error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def validate_and_clean(self, df):
        """Runs integrity checks: Inverted candles, Structural errors, Volatility spikes."""
        print("🔍 Running Deep Validation & Cleaning...")
        
        inv_mask = df['low'] > df['high']
        if inv_mask.any():
            self.log_events(df[inv_mask], "dropped_inverted")
            df = df[~inv_mask]
            
        bad_oc = (df['open'] > df['high']) | (df['open'] < df['low']) | \
                 (df['close'] > df['high']) | (df['close'] < df['low'])
        if bad_oc.any():
             self.log_events(df[bad_oc], "fixed_structure_integrity")
             df.loc[bad_oc, 'high'] = df.loc[bad_oc, ['open', 'close', 'high']].max(axis=1)
             df.loc[bad_oc, 'low'] = df.loc[bad_oc, ['open', 'close', 'low']].min(axis=1)
             df.loc[bad_oc, 'spread'] = np.nan 
             df['spread'] = df['spread'].ffill()

        amp = (df['high'] - df['low']) / df['open']
        spike_mask = amp > self.spike_threshold
        jump_mask = df['close'].pct_change().abs() > self.spike_threshold
        total_bad = spike_mask | jump_mask
        
        if total_bad.any():
            print(f"   ❌ Dropping {total_bad.sum()} volatility anomalies.")
            self.log_events(df[total_bad], "dropped_volatility_spike")
            df = df[~total_bad]
            
        return df

    def run(self):
        """Executes the pipeline with Consensus Protocol."""
        df_m1 = self.load_m1_history()
        df_ticks = self.process_ticks_chunked()

        if df_ticks.empty:
            print("🛑 FATAL: No valid tick data found. Aborting.")
            sys.exit(1)

        print("🛡️ Running Consensus Check (M1 vs Ticks)...")
        
        m1_aligned = df_m1.reindex(df_ticks.index)
        
        diff = (df_ticks['close'] - m1_aligned['close']).abs()
        
        catastrophes = diff[diff > 0.01]
        
        if not catastrophes.empty:
            print(f"   🚩 Detected {len(catastrophes)} catastrophic mismatches (> 100 pips).")
            print("   👉 REJECTING Tick Data for these minutes. Trusting M1 Consensus.")
            self.log_events(df_ticks.loc[catastrophes.index], "rejected_tick_catastrophe")
            
            df_ticks = df_ticks.drop(catastrophes.index)
        else:
            print("   ✅ Consensus Check Passed. No catastrophic errors found.")

        print("🔗 Merging Datasets...")
        df_final = df_ticks.combine_first(df_m1)
        
        if not df_ticks.empty:
            median_spread = df_ticks['spread'].median()
            df_final['spread'] = df_final['spread'].fillna(median_spread)
        else:
            df_final['spread'] = df_final['spread'].fillna(0)

        if self.smooth_spread:
            df_final['spread'] = df_final['spread'].rolling(window=5, center=True, min_periods=1).median()
            
        df_final = self.validate_and_clean(df_final)
        
        print("🔧 Bridging Gaps (Immutable Grid)...")
        df_final.sort_index(inplace=True)
        full_idx = pd.date_range(start=df_final.index.min(), end=df_final.index.max(), freq='1min', name='datetime')
        df_final = df_final.reindex(full_idx)

        mask_nan = df_final['close'].isna()
        gaps = mask_nan.sum()
        
        df_final['is_flat'] = 0
        df_final['is_flat'] = df_final['is_flat'].astype('int8')
        
        if gaps > 0:
            print(f"   ⚠️ Bridging {gaps} gaps with Flat Candles.")
            gap_rows = df_final[mask_nan].copy()
            self.log_events(gap_rows, "gap_bridged_flat")
            
            df_final['close'] = df_final['close'].ffill()
            df_final.loc[mask_nan, 'open'] = df_final.loc[mask_nan, 'close']
            df_final.loc[mask_nan, 'high'] = df_final.loc[mask_nan, 'close']
            df_final.loc[mask_nan, 'low']  = df_final.loc[mask_nan, 'close']
            
            df_final.loc[mask_nan, 'volume'] = 0 
            df_final.loc[mask_nan, 'is_flat'] = 1
            
            df_final.loc[mask_nan, 'spread'] = df_final['spread'].ffill()

        df_final = df_final.dropna(subset=['close'])

        print("   🔒 Final Type Casting...")
        cols = ['open', 'high', 'low', 'close', 'spread', 'volume']
        for c in cols: df_final[c] = df_final[c].astype('float32')
        df_final['is_flat'] = df_final['is_flat'].astype('int8')

        print(f"💾 Saving Institutional Dataset: {self.output_path}")
        df_final.to_csv(self.output_path)
        self.save_audit_log()
        print(f"✅@redouane_boundra /  DONE. Total Rows: {len(df_final)}")

if __name__ == "__main__":
    M1_FILE = r"D:\duka\GBPUSD.csv"
    TICK_FILE = r"D:\duka\GBPUSD1.csv"
    OUTPUT_FILE = r"D:\duka\GBPUSD_PLATINUM_CLEAN.csv"
    
    merger = HybridDataMerger(M1_FILE, TICK_FILE, OUTPUT_FILE, smooth_spread=True)
    merger.run()
