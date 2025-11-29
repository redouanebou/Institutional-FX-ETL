import pandas as pd
import numpy as np
import os
import gc

class HybridDataMerger:
    def __init__(self, m1_path, tick_path, output_path, spike_threshold=0.005, chunk_size=10_000_000, smooth_spread=True, normalize_volume=False):
        self.m1_path = m1_path
        self.tick_path = tick_path
        self.output_path = output_path
        self.audit_path = output_path.replace('.csv', '_AUDIT.csv')
        self.spike_threshold = spike_threshold
        self.chunk_size = chunk_size
        self.smooth_spread = smooth_spread
        self.normalize_volume = normalize_volume
        self.audit_log = []

    def log_events(self, df_subset, reason):
        """
        Logs a batch of rows to the audit trail.
        """
        if df_subset.empty:
            return
            
        logs = df_subset.copy()
        logs['reason'] = reason
        
        if 'datetime' not in logs.columns and isinstance(logs.index, pd.DatetimeIndex):
            logs = logs.reset_index()
            
        cols_to_log = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'reason']
        for col in cols_to_log:
            if col not in logs.columns:
                logs[col] = np.nan
        
        self.audit_log.append(logs[cols_to_log])

    def save_audit_log(self):
        if not self.audit_log:
            print("📝 Audit Log: Clean run. No anomalies.")
            return

        full_log = pd.concat(self.audit_log, ignore_index=True)
        if 'datetime' in full_log.columns:
             full_log['datetime'] = pd.to_datetime(full_log['datetime'])
             
        full_log.to_csv(self.audit_path, index=False)
        print(f"📝 Audit Log Saved: {self.audit_path} ({len(full_log)} records)")

    def load_m1_history(self):
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
        print(f"💎 Processing High-Res Ticks (Chunked): {self.tick_path}")
        chunk_iter = pd.read_csv(
            self.tick_path,
            names=['datetime', 'ask', 'bid', 'av', 'bv'],
            header=0,
            usecols=['datetime', 'ask', 'bid'],
            dtype={'ask': 'float32', 'bid': 'float32'},
            chunksize=self.chunk_size
        )

        resampled_chunks = []
        total_rows_processed = 0

        try:
            for i, chunk in enumerate(chunk_iter):
                chunk['datetime'] = pd.to_datetime(chunk['datetime'], utc=True)
                chunk.set_index('datetime', inplace=True)
                chunk.sort_index(inplace=True)

                chunk['mid'] = (chunk['ask'] + chunk['bid']) / 2
                chunk['spread'] = chunk['ask'] - chunk['bid']

                ohlc_dict = {
                    'mid': 'ohlc',     
                    'spread': 'mean',  
                    'ask': 'count'     
                }
                
                chunk_agg = chunk.resample('1min', label='left', closed='left').agg(ohlc_dict)
                
                chunk_agg.columns = ['_'.join(col).strip() for col in chunk_agg.columns.values]
                chunk_agg.rename(columns={
                    'mid_open': 'open', 'mid_high': 'high', 'mid_low': 'low', 'mid_close': 'close',
                    'spread_mean': 'spread', 'ask_count': 'volume'
                }, inplace=True)

                resampled_chunks.append(chunk_agg)
                count = len(chunk)
                total_rows_processed += count
                print(f"   Processed Chunk {i+1} ({count} ticks) ...")
                
                chunk_stat = pd.DataFrame([{
                    'datetime': chunk.index.min(), 
                    'volume': count, 
                    'reason': f"chunk_{i+1}_processed"
                }])
                self.log_events(chunk_stat, f"chunk_{i+1}_processed")

                del chunk
                gc.collect()

            print("   🔨 Consolidating Chunks...")
            if not resampled_chunks:
                return pd.DataFrame()

            full_agg = pd.concat(resampled_chunks)
            full_agg.sort_index(inplace=True)

            final_ohlc = full_agg.groupby(level=0).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'spread': 'mean'
            })

            final_ohlc.dropna(subset=['close'], inplace=True)
            
            print(f"   🔹 Generated {len(final_ohlc)} Superior Candles from {total_rows_processed} Ticks")
            return final_ohlc

        except Exception as e:
            print(f"❌ Tick Chunk Error: {e}")
            return pd.DataFrame()

    def validate_and_clean(self, df):
        print("🔍 Running Deep Validation & Cleaning...")
        
        inv_mask = df['low'] > df['high']
        if inv_mask.any():
            print(f"   ❌ Found {inv_mask.sum()} inverted candles. Dropping...")
            self.log_events(df[inv_mask], "dropped_inverted")
            df = df[~inv_mask]
            
        bad_oc = (df['open'] > df['high']) | (df['open'] < df['low']) | \
                 (df['close'] > df['high']) | (df['close'] < df['low'])
        if bad_oc.any():
             print(f"   ⚠️ Fixing {bad_oc.sum()} candles with Open/Close outside High/Low bounds.")
             self.log_events(df[bad_oc], "fixed_structure_integrity")
             df.loc[bad_oc, 'high'] = df.loc[bad_oc, ['open', 'close', 'high']].max(axis=1)
             df.loc[bad_oc, 'low'] = df.loc[bad_oc, ['open', 'close', 'low']].min(axis=1)

        amp = (df['high'] - df['low']) / df['open']
        spike_mask = amp > self.spike_threshold
        
        jump_mask = df['close'].pct_change().abs() > self.spike_threshold
        
        total_bad = spike_mask | jump_mask
        
        if total_bad.any():
            print(f"   ❌ Dropping {total_bad.sum()} volatility anomalies (> {self.spike_threshold*100}%).")
            self.log_events(df[total_bad], "dropped_volatility_spike")
            df = df[~total_bad]
            
        return df

    def run(self):
        df_m1 = self.load_m1_history()
        
        df_ticks = self.process_ticks_chunked()

        if df_m1.empty and df_ticks.empty:
            print("❌ No data loaded.")
            return

        print("🔗 Merging Datasets (Overwrite M1 with Ticks)...")
        df_final = df_ticks.combine_first(df_m1)
        
        if not df_ticks.empty:
            median_spread = df_ticks['spread'].median()
            nan_spread_count = df_final['spread'].isna().sum()
            if nan_spread_count > 0:
                print(f"   💡 Filling {nan_spread_count} missing spreads in M1 history with median tick spread: {median_spread:.5f}")
                df_final['spread'].fillna(median_spread, inplace=True)
        else:
            df_final['spread'].fillna(0, inplace=True)

        if self.smooth_spread:
            print("   〰️ Smoothing Spread (Rolling Median)...")
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
            print(f"   ⚠️ Bridging {gaps} gaps.")
            
            gap_rows = df_final[mask_nan].copy()
            self.log_events(gap_rows, "gap_bridged_flat")
            
            df_final['close'] = df_final['close'].ffill()
            df_final.loc[mask_nan, 'open'] = df_final.loc[mask_nan, 'close']
            df_final.loc[mask_nan, 'high'] = df_final.loc[mask_nan, 'close']
            df_final.loc[mask_nan, 'low']  = df_final.loc[mask_nan, 'close']
            df_final.loc[mask_nan, 'volume'] = 0 
            df_final.loc[mask_nan, 'is_flat'] = 1
            
            df_final['spread'] = df_final['spread'].ffill()

        df_final.dropna(subset=['close'], inplace=True)

        if self.normalize_volume:
            print("   ⚖️ Normalizing Volume (0-1 Scale)...")
            vol_min = df_final['volume'].min()
            vol_max = df_final['volume'].max()
            if vol_max > vol_min:
                df_final['volume'] = (df_final['volume'] - vol_min) / (vol_max - vol_min)

        cols = ['open', 'high', 'low', 'close', 'volume', 'spread', 'is_flat']
        df_final = df_final[cols].astype('float32')
        df_final['is_flat'] = df_final['is_flat'].astype('int8')

        print(f"💾 Saving Hybrid Dataset: {self.output_path}")
        df_final.to_csv(self.output_path)
        
        self.save_audit_log()
        
        print(f"✅ DONE. Total Rows: {len(df_final)}")

if __name__ == "__main__":
    M1_FILE = r"D:\duka\GBPUSD.csv"
    TICK_FILE = r"D:\duka\GBPUSD1.csv"
    OUTPUT_FILE = r"D:\duka\GBPUSD_HYBRID_CLEAN.csv"
    
    merger = HybridDataMerger(
        M1_FILE, 
        TICK_FILE, 
        OUTPUT_FILE, 
        chunk_size=50_000_000,
        smooth_spread=True,      
        normalize_volume=False  
    )
    merger.run()