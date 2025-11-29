import pandas as pd

HYBRID_FILE = r"D:\duka\GBPUSD_PLATINUM_FINAL.csv"    # edit the path based on what you have
M1_FILE     = r"D:\duka\GBPUSD.csv"  # edit the path based on what you have

print("🕵️ HUNTING THE CATASTROPHE...")  

df_new = pd.read_csv(HYBRID_FILE, parse_dates=['datetime'], index_col='datetime')

df_old = pd.read_csv(M1_FILE, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Vol'], header=0)
df_old['datetime'] = pd.to_datetime(df_old['Date'] + ' ' + df_old['Time'], format='%Y.%m.%d %H:%M:%S', utc=True)
df_old.set_index('datetime', inplace=True)

common = df_new.index.intersection(df_old.index)
df_new_c = df_new.loc[common]
df_old_c = df_old.loc[common]

diff = (df_new_c['close'] - df_old_c['Close']).abs()
bad_rows = diff[diff > 0.01] 

print(f"🚩 Found {len(bad_rows)} catastrophic mismatches (> 100 pips).")

if not bad_rows.empty:
    print("\nTOP 5 OFFENDERS:")
    for dt in bad_rows.head(5).index:
        print(f"Time: {dt}")
        print(f"   M1 Close:   {df_old_c.loc[dt, 'Close']:.5f}")
        print(f"   Tick Close: {df_new_c.loc[dt, 'close']:.5f}")
        print(f"   Diff:       {diff.loc[dt]:.5f}")
        print("-" * 30)
