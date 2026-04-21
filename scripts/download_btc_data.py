import pathlib
import yfinance as yf

OUTPUT_PATH = pathlib.Path("data/raw/BTC.csv")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = yf.download("BTC-USD", start="2014-01-01", auto_adjust=True, progress=False)

df = df[["Open", "High", "Low", "Close", "Volume"]]
df.columns = ["open", "high", "low", "close", "volume"]

df.index.name = "date"
df = df.reset_index()

df = df.drop_duplicates(subset="date")
df = df.dropna()
df = df.sort_values("date", ascending=True).reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
print(f"Date range: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
