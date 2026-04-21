import pathlib
import yfinance as yf

COINS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "BNB": "BNB-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "DOGE": "DOGE-USD",
}

OUTPUT_DIR = pathlib.Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for name, ticker in COINS.items():
    df = yf.download(ticker, start="2020-01-01", auto_adjust=True, progress=False)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df = df.reset_index()
    df = df.drop_duplicates(subset="date").dropna()
    df = df.sort_values("date").reset_index(drop=True)
    path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"{name}: {len(df)} rows  ({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")
