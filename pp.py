import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ==========================
# 1. 데이터 다운로드
# ==========================
stocks = yf.download("^GSPC", start="2015-01-01", end="2025-01-01")  # S&P500
bonds = yf.download("^TNX", start="2015-01-01", end="2025-01-01")   # 10Y Treasury Yield

# ==========================
# 2. 데이터 정리
# ==========================
df = pd.DataFrame()

df["stock"] = stocks["Close"]
df["bond"] = bonds["Close"]

df = df.dropna()

# ==========================
# 3. 정규화 (비교용)
# ==========================
df["stock_norm"] = df["stock"] / df["stock"].iloc[0] * 100
df["bond_norm"] = df["bond"] / df["bond"].iloc[0] * 100

# ==========================
# 4. 차트
# ==========================
plt.figure(figsize=(12,6))

plt.plot(df.index, df["stock_norm"], label="S&P 500 (Normalized)")
plt.plot(df.index, df["bond_norm"], label="10Y Treasury Yield (Normalized)")

plt.title("Stocks vs Bonds Performance")
plt.xlabel("Date")
plt.ylabel("Normalized Value")
plt.legend()
plt.grid(True)

plt.show()