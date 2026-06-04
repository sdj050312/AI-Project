import yfinance as yf
from fredapi import Fred
import numpy as np
import os
from dotenv import load_dotenv
import matplotlib
import matplotlib.pyplot as plt
import io
import base64

from sklearn.ensemble import RandomForestRegressor

# =========================
# Matplotlib (Flask 필수 설정)
# =========================
matplotlib.use("Agg")

# =========================
# ENV + FRED API
# =========================
load_dotenv()

fred = Fred(api_key=os.getenv("FRED_API_KEY"))

# =========================
# SAFE FUNCTION
# =========================
def safe_last(series):
    series = series.dropna()
    return float(series.iloc[-1])

# =========================
# FRED DATA
# =========================
def get_fred_data():

    us_rate = fred.get_series("DGS10")
    cpi = fred.get_series("CPIAUCSL")
    unemployment = fred.get_series("UNRATE")

    return {
        "us_rate": safe_last(us_rate),
        "cpi": safe_last(cpi),
        "unemployment": safe_last(unemployment)
    }

# =========================
# PRICE DATA
# =========================
def get_price(ticker):

    df = yf.download(
        ticker,
        period="6mo",
        interval="1d",
        auto_adjust=True
    )

    df = df.dropna()

    return df["Close"].astype(float).to_numpy().flatten()

# =========================
# DATASET BUILD
# =========================
def build_dataset(ticker):

    fred_data = get_fred_data()
    price = get_price(ticker)

    X = []
    y = []

    # 20일 이동평균을 사용하므로 시작 인덱스 20
    for i in range(20, len(price)):

        window5 = price[i-5:i]
        window10 = price[i-10:i]
        window20 = price[i-20:i]

        features = [

            # =====================
            # FRED (3)
            # =====================
            float(fred_data["us_rate"]),
            float(fred_data["cpi"]),
            float(fred_data["unemployment"]),

            # =====================
            # Price Lag (6)
            # =====================
            float(price[i-1]),
            float(price[i-2]),
            float(price[i-3]),
            float(price[i-4]),
            float(price[i-5]),
            float(price[i-10]),

            # =====================
            # Moving Average (3)
            # =====================
            float(np.mean(window5)),
            float(np.mean(window10)),
            float(np.mean(window20)),

            # =====================
            # Volatility (3)
            # =====================
            float(np.std(window5)),
            float(np.std(window10)),
            float(np.std(window20)),

            # =====================
            # Momentum (3)
            # =====================
            float((price[i-1] - price[i-2]) / (price[i-2] + 1e-8)),
            float((price[i-1] - price[i-5]) / (price[i-5] + 1e-8)),
            float((price[i-1] - price[i-10]) / (price[i-10] + 1e-8)),

            # =====================
            # Range (2)
            # =====================
            float(np.max(window10) - np.min(window10)),
            float(price[i-1] - np.mean(window10))
        ]

        X.append(features)
        y.append(float(price[i]))

    return (
        np.array(X, dtype=float),
        np.array(y, dtype=float)
    )

# =========================
# MODEL + SCATTER
# =========================
def make_scatter(ticker="005930.KS"):

    X, y = build_dataset(ticker)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)

    pred = model.predict(X)

    # =====================
    # SCATTER PLOT
    # =====================
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        y,
        pred,
        alpha=0.6
    )

    min_v = min(y.min(), pred.min())
    max_v = max(y.max(), pred.max())

    ax.plot(
        [min_v, max_v],
        [min_v, max_v],
        "r--",
        linewidth=2
    )

    ax.set_title(
        f"{ticker} Actual vs Prediction (RandomForest)"
    )

    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.grid(True)

    # =====================
    # BASE64 IMAGE
    # =====================
    buf = io.BytesIO()

    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight"
    )

    buf.seek(0)

    img = base64.b64encode(
        buf.getvalue()
    ).decode("utf-8")

    plt.close(fig)

    return {
        "image": img,
        "actual": y.tolist(),
        "pred": pred.tolist(),
        "feature_count": X.shape[1],
         "X": X.tolist()  
    }

# =========================
# TEST
# =========================
if __name__ == "__main__":

    result = make_scatter("005930.KS")

    print("Feature Count:", result["feature_count"])
    print("Actual Sample:", len(result["actual"]))
    print("Prediction Sample:", len(result["pred"]))