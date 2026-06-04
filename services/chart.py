import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Concatenate
from sklearn.metrics import mean_squared_error, r2_score


# ==========================
# 1. 데이터 로드 (yfinance)
# ==========================
def load_data(ticker="005930.KS"):

    df = yf.download(ticker, period="6mo", interval="1d")
    df = df.dropna()

    if len(df) == 0:
        raise ValueError("No data found for ticker")

    # 🔥 무조건 1차원 변환
    stock = df["Close"].to_numpy().reshape(-1).astype(float)

    rate = (
        df["Close"]
        .rolling(window=5)
        .mean()
        .bfill()
        .to_numpy()
        .reshape(-1)
        .astype(float)
    )

    # 🔥 0 나누기 방지
    stock = stock / (stock[0] if stock[0] != 0 else 1)
    rate = rate / (rate[0] if rate[0] != 0 else 1)

    return stock, rate


# ==========================
# 2. dataset 생성
# ==========================
def create_dataset(stock, rate, window=10):

    stock = np.asarray(stock).reshape(-1)
    rate = np.asarray(rate).reshape(-1)

    X_stock, X_rate, y = [], [], []

    for i in range(len(stock) - window):

        X_stock.append(stock[i:i+window])
        X_rate.append(rate[i:i+window])

        y.append(float(stock[i+window]))  # 🔥 float 강제

    return (
        np.array(X_stock),
        np.array(X_rate),
        np.asarray(y).reshape(-1)
    )


# ==========================
# 3. 메인
# ==========================
def make_chart(ticker="005930.KS"):

    stock, rate = load_data(ticker)

    WINDOW = 10
    X_stock, X_rate, y = create_dataset(stock, rate, WINDOW)

    y = np.asarray(y).reshape(-1)

    # ======================
    # 데이터 보호
    # ======================
    if len(y) == 0:
        return {
            "image": "",
            "actual_list": [],
            "pred_list": [],
            "actual_last": 0,
            "pred_last": 0,
            "rmse": 0,
            "r2": 0,
            "live_price": 0,
            "ticker": ticker
        }

    # ======================
    # 모델
    # ======================
    stock_input = Input(shape=(WINDOW,))
    x1 = Dense(32, activation="relu")(stock_input)
    x1 = Dense(16, activation="relu")(x1)

    rate_input = Input(shape=(WINDOW,))
    x2 = Dense(32, activation="relu")(rate_input)
    x2 = Dense(16, activation="relu")(x2)

    merged = Concatenate()([x1, x2])

    x = Dense(64, activation="relu")(merged)
    x = Dense(32, activation="relu")(x)

    output = Dense(1)(x)

    model = Model([stock_input, rate_input], output)
    model.compile(optimizer="adam", loss="mse")

    model.fit([X_stock, X_rate], y, epochs=30, batch_size=8, verbose=0)

    # ======================
    # 예측
    # ======================
    pred = model.predict([X_stock, X_rate], verbose=0).reshape(-1)

    # 🔥 강제 float 변환 (핵심)
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)

    # ======================
    # 평가
    # ======================
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    r2 = float(r2_score(y, pred))

    # ======================
    # 그래프
    # ======================
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(y, label="Actual Stock")
    ax.plot(pred, label="Prediction")

    ax.set_title(f"{ticker} Stock Prediction")
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    img = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    # ======================
    # live price
    # ======================
    try:
        live_price = float(
            yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        )
    except:
        live_price = 0.0

    # ======================
    # return (완전 안전)
    # ======================
    return {
        "image": img,
        "actual_list": y.tolist(),
        "pred_list": pred.tolist(),
        "actual_last": float(y[-1]),
        "pred_last": float(pred[-1]),
        "rmse": rmse,
        "r2": r2,
        "live_price": live_price,
        "ticker": ticker
    }