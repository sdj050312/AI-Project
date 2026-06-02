import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Concatenate


def make_stock_chart():

    # ==========================
    # 1. 데이터 로드 (안전 경로)
    # ==========================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(BASE_DIR, "data", "gdp_money_supply_data.xls")

    df = pd.read_excel(file_path, engine="openpyxl")
    df.columns = df.columns.str.strip()

    df = df[df["id"] == 2].copy()
    df = df.sort_values("date_price").dropna(subset=["stock", "base_rate"])

    stock = df["stock"].values
    rate = df["base_rate"].values

    stock = stock / stock[0]
    rate = rate / rate[0]

    # ==========================
    # 2. window 데이터
    # ==========================
    def create_dataset(stock, rate, window=10):
        X_stock, X_rate, y = [], [], []

        for i in range(len(stock) - window):
            X_stock.append(stock[i:i+window])
            X_rate.append(rate[i:i+window])
            y.append(stock[i+window])

        return np.array(X_stock), np.array(X_rate), np.array(y)

    X_stock, X_rate, y = create_dataset(stock, rate, window=10)

    # ==========================
    # 3. 모델
    # ==========================
    stock_input = Input(shape=(10,))
    x1 = Dense(32, activation="relu")(stock_input)
    x1 = Dense(16, activation="relu")(x1)

    rate_input = Input(shape=(10,))
    x2 = Dense(32, activation="relu")(rate_input)
    x2 = Dense(16, activation="relu")(x2)

    merged = Concatenate()([x1, x2])

    x = Dense(64, activation="relu")(merged)
    x = Dense(32, activation="relu")(x)

    output = Dense(1)(x)

    model = Model(inputs=[stock_input, rate_input], outputs=output)
    model.compile(optimizer="adam", loss="mse")

    model.fit([X_stock, X_rate], y, epochs=50, batch_size=8, verbose=0)

    pred = model.predict([X_stock, X_rate])

    # ==========================
    # 4. 그래프 생성
    # ==========================
    plt.figure(figsize=(12,6))

    plt.plot(y, label="Actual")
    plt.plot(pred, label="Predicted")

    plt.legend()
    plt.grid(True)
    plt.title("Japan Stock Prediction")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    img = base64.b64encode(buf.getvalue()).decode("utf-8")

    plt.close()

    return img