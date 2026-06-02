import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Concatenate

# ==========================
# 1. 데이터 로드
# ==========================
df = pd.read_excel(
    "../data/gdp_money_supply_data.xls",
    engine="openpyxl"
)

df.columns = df.columns.str.strip()

# ==========================
# 2. 일본 데이터 필터
# ==========================
df = df[df["id"] == 2].copy()
df = df.sort_values("date_price").dropna(subset=["stock", "base_rate"])

# ==========================
# 3. 스케일링 (중요)
# ==========================
stock = df["stock"].values
rate = df["base_rate"].values

stock = stock / stock[0]
rate = rate / rate[0]

# ==========================
# 4. Window 데이터 생성
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
# 5. Functional API 모델
# ==========================

# stock branch
stock_input = Input(shape=(10,), name="stock_input")
x1 = Dense(32, activation="relu")(stock_input)
x1 = Dense(16, activation="relu")(x1)

# rate branch
rate_input = Input(shape=(10,), name="rate_input")
x2 = Dense(32, activation="relu")(rate_input)
x2 = Dense(16, activation="relu")(x2)

# merge
merged = Concatenate()([x1, x2])

x = Dense(64, activation="relu")(merged)
x = Dense(32, activation="relu")(x)

output = Dense(1)(x)

model = Model(inputs=[stock_input, rate_input], outputs=output)

model.compile(
    optimizer="adam",
    loss="mse"
)

# ==========================
# 6. 학습
# ==========================
model.fit(
    [X_stock, X_rate],
    y,
    epochs=30,
    batch_size=8,
    verbose=1
)

# ==========================
# 7. 예측 (train 구간)
# ==========================
pred = model.predict([X_stock, X_rate])

# ==========================
# 8. 미래 2050까지 확장
# ==========================

future_steps = 30  # 미래 30 step (대충 2050 느낌)
last_stock = X_stock[-1]
last_rate = X_rate[-1]

future_pred = []

for _ in range(future_steps):
    p = model.predict(
        [last_stock.reshape(1,10), last_rate.reshape(1,10)]
    )[0][0]

    future_pred.append(p)

    # window shift
    last_stock = np.append(last_stock[1:], p)
    last_rate = np.append(last_rate[1:], last_rate[-1])  # 금리는 고정 or 완만 변화

# ==========================
# 9. x축 (연도 생성)
# ==========================
start_year = 2001

actual_years = list(range(start_year + 10, start_year + 10 + len(y)))
future_years = list(range(actual_years[-1] + 1, actual_years[-1] + 1 + len(future_pred)))

# ==========================
# 10. 그래프
# ==========================
plt.figure(figsize=(14,6))

plt.plot(actual_years, y, label="Actual Stock")
plt.plot(future_years, future_pred, label="Predicted Stock (to 2050)", linestyle="--")

plt.title("Japan Stock Forecast (Functional API + Base Rate)")
plt.xlabel("Year")
plt.ylabel("Normalized Stock Price")

plt.xticks(range(2000, 2051, 5))
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()