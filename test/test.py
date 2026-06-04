import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Concatenate

from sklearn.metrics import mean_squared_error, r2_score

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

df = (
    df.sort_values("date_price")
      .dropna(subset=["stock", "base_rate"])
)

# ==========================
# 3. 데이터 정규화
# ==========================
stock = df["stock"].astype(float).values
rate = df["base_rate"].astype(float).values

stock = stock / stock[0]
rate = rate / rate[0]

# ==========================
# 4. Window 생성
# ==========================
WINDOW = 10

def create_dataset(stock, rate, window=10):

    X_stock = []
    X_rate = []
    y = []

    for i in range(len(stock) - window):

        X_stock.append(
            stock[i:i + window]
        )

        X_rate.append(
            rate[i:i + window]
        )

        y.append(
            stock[i + window]
        )

    return (
        np.array(X_stock),
        np.array(X_rate),
        np.array(y)
    )

X_stock, X_rate, y = create_dataset(
    stock,
    rate,
    WINDOW
)

print("X_stock :", X_stock.shape)
print("X_rate  :", X_rate.shape)
print("y       :", y.shape)

# ==========================
# 5. Functional API 모델
# ==========================

# Stock Branch
stock_input = Input(
    shape=(WINDOW,),
    name="stock_input"
)

x1 = Dense(
    32,
    activation="relu"
)(stock_input)

x1 = Dense(
    16,
    activation="relu"
)(x1)

# Rate Branch
rate_input = Input(
    shape=(WINDOW,),
    name="rate_input"
)

x2 = Dense(
    32,
    activation="relu"
)(rate_input)

x2 = Dense(
    16,
    activation="relu"
)(x2)

# Merge
merged = Concatenate()([
    x1,
    x2
])

x = Dense(
    64,
    activation="relu"
)(merged)

x = Dense(
    32,
    activation="relu"
)(x)

output = Dense(1)(x)

model = Model(
    inputs=[
        stock_input,
        rate_input
    ],
    outputs=output
)

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# ==========================
# 6. 학습
# ==========================
history = model.fit(
    [X_stock, X_rate],
    y,
    epochs=50,
    batch_size=8,
    verbose=1
)

# ==========================
# 7. 학습 데이터 예측
# ==========================
pred = model.predict(
    [X_stock, X_rate],
    verbose=0
).flatten()

# ==========================
# 8. 정확도 계산
# ==========================
rmse = np.sqrt(
    mean_squared_error(
        y,
        pred
    )
)

r2 = r2_score(
    y,
    pred
)

print("\n===== MODEL SCORE =====")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")

# ==========================
# 9. 미래 예측 (2050년)
# ==========================

start_year = 2001

actual_years = list(
    range(
        start_year + WINDOW,
        start_year + WINDOW + len(y)
    )
)

last_year = actual_years[-1]

future_steps = max(
    0,
    2050 - last_year
)

print("\nFuture Steps :", future_steps)

last_stock = X_stock[-1].copy()
last_rate = X_rate[-1].copy()

future_pred = []

for _ in range(future_steps):

    p = model.predict(
        [
            last_stock.reshape(1, WINDOW),
            last_rate.reshape(1, WINDOW)
        ],
        verbose=0
    )[0][0]

    # 음수 방지
    p = max(0, p)

    future_pred.append(p)

    # Window Shift
    last_stock = np.append(
        last_stock[1:],
        p
    )

    # 금리는 마지막 값 유지
    last_rate = np.append(
        last_rate[1:],
        last_rate[-1]
    )

future_years = list(
    range(
        last_year + 1,
        2051
    )
)

# ==========================
# 10. 그래프
# ==========================
plt.figure(
    figsize=(14, 6)
)

# 실제 데이터
plt.plot(
    actual_years,
    y,
    label="Actual Stock"
)

# 학습 예측
plt.plot(
    actual_years,
    pred,
    label="Model Prediction"
)

# 미래 예측 연결
future_plot = [y[-1]] + future_pred

future_years_plot = (
    [actual_years[-1]]
    + future_years
)

plt.plot(
    future_years_plot,
    future_plot,
    "--",
    linewidth=2,
    label="Forecast to 2050"
)

plt.title(
    "America Korea Japan Stock Forecast using Base Rate"
)

plt.xlabel("Year")
plt.ylabel("Normalized Stock Price")

plt.xticks(
    range(
        2000,
        2051,
        5
    )
)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()