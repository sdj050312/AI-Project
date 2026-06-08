import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate, Dense

# CMD 화면을 깔끔하게 정리하기 위한 텐서플로 로그 제어
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("\n" + "=" * 65)
print("🚀 [CMD TEST] 글로벌 국채/주가 분석 및 연 20% 수익률 시뮬레이션")
print("=" * 65)

# ==========================
# 1. 데이터 로드
# ==========================
data_path = "../data/gdp_money_supply_data.xls"
if not os.path.exists(data_path):
    # 파일이 없을 때를 대비한 테스트용 더미 데이터 생성
    dates = pd.date_range(start="2001-01-01", periods=50, freq="YE")
    df = pd.DataFrame(
        {
            "id": [2] * 50,
            "date_price": dates,
            "stock": np.linspace(100, 300, 50) + np.random.normal(0, 10, 50),
            "base_rate": np.linspace(0.25, 2.5, 50),
        }
    )
else:
    df = pd.read_excel(data_path, engine="openpyxl")

df.columns = df.columns.str.strip()

# ==========================
# 2. 일본 데이터 필터링
# ==========================
df = df[df["id"] == 2].copy()
df = df.sort_values("date_price").dropna(subset=["stock", "base_rate"])

# ==========================
# 3. 데이터 정규화 (첫 번째 값 기준)
# ==========================
stock = df["stock"].astype(float).values
rate = df["base_rate"].astype(float).values

stock = stock / stock[0]
rate = rate / rate[0]

# ==========================
# 4. Window 데이터셋 생성
# ==========================
WINDOW = 10


def create_dataset(stock, rate, window=10):
    X_stock, X_rate, y = [], [], []
    for i in range(len(stock) - window):
        X_stock.append(stock[i : i + window])
        X_rate.append(rate[i : i + window])
        y.append(stock[i + window])
    return np.array(X_stock), np.array(X_rate), np.array(y)


X_stock, X_rate, y = create_dataset(stock, rate, WINDOW)

# ==========================
# 5. Functional API 모델 빌드
# ==========================
stock_input = Input(shape=(WINDOW,), name="stock_input")
x1 = Dense(32, activation="relu")(stock_input)
x1 = Dense(16, activation="relu")(x1)

rate_input = Input(shape=(WINDOW,), name="rate_input")
x2 = Dense(32, activation="relu")(rate_input)
x2 = Dense(16, activation="relu")(x2)

merged = Concatenate()([x1, x2])
x = Dense(64, activation="relu")(merged)
x = Dense(32, activation="relu")(x)
output = Dense(1)(x)

model = Model(inputs=[stock_input, rate_input], outputs=output)
model.compile(optimizer="adam", loss="mse")

# ==========================
# 6. 모델 학습 실행
# ==========================
model.fit([X_stock, X_rate], y, epochs=50, batch_size=8, verbose=0)

# ==========================
# 7. 미래 예측 연산 (2050년까지 딥러닝 vs 연 20% 고정 수익률)
# ==========================
start_year = 2001
actual_years = list(range(start_year + WINDOW, start_year + WINDOW + len(y)))
last_year = actual_years[-1]
future_steps = max(0, 2050 - last_year)

last_stock = X_stock[-1].copy()
last_rate = X_rate[-1].copy()

# 비교를 위한 시작점 설정 (실제 데이터의 마지막 주가 값)
base_stock_value = y[-1]

future_pred_dl = []  # 딥러닝 예측값 배열
future_pred_20pct = []  # 연당 20% 성장값 배열

current_20pct_value = base_stock_value

for _ in range(future_steps):
    # 1) 딥러닝 모델 기반 예측
    p = model.predict(
        [last_stock.reshape(1, WINDOW), last_rate.reshape(1, WINDOW)], verbose=0
    )[0][0]
    p = max(0, p)
    future_pred_dl.append(p)

    # 딥러닝용 윈도우 시프트
    last_stock = np.append(last_stock[1:], p)
    last_rate = np.append(last_rate[1:], last_rate[-1])

    # 2) 연당 수익률 20% 복리 계산 시나리오 (현재값 * 1.2)
    current_20pct_value = current_20pct_value * 1.2
    future_pred_20pct.append(current_20pct_value)

future_years = list(range(last_year + 1, 2051))

# ==========================
# 8. CMD 출력용 최종 결과 비교 테이블
# ==========================
print(f"🔮 [시뮬레이션 결과] 2050년 장기 자산 예측 비교 스냅샷")
print("-" * 65)
print("  연도(Year)   |   🤖 딥러닝 모델 예측   |   📈 연당 수익률 20% 고정")
print("-" * 65)

# 5년 단위 및 최종 2050년 데이터를 추출하여 출력
for yr, dl_val, pct_val in zip(future_years, future_pred_dl, future_pred_20pct):
    if yr % 5 == 0 or yr == 2050:
        print(f"    {yr}년      |         {dl_val:.4f}         |         {pct_val:.4f}")

print("-" * 65)
print(f"💡 기준점({last_year}년 마지막 주가): {base_stock_value:.4f}")
print("🎉 워킹 테스트가 성공적으로 완료되었습니다.\n")