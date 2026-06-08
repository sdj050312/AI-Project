import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate, Dense

# CMD 화면을 깔끔하게 정리하기 위한 텐서플로 로그 제어
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("\n" + "=" * 50)
print("🚀 [CMD TEST] 글로벌 국채/주가 데이터 분석 시스템 시작")
print("=" * 50)

# ==========================
# 1. 데이터 로드
# ==========================
data_path = "../data/gdp_money_supply_data.xls"
if not os.path.exists(data_path):
    print(f"❌ 에러: 데이터 파일 경로를 찾을 수 없습니다. ({data_path})")
    print("   테스트용 가상 데이터를 생성하여 진행합니다.")
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

print(f"📊 필터링된 일본(ID:2) 데이터 개수: {len(df)}개")

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

print("\n📦 인공지능 학습용 데이터셋 형태 (Shape):")
print(f" - 주가 입력 데이터 (X_stock) : {X_stock.shape}")
print(f" - 금리 입력 데이터 (X_rate)  : {X_rate.shape}")
print(f" - 예측 대상 데이터 (y)       : {y.shape}")

# ==========================
# 5. Functional API 모델 빌드
# ==========================
# Stock Branch
stock_input = Input(shape=(WINDOW,), name="stock_input")
x1 = Dense(32, activation="relu")(stock_input)
x1 = Dense(16, activation="relu")(x1)

# Rate Branch
rate_input = Input(shape=(WINDOW,), name="rate_input")
x2 = Dense(32, activation="relu")(rate_input)
x2 = Dense(16, activation="relu")(x2)

# Merge
merged = Concatenate()([x1, x2])
x = Dense(64, activation="relu")(merged)
x = Dense(32, activation="relu")(x)
output = Dense(1)(x)

model = Model(inputs=[stock_input, rate_input], outputs=output)
model.compile(optimizer="adam", loss="mse")

# ==========================
# 6. 모델 학습 실행
# ==========================
print("\n🤖 인공지능 모델 딥러닝 학습을 시작합니다... (50 Epochs)")
history = model.fit(
    [X_stock, X_rate], y, epochs=50, batch_size=8, verbose=0
)  # CMD창 폭주 방지를 위해 verbose=0 처리
print("✅ 학습 완료!")

# ==========================
# 7. 학습 데이터 내부 예측 및 정확도 검증
# ==========================
pred = model.predict([X_stock, X_rate], verbose=0).flatten()

rmse = np.sqrt(mean_squared_error(y, pred))
r2 = r2_score(y, pred)

print("\n📊 [검증 결과] MODEL PERFORMANCE SCORE")
print("-" * 40)
print(f" 🔹 Root Mean Squared Error (RMSE) : {rmse:.6f}")
print(f" 🔹 결정계수 (R² Score)             : {r2:.6f}")
print("-" * 40)

# ==========================
# 8. 미래 예측 연산 (2050년까지)
# ==========================
start_year = 2001
actual_years = list(range(start_year + WINDOW, start_year + WINDOW + len(y)))
last_year = actual_years[-1]
future_steps = max(0, 2050 - last_year)

last_stock = X_stock[-1].copy()
last_rate = X_rate[-1].copy()
future_pred = []

for _ in range(future_steps):
    p = model.predict(
        [last_stock.reshape(1, WINDOW), last_rate.reshape(1, WINDOW)], verbose=0
    )[0][0]
    p = max(0, p)  # 음수 방지

    future_pred.append(p)

    # Window 데이터 쉬프트 이동 연산
    last_stock = np.append(last_stock[1:], p)
    last_rate = np.append(last_rate[1:], last_rate[-1])

future_years = list(range(last_year + 1, 2051))

# ==========================
# 9. CMD 출력용 결과 테이블 생성
# ==========================
print(f"\n🔮 [예측 완료] 2050년까지의 장기 주가 예측 시뮬레이션 결과")
print("-" * 40)
print("  연도(Year)   |   정규화된 주가 예측치(Normalized Price)")
print("-" * 40)

# 5년 단위로 스냅샷 쪼개서 CMD에 출력
for yr, val in zip(future_years, future_pred):
    if yr % 5 == 0 or yr == 2050:
        print(f"    {yr}년      |      {val:.4f}")

print("-" * 40)
print("🎉 CMD 텍스트 기반 시뮬레이션 테스트가 정상적으로 종료되었습니다.\n")