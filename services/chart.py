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

    # [기존] 종가 데이터 및 이동평균
    stock_raw = df["Close"].to_numpy().reshape(-1).astype(float)
    rate_raw = (
        df["Close"]
        .rolling(window=5)
        .mean()
        .bfill()
        .to_numpy()
        .reshape(-1)
        .astype(float)
    )

    # 🔥 [추가] 일일 수익률 계산 (Daily Return)
    # pct_change()와 유사하게 (현재가 - 전일가) / 전일가 로 계산 후 첫 원소 결측치 채움
    returns_raw = df["Close"].pct_change().bfill().to_numpy().reshape(-1).astype(float)

    # 0 나누기 방지 및 스케일링 (기존 로직 유지)
    stock = stock_raw / (stock_raw[0] if stock_raw[0] != 0 else 1)
    rate = rate_raw / (rate_raw[0] if rate_raw[0] != 0 else 1)
    
    # 수익률은 비율 자체이므로 별도의 가격 스케일링 없이 그대로 사용하거나, 
    # 분석 안정성을 위해 유지합니다.
    return stock, rate, returns_raw


# ==========================
# 2. dataset 생성 (수익률 변수 추가)
# ==========================
def create_dataset(stock, rate, returns_raw, window=10):
    stock = np.asarray(stock).reshape(-1)
    rate = np.asarray(rate).reshape(-1)
    returns_raw = np.asarray(returns_raw).reshape(-1)

    X_stock, X_rate, y_stock, y_return = [], [], [], []

    for i in range(len(stock) - window):
        X_stock.append(stock[i:i+window])
        X_rate.append(rate[i:i+window])
        
        # 타겟 설정
        y_stock.append(float(stock[i+window]))         # 다음 날의 스케일링된 종가
        y_return.append(float(returns_raw[i+window]))  # 다음 날의 수익률

    return (
        np.array(X_stock),
        np.array(X_rate),
        np.asarray(y_stock).reshape(-1),
        np.asarray(y_return).reshape(-1)
    )


# ==========================
# 3. 메인 (예측 및 차트 생성)
# ==========================
def make_chart(ticker="005930.KS"):
    stock, rate, returns_raw = load_data(ticker)

    WINDOW = 10
    X_stock, X_rate, y_stock, y_return = create_dataset(stock, rate, returns_raw, WINDOW)

    # ======================
    # 데이터 보호
    # ======================
    if len(y_stock) == 0:
        return {
            "image": "",
            "actual_list": [], "pred_list": [],
            "actual_return_list": [], "pred_return_list": [],
            "actual_last": 0, "pred_last": 0,
            "actual_return_last": 0, "pred_return_last": 0,
            "rmse": 0, "r2": 0,
            "rmse_return": 0, "r2_return": 0,
            "live_price": 0, "ticker": ticker
        }

    # ======================
    # 모델 빌드 (다중 출력 모델 구조)
    # ======================
    # 입력층
    stock_input = Input(shape=(WINDOW,), name="stock_in")
    rate_input = Input(shape=(WINDOW,), name="rate_in")
    
    # 특징 추출 (공유 및 결합)
    x1 = Dense(32, activation="relu")(stock_input)
    x1 = Dense(16, activation="relu")(x1)
    
    x2 = Dense(32, activation="relu")(rate_input)
    x2 = Dense(16, activation="relu")(x2)
    
    merged = Concatenate()([x1, x2])
    
    # 공통 은닉층
    x = Dense(64, activation="relu")(merged)
    x = Dense(32, activation="relu")(x)
    
    # 🔥 출력층 분리 (1. 종가 예측, 2. 수익률 예측)
    output_stock = Dense(1, name="stock_out")(x)
    output_return = Dense(1, name="return_out")(x)

    model = Model(inputs=[stock_input, rate_input], outputs=[output_stock, output_return])
    model.compile(optimizer="adam", loss={"stock_out": "mse", "return_out": "mse"})

    # 학습 진행 (y_stock과 y_return을 리스트로 전달)
    model.fit([X_stock, X_rate], [y_stock, y_return], epochs=30, batch_size=8, verbose=0)

    # ======================
    # 예측 및 형변환
    # ======================
    pred_stock_out, pred_return_out = model.predict([X_stock, X_rate], verbose=0)
    
    pred_stock = np.asarray(pred_stock_out, dtype=float).reshape(-1)
    pred_return = np.asarray(pred_return_out, dtype=float).reshape(-1)
    
    y_stock = np.asarray(y_stock, dtype=float).reshape(-1)
    y_return = np.asarray(y_return, dtype=float).reshape(-1)

    # ======================
    # 평가 지표 산출
    # ======================
    rmse_stock = float(np.sqrt(mean_squared_error(y_stock, pred_stock)))
    r2_stock = float(r2_score(y_stock, pred_stock))
    
    rmse_return = float(np.sqrt(mean_squared_error(y_return, pred_return)))
    r2_return = float(r2_score(y_return, pred_return))

    # ======================
    # 그래프 시각화 (종가 차트 + 수익률 차트 2단 구성)
    # ======================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 1층: 종가 예측
    ax1.plot(y_stock, label="Actual Stock (Scaled)", color="blue")
    ax1.plot(pred_stock, label="Predicted Stock (Scaled)", color="orange", linestyle="--")
    ax1.set_title(f"{ticker} Stock Price Prediction")
    ax1.legend()
    ax1.grid(True)

    # 2층: 수익률 예측
    ax2.plot(y_return, label="Actual Return", color="green")
    ax2.plot(pred_return, label="Predicted Return", color="red", linestyle="--")
    ax2.set_title(f"{ticker} Daily Return Prediction")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    # 이미지 인코딩
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    # ======================
    # Live Price 가져오기
    # ======================
    try:
        live_price = float(yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1])
    except:
        live_price = 0.0

    # ======================
    # 최종 결과 반환
    # ======================
    return {
        "image": img,
        "ticker": ticker,
        "live_price": live_price,
        
        # 종가 데이터
        "actual_list": y_stock.tolist(),
        "pred_list": pred_stock.tolist(),
        "actual_last": float(y_stock[-1]),
        "pred_last": float(pred_stock[-1]),
        "rmse": rmse_stock,
        "r2": r2_stock,
        
        # 🔥 추가된 수익률 데이터
        "actual_return_list": y_return.tolist(),
        "pred_return_list": pred_return.tolist(),
        "actual_return_last": float(y_return[-1]),
        "pred_return_last": float(pred_return[-1]),
        "rmse_return": rmse_return,
        "r2_return": r2_return
    }

# 사용 예시
# res = make_chart("005930.KS")
# print("마지막 예측 수익률:", res["pred_return_last"])