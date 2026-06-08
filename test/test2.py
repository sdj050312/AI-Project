import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate

# ==========================================
# 1. FRED API 연동 및 데이터 수집
# ==========================================
# 발급받으신 FRED API Key를 여기에 입력하세요.
FRED_API_KEY = 'YOUR_FRED_API_KEY' 
fred = Fred(api_key=FRED_API_KEY)

def fetch_macro_data():
    """
    FRED API로부터 프로젝트에 필요한 핵심 거시경제 변수들을 수집합니다.
    """
    print(">> FRED API로부터 거시경제 데이터를 가져오는 중...")
    
    # FRED의 고유 Series ID를 사용하여 데이터 호출
    macro_dict = {
        'T10Y2Y': fred.get_series('T10Y2Y'),      # 1. 미국의 장단기 금리차 (10년물 - 2년물)
        'CPI': fred.get_series('CPIAUCSL'),       # 2. 미국 소비자 물가 지수 (CPI)
        'DGS10': fred.get_series('DGS10'),        # 3. 미국채 10년물 금리
    }
    
    # 데이터프레임 변환 및 일별 데이터로 정렬
    macro_df = pd.DataFrame(macro_dict)
    macro_df.index = pd.to_datetime(macro_df.index)
    
    # 주말이나 휴일 등 결측치는 직전 데이터로 채움 (Forward Fill)
    macro_df = macro_df.fillna(method='ffill')
    return macro_df

# ==========================================
# 2. 주가 데이터(yfinance)와 FRED 데이터 병합 및 전처리
# ==========================================
def prepare_integrated_dataset(ticker):
    # ① yfinance를 통한 주가 데이터 수집 (예: 코스피 '^KS11')
    stock_df = yf.download(ticker, start="2020-01-01", end="2026-01-01")
    
    # 공식 ①: 가격 -> 정상성(Stationarity) 확보를 위한 로그 수익률 변환
    stock_df['Log_Return'] = np.log(stock_df['Close'] / stock_df['Close'].shift(1))
    
    # 외부 시장 심리 지표 추가 (yfinance 기반 변수)
    stock_df['VIX'] = yf.download('^VIX', start="2020-01-01", end="2026-01-01")['Close']
    stock_df['Copper'] = yf.download('HG=F', start="2020-01-01", end="2026-01-01")['Close']
    
    # ② FRED 데이터 가져오기
    macro_df = fetch_macro_data()
    
    # ③ 날짜(Index) 기준으로 주가 데이터와 거시경제 데이터 병합 (Inner Join)
    # 데이터의 시간축(Timeline)을 하나로 통일하는 과정
    integrated_df = stock_df[['Log_Return', 'VIX', 'Copper']].merge(
        macro_df, left_index=True, right_index=True, how='inner'
    )
    
    return integrated_df.dropna()

# ==========================================
# 3. Functional API를 위한 다중 입력(Multi-Input) 시퀀스 제작
# ==========================================
def create_multi_input_sequences(df, window_size=20):
    """
    Input 1: 과거 20일간의 '로그 수익률' 패턴 (LSTM 입력용)
    Input 2: 현재 시점의 'FRED 거시경제 변수 + 시장심리 변수' (Dense 입력용)
    Target: 다음 날의 '로그 수익률'
    """
    ts_inputs, macro_inputs, targets = [], [], []
    
    # 스케일링을 위해 거시경제/심리 변수들만 선택
    feature_cols = ['VIX', 'Copper', 'T10Y2Y', 'CPI', 'DGS10']
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    for i in range(len(df) - window_size):
        # Input 1: 과거 window_size 동안의 주가 수익률 흐름
        ts_inputs.append(df.iloc[i:i+window_size]['Log_Return'].values)
        
        # Input 2: 현재 시점의 거시경제 지표 5개
        macro_inputs.append(df.iloc[i+window_size][feature_cols].values)
        
        # Target: 내일의 주가 수익률
        targets.append(df.iloc[i+window_size]['Log_Return'])
        
    return np.array(ts_inputs), np.array(macro_inputs), np.array(targets)

# ==========================================
# 4. 실행 및 확인
# ==========================================
# 전체 파이프라인 구동
dataset = prepare_integrated_dataset('^KS11')
X_ts, X_macro, y = create_multi_input_sequences(dataset, window_size=20)

# 차원 맞추기 (LSTM은 3차원 입력을 받음: [샘플 수, 타임스텝, 피처 수])
X_ts = X_ts.reshape(-1, 20, 1)

print("\n--- [데이터 전처리 및 결합 완료] ---")
print(f"1. 전체 데이터셋 크기: {dataset.shape}")
print(f"2. LSTM 입력 차원 (과거 주가 패턴): {X_ts.shape}")
print(f"3. Dense 입력 차원 (FRED 거시경제 변수): {X_macro.shape}")
print(f"4. 예측 목표치 차원 (내일의 수익률): {y.shape}")