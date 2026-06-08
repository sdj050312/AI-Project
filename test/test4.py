import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate

# ==========================================
# 1. 데이터 수집 및 20개 변수 환경 세팅
# ==========================================
def generate_mock_integrated_dataset():
    print(">> 데이터 수집 및 병합 시작 (20개 변수 세트 구축)...")
    stock_df = yf.download('^KS11', start="2021-01-01", end="2026-01-01", progress=False)
    df = pd.DataFrame(index=stock_df.index)
    
    close_prices = stock_df['Close'].squeeze()
    df['Log_Return'] = np.log(close_prices / close_prices.shift(1))
    
    np.random.seed(42)
    features_20 = [
        'VIX', 'SP500', 'Nasdaq', 'Nikkei', 'Copper', 'Gold', 'Crude_Oil', 
        'Dollar_Index', 'Exchange_Rate', 'Volume', 'T10Y2Y', 'DGS10', 'DGS2', 
        'FEDFUNDS', 'CPI', 'PPI', 'UNRATE', 'INDPRO', 'PAYEMS', 'UMCSENT'
    ]
    for col in features_20:
        df[col] = np.cumsum(np.random.normal(0, 1, size=len(df))) + 100
        
    df['DGS2'] = df['DGS10'] * 0.95 + np.random.normal(0, 0.1, size=len(df))
    df['PPI'] = df['CPI'] * 1.02 + np.random.normal(0, 0.2, size=len(df))
    
    return df.dropna()

# ==========================================
# 2. 변수 정제 및 중요도 테이블 도출
# ==========================================
def advanced_feature_selection_table(df, feature_cols, target_col, corr_threshold=0.75, top_n=5):
    X_raw = df[feature_cols].ffill().bfill()
    
    corr_matrix = X_raw.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]
    
    drop_reasons = []
    for col in to_drop:
        correlated_with = upper_tri.index[upper_tri[col] > corr_threshold].tolist()
        drop_reasons.append({"변수명": col, "판정": "탈락 (다중공선성)", "원인 변수": ", ".join(correlated_with)})
    drop_table = pd.DataFrame(drop_reasons)
    
    X_filtered = X_raw.drop(columns=to_drop)
    
    y_target = df[target_col].shift(-1).loc[X_filtered.index].fillna(0)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_filtered, y_target)
    
    importance_table = pd.DataFrame({
        '변수명': X_filtered.columns,
        '중요도(Importance)': rf.feature_importances_
    }).sort_values(by='중요도(Importance)', ascending=False).reset_index(drop=True)
    importance_table['순위'] = importance_table.index + 1
    
    final_features = importance_table['변수명'].head(top_n).tolist()
    
    return importance_table[['순위', '변수명', '중요도(Importance)']], drop_table, final_features

# ==========================================
# 3. 딥러닝 다중 입력 시퀀스 데이터 변환
# ==========================================
def create_sequences(df, selected_features, window_size=20):
    ts_inputs, macro_inputs, targets = [], [], []
    
    df_scaled = df.copy()
    scaler = MinMaxScaler()
    df_scaled[selected_features] = scaler.fit_transform(df_scaled[selected_features])
    
    for i in range(len(df_scaled) - window_size):
        # Input 1: 과거 window_size 동안의 주가 수익률 흐름
        ts_inputs.append(df_scaled.iloc[i:i+window_size]['Log_Return'].values)
        # Input 2: 현재 시점의 엄선된 거시경제 중요 변수들
        macro_inputs.append(df_scaled.iloc[i+window_size][selected_features].values)
        # Target: 내일의 주가 수익률
        targets.append(df_scaled.iloc[i+window_size]['Log_Return'])
        
    return np.array(ts_inputs), np.array(macro_inputs), np.array(targets)

# ==========================================
# [신규 추가] 4.