import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# 1. 수집할 주요 경제 지표 티커 리스트 (글로벌 지수, 금리, 환율, 원자재 등)
tickers = {
    'KOSPI': '^KS11',
    'KOSDAQ': '^KQ11',
    'S&P500': '^GSPC',
    'NASDAQ': '^IXIC',
    'DOW': '^DJI',
    'Nikkei': '^N225',
    'Gold': 'GC=F',
    'Crude_Oil': 'CL=F',
    'USD_KRW': 'KRW=X',
    'US_10Y_Bond': '^TNX',
    'US_2Y_Bond': '^FVX',  # Added US 2-year bond yield
    'VIX': '^VIX',
    'Samsung': '005930.KS',
    'Apple': 'AAPL',
    'Tesla': 'TSLA',
    'Copper': 'HG=F' # Added Copper futures
}

def fetch_financial_data(ticker_dict):
    print("🔄 데이터를 수집 중입니다. 잠시만 기다려주세요...")
    # yfinance 최신 버전의 MultiIndex 대응을 위해 group_by='ticker' 사용
    data_raw = yf.download(list(ticker_dict.values()), start='2015-01-01', end='2024-12-31', progress=False)

    combined_df = pd.DataFrame(index=data_raw.index)

    # 각 티커별로 'Close' 가격만 추출하여 이름 변경
    inv_tickers = {v: k for k, v in ticker_dict.items()}
    for ticker_symbol in data_raw.columns.levels[1]:
        name = inv_tickers.get(ticker_symbol, ticker_symbol)
        combined_df[name] = data_raw['Close'][ticker_symbol]

    return combined_df

# 데이터 로드
market_df = fetch_financial_data(tickers)

# 2. 파생 변수 생성을 통해 변수 개수 50개 이상으로 확장
final_features = market_df.copy()

for col in market_df.columns:
    # 이동평균선 (MA5, MA20, MA60)
    final_features[f'{col}_MA5'] = market_df[col].rolling(window=5).mean()
    final_features[f'{col}_MA20'] = market_df[col].rolling(window=20).mean()
    final_features[f'{col}_MA60'] = market_df[col].rolling(window=60).mean()

    # 수익률 및 변동성
    final_features[f'{col}_Return'] = market_df[col].pct_change() * 100
    final_features[f'{col}_Vol'] = final_features[f'{col}_Return'].rolling(window=20).std()

# 결측치 제거
final_features = final_features.ffill().dropna()

print(f"\n✅ 변수 생성 완료!")
print(f"📊 최종 데이터셋 형태: {final_features.shape} (행, 변수 개수)")
print(f"📌 주요 변수 목록 (일부): {list(final_features.columns[:10])}...")
display(final_features.head())


# 한글 설정 (설치 후 런타임 재시작 필요)
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 생성
np.random.seed(42)
dates = pd.date_range(start='1996-01-01', periods=360, freq='ME')
mock_data = {
    'KOSPI': np.cumsum(np.random.normal(0.5, 2, 360)) + 1000,
    'Real_Estate': np.cumsum(np.random.normal(0.2, 0.5, 360)) + 50,
    'Bond_Yield': np.random.uniform(2.0, 6.0, 360)
}
df = pd.DataFrame(mock_data, index=dates)

# 2. 전처리
df['Next_Month_KOSPI_Return'] = df['KOSPI'].pct_change(1).shift(-1) * 100
df['KOSPI_Return'] = df['KOSPI'].pct_change(1) * 100
df['RE_Return'] = df['Real_Estate'].pct_change(1) * 100
df['Bond_Yield_Diff'] = df['Bond_Yield'].diff()
df = df.dropna()

features = ['KOSPI_Return', 'RE_Return', 'Bond_Yield', 'Bond_Yield_Diff']
X = df[features]
y = df['Next_Month_KOSPI_Return']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. 시각화
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [features[i] for i in indices]
sns.barplot(x=importances[indices], y=feature_names, hue=feature_names, palette='viridis', legend=False)
plt.title('변수 중요도 (Variable Importance)')

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='crimson')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('실제 수익률 vs 모델 예측 수익률')

plt.tight_layout()
plt.show()

# 5. 결과 출력
print(f"국채 금리를 포함한 변수들로 학습된 모델의 최종 성과입니다.")


# 변수 중요도 시각화
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=predictive_feature_importance_df.head(20), palette='viridis')
plt.title('KOSPI 다음 날 수익률 예측 모델의 상위 20개 변수 중요도')
plt.xlabel('중요도')
plt.ylabel('변수')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()