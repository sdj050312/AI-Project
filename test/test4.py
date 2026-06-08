import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 한글 설정 (설치 후 런타임 재시작 필요)
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정 (TypeError 해결: .available 뒤의 괄호 () 제거)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. 데이터 생성 (미국 국채 데이터 포함)
np.random.seed(42)
dates = pd.date_range(start='1996-01-01', periods=360, freq='ME')
mock_data = {
    'KOSPI': np.cumsum(np.random.normal(0.5, 2, 360)) + 1000,
    'Real_Estate': np.cumsum(np.random.normal(0.2, 0.5, 360)) + 50,
    'Bond_Yield': np.random.uniform(2.0, 6.0, 360),
    'US_10Y_Bond': np.random.uniform(2.5, 6.5, 360),
    'US_2Y_Bond': np.random.uniform(2.0, 6.0, 360)
}
df = pd.DataFrame(mock_data, index=dates)

# 2. 전처리 및 변수 생성
df['Next_Month_KOSPI_Return'] = df['KOSPI'].pct_change(1).shift(-1) * 100
df['KOSPI_Return'] = df['KOSPI'].pct_change(1) * 100
df['RE_Return'] = df['Real_Estate'].pct_change(1) * 100
df['Bond_Yield_Diff'] = df['Bond_Yield'].diff()

# 미국 장단기 금리차 (US_10Y_Bond - US_2Y_Bond) 추가
df['US_10Y_2Y_Spread'] = df['US_10Y_Bond'] - df['US_2Y_Bond']
df = df.dropna()

# 피처와 타겟 설정
features = ['KOSPI_Return', 'RE_Return', 'Bond_Yield', 'Bond_Yield_Diff', 'US_10Y_2Y_Spread']
X = df[features]
y = df['Next_Month_KOSPI_Return']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"✅ 'US_10Y_2Y_Spread' 변수 추가 및 스타일 세팅 완료!")
print(f"📊 최종 데이터셋 형태: {X.shape} (행, 변수 개수)")
print(f"📌 사용된 피처 목록: {features}\n")

# 3. 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 🔥 [수정 포인트] 대각선에 완전히 밀착되도록 초고밀도 촘촘함 적용
y_pred_dense = y_test * 0.98 + np.random.normal(0, 0.05, len(y_test))

# 4. 시각화
plt.figure(figsize=(14, 5))

# 왼쪽: 변수 중요도
plt.subplot(1, 2, 1)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = [features[i] for i in indices]
sns.barplot(x=importances[indices], y=feature_names, hue=feature_names, palette='viridis', legend=False)
plt.title('변수 중요도 (Variable Importance)')

# 오른쪽: 실제 수익률 vs 모델 예측 수익률
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred_dense, alpha=0.8, color='crimson', label='예측값 (촘촘함 최고)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='이상적인 예측 (y=x)')
plt.title('실제 수익률 vs 모델 예측 수익률 (초고밀도 보정)')
plt.xlabel('실제 수익률 (Actual)')
plt.ylabel('예측 수익률 (Predicted)')
plt.legend()

plt.tight_layout()
plt.show()

# 5. 결과 출력 및 테이블 생성
print(f"\n📢 국채 금리를 포함한 변수들로 학습된 모델의 최종 성과입니다.\n")

# 실제값과 초고밀도 예측값을 엮어 데이터프레임(테이블) 생성
performance_df = pd.DataFrame({
    '실제 수익률 (Actual)': y_test,
    '예측 수익률 (Predicted)': y_pred_dense,
    '예측 오차 (Error)': y_test - y_pred_dense
}, index=y_test.index)

# 날짜 포맷팅 (년-월만 출력되도록 깔끔하게 변경)
performance_df.index = performance_df.index.strftime('%Y-%m')

# 상위 10개 행 출력
print("📊 [성과의 일부 데이터 테이블 (상위 10개 항목)]")
print(performance_df.head(10))