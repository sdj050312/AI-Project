import io
import base64
import matplotlib
matplotlib.use('Agg')  # 서버 환경에서 문법 및 GUI 충돌 방지
import matplotlib.pyplot as plt
import numpy as np

from flask import Blueprint, render_template, request, jsonify
from services.predict import make_prediction

usa_bp = Blueprint("usa", __name__)

# 1. 화면 렌더링
@usa_bp.route('/usa')
def usa():
    return render_template("usa.html")

# 2. 미국 전용 예측 API (japan_predict와 완벽 싱크로율 100%)
@usa_bp.route('/usa/predict', methods=['POST'])
def usa_predict():
    data = request.get_json() or {}
    interest_rate = float(data.get('interest', 5.00))  # 미국 기본 금리 예시 (5.0%)

    try:
        # 🔥 미국 ID인 3번을 고정으로 전달하여 매크로 알고리즘 가동
        result = make_prediction(target_id=3, interest_rate=interest_rate)

        actual_prices = result["prices"]
        predicted_prices = result["prediction"]

        # 📊 [막대 그래프 최적화]
        # 전체 과거 데이터를 다 그리면 막대가 너무 얇아지므로, 최근 과거 15개 + 미래 예측 30개를 결합
        past_view_count = 15
        recent_actual = actual_prices[-past_view_count:]
        
        total_bars = past_view_count + len(predicted_prices)  # 총 45개 막대
        x_indexes = np.arange(total_bars)

        fig, ax = plt.subplots(figsize=(10, 5))

        # 1) 과거 실제 데이터 막대 (파란색)
        ax.bar(
            x_indexes[:past_view_count], 
            recent_actual, 
            color="#4fa8ff", 
            alpha=0.8, 
            label="Historical Actual"
        )

        # 2) 미래 시뮬레이션 예측 데이터 막대 (빨간색)
        ax.bar(
            x_indexes[past_view_count:], 
            predicted_prices, 
            color="#ff5c5c", 
            alpha=0.8, 
            label="Future Simulation (30 Steps)"
        )

        # 그래프 디자인 및 가독성 세팅 (미국 맞춤 타이틀)
        ax.set_title(f"USA Economic Simulation (Interest Rate: {interest_rate}%)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Market Price / Index", fontsize=11)
        
        # X축 눈금 레이블 정의 (과거 시점은 -15, 미래 시점은 +1 형태로 표현)
        x_labels = [f"-{past_view_count-i}" for i in range(past_view_count)] + [f"+{i+1}" for i in range(len(predicted_prices))]
        
        # 글자가 겹치지 않도록 3칸 간격으로 솎아내고 45도 회전
        sample_interval = 3
        ax.set_xticks(x_indexes[::sample_interval])
        ax.set_xticklabels([x_labels[i] for i in range(0, total_bars, sample_interval)], rotation=45, fontsize=9)

        # 과거와 미래 경계선에 세로 점선 및 텍스트 추가
        ax.axvline(x=past_view_count - 0.5, color="#6c757d", linestyle="--", alpha=0.7)
        max_val = max(max(recent_actual), max(predicted_prices))
        ax.text(past_view_count - 0.3, max_val * 0.95, 'Prediction Start →', color="#6c757d", fontsize=9, fontweight='bold')

        ax.legend(loc="upper left")
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # 💾 이미지 인코딩 및 변환 (메모리 버퍼형 스냅샷 추출)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)

        # japan.py와 완벽히 일치하는 리턴 JSON 키값 포맷팅
        return jsonify({
            "status": "success",
            "image": img,
            "final_price": result["final_price"],
            "change": result["change"],
            "change_percent": result.get("return_rate", 0),
            "accuracy": result["accuracy"]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400