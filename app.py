import io
import base64
import matplotlib.pyplot as plt
from services.scatter import make_scatter
import matplotlib
matplotlib.use("Agg")
from flask import Flask, render_template, request, jsonify

from routes.korea import korea_bp
from routes.japan import japan_bp
from routes.usa import usa_bp

from services.chart import make_chart
from services.predict import make_prediction

app = Flask(__name__)

# ======================
# Blueprint
# ======================
app.register_blueprint(korea_bp)
app.register_blueprint(japan_bp)
app.register_blueprint(usa_bp)

# ======================
# 메인
# ======================
@app.route('/')
def index():
    return render_template('index.html')

#scatter 페이지 ------------------------------
@app.route("/scatter")
def scatter():
    ticker = request.args.get("ticker", "005930.KS")
    data = make_scatter(ticker)
    return render_template(
        "scatter.html",
        image=data["image"],
        actual=data["actual"],
        pred=data["pred"],
        X=data["X"]   # 🔥 이거 하나만 추가
    )

# ======================
# 차트 페이지
# ======================
@app.route('/chart')
def chart():
    ticker = request.args.get("ticker", "005930.KS")
    try:
        data = make_chart(ticker)
    except Exception as e:
        print("chart error:", e)
        return f"Error: {e}"

    if not data:
        return "No data returned"

    return render_template(
        "chart.html",
        image=data.get("image", ""),
        actual_list=data.get("actual_list", []),
        pred_list=data.get("pred_list", []),
        actual_last=data.get("actual_last", 0),
        pred_last=data.get("pred_last", 0),
        rmse=data.get("rmse", 0),
        r2=data.get("r2", 0),
        live_price=data.get("live_price", 0),
        ticker=data.get("ticker", ticker)
    )

# ======================
# API
# ======================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # 🔥 하드코딩된 target_id(1)를 프론트엔드가 보낸 값으로 동적 변경 (기본값 2: 한국)
    target_id = int(data.get('target_id', 2)) 
    interest_rate = float(data.get('interest', 3.5))

    try:
        # 모델 예측 함수 실행
        result = make_prediction(
            target_id=target_id,
            interest_rate=interest_rate
        )

        # Matplotlib을 이용한 결과 시각화 차트 생성
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(result["prices"], label="Actual", color="#4fa8ff")
        ax.plot(result["prediction"], label="Prediction", color="#ff5c5c", linestyle="--")
        
        # 차트 타이틀에 국가명 명시
        ax.set_title(f"{result['country']} Economic Simulation")
        ax.legend()
        ax.grid(True)

        # 이미지를 바이너리로 인코딩
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig) # 메모리 해제

        # 💡 앞서 매핑한 변수명(return_rate 등)과 프론트엔드 key값에 맞춰 반환
        return jsonify({
            "status": "success",
            "image": img,
            "country": result["country"],
            "current_price": result["current_price"],
            "final_price": result["final_price"],
            "change": result["change"],
            "change_percent": result.get("return_rate", 0), # 혹은 기존 변수명 사용
            "accuracy": result["accuracy"],
            "map_summary": result["map"]
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ======================
# 실행부 (클라우드타입 배포 설정 반영)
# ======================
if __name__ == '__main__':
    # host='0.0.0.0' 설정을 해야 클라우드타입 외부 환경에서 접근할 수 있습니다.
    # port=5000은 클라우드타입 서비스 설정 항목의 포트 번호와 동일해야 합니다.
    app.run(host='0.0.0.0', port=5000, debug=True)