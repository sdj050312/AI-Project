import io
import base64
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify

from routes.korea import korea_bp
from routes.japan import japan_bp
from routes.thiland import th_bp

from services.predict import make_prediction
from services.chart import make_stock_chart  # ⭐ 추가

app = Flask(__name__)

# ======================
# Blueprint 등록
# ======================
app.register_blueprint(korea_bp)
app.register_blueprint(japan_bp)
app.register_blueprint(th_bp)


# ======================
# 메인 페이지
# ======================
@app.route('/')
def index():
    return render_template('index.html')


# ======================
# 차트 페이지
# ======================

@app.route("/")
def home():
    return "Server Running"

@app.route("/chart")
def chart():
    img = make_stock_chart()
    return render_template("chart.html", chart=img)

# ======================
# 예측 API
# ======================
@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    interest_rate = float(data.get('interest', 0.1))

    country = "japan"
    target_id = 1
    target_year = 2035

    # ======================
    # ML 예측
    # ======================
    result = make_prediction(
        target_id=target_id,
        interest_rate=interest_rate
    )

    # ======================
    # 그래프 생성 (API 응답용)
    # ======================
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(result["prices"], 'ko-', label="Actual")
    ax.plot(result["prediction"], 'r--o', label="Prediction")

    ax.set_title("Real Estate Price Prediction (Japan Default)")
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    plt.close(fig)

    # ======================
    # JSON 응답
    # ======================
    return jsonify({
        "image": image_base64,
        "final_price": result["final_price"],
        "change": result["change"],
        "change_percent": result["change_percent"],
        "loss": result.get("loss", 0),
        "accuracy": result.get("accuracy", 0)
    })


if __name__ == '__main__':
    app.run(debug=True)