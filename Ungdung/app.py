from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)

# Lấy đường dẫn file CSV cùng thư mục app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "customers.csv")

# 1. Load và train model từ dữ liệu
df = pd.read_csv(file_path)
X = df[['Age', 'Annual Income ($)', 'Spending Score (1-100)', 'Work Experience', 'Family Size']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans (k=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Lưu model & scaler
joblib.dump(kmeans, os.path.join(BASE_DIR, "kmeans_model.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = int(request.form["age"])
        income = float(request.form["income"])
        score = int(request.form["score"])
        exp = int(request.form["exp"])
        family = int(request.form["family"])

        # Load lại model và scaler
        scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
        kmeans = joblib.load(os.path.join(BASE_DIR, "kmeans_model.pkl"))

        # Chuẩn hóa và dự đoán
        input_data = [[age, income, score, exp, family]]
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]

        return render_template("result.html",
                               cluster=cluster, age=age, income=income,
                               score=score, exp=exp, family=family)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
