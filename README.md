# 📊 Customer Churn Prediction

This project predicts customer churn using machine learning and deploys an API using FastAPI.

## 🚀 Features
- **ML Pipeline:** Data processing, model training (RandomForest).
- **API Deployment:** FastAPI + Docker for serving predictions.
- **CI/CD:** GitHub Actions for automated testing & Docker deployment.

## 🔧 Setup Instructions
1️⃣ Clone the repository:  
```bash
git clone https://github.com/VKritskaia/customer-churn-prediction.git
cd customer-churn-prediction

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt

3️⃣ Train the model:
```bash
python src/train.py

4️⃣ Run the API:
```bash
uvicorn api.main:app --reload

5️⃣ Test API:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"Tenure":12,"MonthlyCharges":70.5,"TotalCharges":500.0,"Contract_Two year":0,"PaymentMethod_Credit card":1}'

🐳 Deploy with Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api