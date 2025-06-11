# 🧠 Customer Lifetime Value (CLV) Prediction

This repository contains an end-to-end pipeline for predicting **Customer Lifetime Value (CLV)** using the [Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail). The model leverages **Artificial Neural Networks (ANN)** for prediction and utilizes **MLflow** for robust experiment tracking and reproducibility.

---

## 📊 Overview

**Customer Lifetime Value (CLV)** is a key business metric that estimates the total net profit attributed to the entire future relationship with a customer. Predicting CLV enables better:

- Marketing strategy formulation
- Customer segmentation
- Retention campaign targeting
- Revenue forecasting

This project includes data processing, modeling, training, evaluation, and experiment tracking.

---

## 📁 Dataset Description

The **Online Retail Dataset** is a real-world e-commerce transaction dataset that contains:

- **Country**: Country of customer residence
- **InvoiceDate**: Date of transaction
- **CustomerID**: Unique identifier for each customer
- **InvoiceNo**: Unique invoice number per transaction
- **StockCode**: Product identifier
- **Description**: Product description
- **Quantity**: Quantity of items purchased
- **UnitPrice**: Price per item

📥 **Sources**:
- UCI: [https://archive.ics.uci.edu/dataset/352/online+retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- Kaggle: Search for "Online Retail Dataset"

---

## 🧪 Problem Statement

The goal is to predict the **monetary value of a customer's future purchases**, using historical data. We approach this as a **regression problem**, where the target is the total value a customer is expected to generate over a defined horizon.

---

## 🧠 Modeling Approach

### 🔍 Feature Engineering
Key features engineered include:
- Recency, Frequency, Monetary (RFM) metrics
- Customer tenure
- Purchase patterns
- Time-based aggregations

### 🧱 Model Architecture: Artificial Neural Network (ANN)
- Input layer aligned with number of engineered features
- 2–3 hidden layers with ReLU activation
- Output layer with linear activation for regression
- Optimized with Adam, MSE loss function

### 🧵 Training Pipeline
- Data preprocessing using `pandas` and `scikit-learn`
- Train/validation split
- Training monitored and logged with `MLflow`
- Model persisted and versioned for reproducibility

---

## 🧰 Tools & Frameworks

| Tool       | Purpose                        |
|------------|--------------------------------|
| Python     | Core programming language      |
| Pandas     | Data manipulation              |
| NumPy      | Numerical computation          |
| Scikit-learn | Preprocessing utilities      |
| TensorFlow/Keras or PyTorch | Model implementation |
| MLflow     | Experiment tracking and model registry |

---

## 🔬 MLflow Tracking

MLflow is used to:
- Log metrics like loss, MAE, RMSE
- Track hyperparameters
- Save and register models
- Compare different model runs visually

Example:
```bash
mlflow ui
Access the MLflow dashboard at http://localhost:5000 to explore and compare experiments.

📈 Results
The ANN-based model demonstrated strong generalization capability, especially when fine-tuned with:

Proper scaling and feature normalization

Balanced mini-batches during training

Dropout regularization to prevent overfitting



📌 How to Run
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/clv-ann-mlflow.git
cd clv-ann-mlflow
2. Setup environment
bash
Copy
Edit
pip install -r requirements.txt
3. Launch training
bash
Copy
Edit
python train.py
4. Start MLflow UI
bash
Copy
Edit
mlflow ui
📚 References
UCI ML Repository: Online Retail Data Set

Kaggle: Online Retail Dataset

MLflow Documentation

CLV Theory - Harvard Business Review

🧠 Future Improvements
Integrate RFM-based customer segmentation prior to modeling

Use sequence models (RNNs) for purchase pattern learning

Deploy the model as a REST API with FastAPI or Flask

Extend to real-time prediction for live transactions