
# BookGuardAI-Travel-Anomaly-Detection

A travel booking anomaly detection system developed using PyTorch Autoencoder, FastAPI, and Streamlit to identify unusual booking behaviour and support further investigation.

---

## 📌 Features
- Analyse travel booking records
- Detect unusual booking patterns using an Autoencoder
- Extract booking and historical behavioural features
- Calculate reconstruction errors and identify anomalies
- Flag unusual bookings for further review
- Perform live anomaly detection using FastAPI
- Visualise detection results through an interactive dashboard
- Analyse user-level and travel-market patterns

---

## 🛠️ Technologies Used
- Python
- PyTorch
- Pandas and NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Streamlit
- Plotly
- Matplotlib

---

## 📂 Project Structure
- app.py → FastAPI backend application
- dashboard.py → Streamlit dashboard
- bookguard_autoencoder.pt → Trained Autoencoder model
- preprocessing.pkl → Saved preprocessing components
- model_config.json → Model configuration and feature details
- booking_anomaly_results.csv → Anomaly detection results
- booking_features.csv → Engineered booking features
- feature_comparison.csv → Feature comparison results
- flagged_user_analysis.csv → User-level anomaly analysis
- flagged_market_analysis.csv → Travel-market analysis
- threshold_comparison.csv → Threshold sensitivity analysis
- run_summary.json → Pipeline execution summary
- requirements.txt → Required Python libraries

---

## ⚙️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/thanvinadella/BookGuardAI-Travel-Anomaly-Detection.git
cd BookGuardAI-Travel-Anomaly-Detection
```

### 2. Create a Virtual Environment (Optional)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI Backend

```bash
uvicorn app:app --reload
```

The API runs at:

http://localhost:8000

### 5. Start the Streamlit Dashboard

Open another terminal in the project directory and run:

```bash
streamlit run dashboard.py
```

The dashboard runs at:

http://localhost:8501

---

## 🧠 Model Architecture
- Deep Learning-based Autoencoder
- Input features: 12
- Encoder: 12 → 8 → 4
- Decoder: 4 → 8 → 12
- Activation function: ReLU
- Anomaly detection using reconstruction error
- Threshold selection using the 95th percentile of training errors

---

## 📊 Model Evaluation
- Training records: 6,394
- Future testing records: 1,599
- Flagged bookings: 92
- Normal bookings: 1,507
- Anomaly rate: 5.75%

The model uses a chronological train-test split to evaluate its ability to identify unusual patterns in future booking records.

**Note:** Flagged bookings represent potential anomalies requiring further investigation. They are not confirmed fraudulent transactions because verified fraud labels are unavailable.

---

## 🗄️ Data Processing
- Selected actual booking records from the original travel dataset.
- Integrated supplementary synthetic payment and cancellation information.
- Performed data cleaning and missing-value handling.
- Extracted booking and historical behavioural features.
- Applied median imputation and StandardScaler.
- Split the data chronologically into training and testing sets.

---

## 📊 Dashboard Modules
- **Overview:** Displays booking statistics and anomaly distribution.
- **Booking Detection:** Allows users to check individual bookings through the API.
- **Anomaly Records:** Displays flagged and normal booking records with filters.
- **Analytics & Insights:** Presents visual analysis of anomaly patterns.
- **Fraud Investigation:** Supports user-level and market-level investigation.

---

## 🎯 Objective

To develop an intelligent travel booking anomaly detection system that learns normal booking behaviour and identifies unusual activities using an unsupervised deep learning approach. The system integrates an Autoencoder model, FastAPI backend, and interactive Streamlit dashboard to support anomaly detection and investigation.
