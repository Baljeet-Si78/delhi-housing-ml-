# Delhi Housing Price Predictor 🏠📊

A complete, end-to-end Machine Learning web application designed to estimate residential real estate valuations across the Delhi-NCR region. Built using the Python Data Science stack, Scikit-Learn, and Streamlit.

---

## 📌 Project Overview

Real estate valuation in metropolitan regions like Delhi-NCR involves complex, non-linear relationships between physical property dimensions and localized spatial micro-markets. This project delivers an automated, transparent decision-support system by evaluating multiple supervised machine learning algorithms and deploying the optimal estimator via a low-latency web application.

Developed as a Capstone Project during the 2-Month Summer Industrial Training Program at **NIELIT Karkardooma, Delhi**.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Joblib
* **Visualization:** Matplotlib, Seaborn
* **Web Framework:** Streamlit
* **Version Control & Hosting:** Git, GitHub, Streamlit Community Cloud

---

## 📁 Repository Structure

```text
├── Delhi_v2.csv                 # Raw real estate transaction dataset
├── cleaned_delhi_housing.csv    # Cleaned dataset post-outlier removal
├── clean_data.py                # Script for data auditing & cleaning
├── train_model.py               # Model training script
├── tune_and_compare.py          # GridSearch hyperparameter tuning & evaluation script
├── model_analysis.py            # Feature importance & model diagnostic analysis
├── predict.py                   # Script for offline model inference testing
├── app.py                       # Interactive Streamlit web application interface
├── delhi_house_model.pkl        # Serialized Random Forest model & preprocessing pipeline
├── feature_importance.csv       # Feature weight rankings output
├── model_comparison.csv         # Comparative metric results across algorithms
├── feature_importance.png       # Feature importance bar chart asset
├── actual_vs_predicted.png      # Actual vs. Predicted price scatter plot asset
├── requirements.txt             # Environment dependencies for deployment
└── README.md                    # Project documentation
