# 🌲 Algerian Forest Fire Prediction Pipeline 

### Tech Stack Used
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-%2341A694.svg?style=for-the-badge&logo=seaborn&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)

## 📌 Project Overview
This is an end-to-end Machine Learning regression project designed to predict the **Fire Weather Index (FWI)** of the Algerian forest regions based on meteorological data. 

The project encompasses the entire data science lifecycle: taking raw, unformatted data, structurally cleaning it, performing extensive Exploratory Data Analysis (EDA), building a mathematically robust regularized model to avoid overfitting, and finally deploying that model to a live **Flask Web Application** for real-time user predictions.

## 🧹 Phase 1: Data Cleaning & EDA
Real-world data is inherently messy. A significant portion of this project was dedicated to strictly preprocessing the raw CSV before any machine learning could occur. 
* **Dataset Merging:** Combined instances from two distinct regions (Bejaia Region and Sidi Bel-abbes Region) into one unified DataFrame to increase the model's geographical generalization.
* **Structural Formatting:** Stripped hidden whitespace from column headers and string values to prevent key errors during automated processing.
* **Handling Missing Data:** Programmatically identified and dropped corrupted rows and `NaN` values.
* **Type Casting & Encoding:** Converted raw string data into workable numeric formats (integers and floats) and transformed the textual target column (`Classes` containing "fire" and "not fire") into binary numerical values (`1` and `0`) for mathematical correlation.

## 🧠 Phase 2: Model Training & Regularization
To ensure the model performs perfectly on unseen testing data without simply memorizing the training dataset, standard Linear Regression was bypassed. 
* **Feature Scaling:** Applied `StandardScaler` to ensure all meteorological features (Temperature, Wind Speed, Humidity) contributed equally to the model without dominating one another due to differing units.
* **Regularization:** Rigorously tested **Lasso Regression (L1)**, **Ridge Regression (L2)**, and **Elastic Net** utilizing Cross-Validation to dynamically hunt for the perfect mathematical penalty.
* **Result:** The final tuned **Ridge** model successfully shrank the weights of highly correlated features, providing massive armor against overfitting and achieving an $R^2$ score in the upper 90s on unseen testing data.

## 🚀 Phase 3: Web Deployment Architecture
The core Machine Learning model (`ridge.pkl`) and the data scaler (`scaler.pkl`) were serialized and integrated into a lightweight, interactive **Flask** web server (`application.py`). 

This application bridges the gap between backend machine learning and frontend user experience, allowing end-users to input real-time meteorological data through a clean HTML interface (`home.html`) and instantly receive an FWI prediction.

## 📂 Repository Structure
This project utilizes an isolated, enterprise-level microservice file structure:
```text
Algerian-Forest-Fire-Project/
│
├── models/
│   ├── ridge.pkl               # Serialized Ridge Regression model
│   └── scaler.pkl              # Serialized StandardScaler
│
├── notebooks/
│   ├── Algerian_forest_fire... # Raw and Cleaned datasets (.csv)
│   ├── Algerianforestfire...   # Step-by-step EDA and data scrubbing (.ipynb)
│   └── Model Training...       # Model training, scaling, and Cross-Validation (.ipynb)
│
├── templates/              
│   ├── home.html               # The frontend UI for real-time data input and predictions
│   └── index.html              # The landing page UI
│
├── application.py              # The core Flask web server and prediction logic
└── requirements.txt            # Explicit project dependencies for environment replication
