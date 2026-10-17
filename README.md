# Car Purchase Amount Prediction using Deep Learning 🚗💻

A professional machine learning project focused on building a high-performance deep learning regression model to predict car purchase amounts, designed to strengthen your professional portfolio.

## 🚀 Project Overview
This project implements an end-to-end data processing pipeline and a deep learning regression model to accurately predict car purchase amounts, focusing on minimizing error metrics and achieving high evaluation standards.

---

## 🛠️ Feature Engineering & Preprocessing
- **Categorical Encoding:** The `country` feature was encoded using One-Hot Encoding, followed by dimensionality reduction via **PCA** (retaining 95% variance across 200 components).
- **Feature & Target Scaling:** Successfully implemented `StandardScaler` on both the input features (`X`) and the target variable (`y`) to ensure training stability and optimal convergence.

---

## 🧠 Model Architecture
Built using **TensorFlow / Keras**:
- **Model Type:** Sequential Neural Network.
- **Activation Functions:** `ReLU` for hidden layers and a single output node for regression.
- **Optimizer & Loss:** `Adam` optimizer with Mean Squared Error (`MSE`) loss.
- **Regularization:** `EarlyStopping` with `restore_best_weights=True` to prevent overfitting.

---

## 📊 Evaluation & Metrics
Post-training predictions utilize `inverse_transform` to map scaled values back to their original real-world scale, computing reliable evaluation metrics:
- **Mean Absolute Error (MAE)**
- **R² Score**

---

## 🧰 Tech Stack
- **Python**
- **TensorFlow & Keras**
- **Scikit-Learn**
- **Pandas & NumPy**
- **Matplotlib**
