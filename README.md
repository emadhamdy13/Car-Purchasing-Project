# Car Purchase Amount Prediction - End-to-End Deep Learning Pipeline 🚗💻

A comprehensive deep learning regression project designed to predict **Car Purchase Amounts** based on customer demographics and financial indicators. This repository demonstrates standard data processing, dimensionality reduction, target scaling, and neural network optimization.

---

## 📌 Project Overview
Predicting customer purchase capacity is critical for personalized marketing and dynamic pricing strategy. This project implements a full end-to-end Machine Learning pipeline:
1. Feature extraction and high-cardinality categorical encoding.
2. Dimensionality reduction with **PCA**.
3. Dual-scaling architecture for input features ($X$) and target variables ($y$).
4. Neural Network regression using **TensorFlow / Keras**.
5. Real-world metric evaluation using inverse transformation.

---

## 🛠️ Data Preprocessing & Engineering Strategy

- **Categorical Encoding & PCA:**  
  The dataset contained high-cardinality categorical features (such as `country`). These were encoded using **One-Hot Encoding**, followed by **Principal Component Analysis (PCA)** retaining **95% of total variance** (across 200 components) to avoid the curse of dimensionality.

- **Dual-Scaling Strategy (`StandardScaler`):**  
  - **Feature Scaling ($X$):** Standardized input features to accelerate gradient descent convergence.
  - **Target Scaling ($y$):** Applied `StandardScaler` to `y_train` (`scaler_y`) to eliminate exploding gradient issues and stabilize the loss function during training.

---

## 🧠 Model Architecture & Training

Built using **TensorFlow / Keras**:
- **Layer 1:** Dense Layer (32 neurons, `ReLU` activation).
- **Regularization:** Dropout (0.2) to mitigate overfitting and encourage generalization.
- **Layer 2:** Dense Layer (16 neurons, `ReLU` activation).
- **Output Layer:** Single Dense neuron for continuous numerical output.
- **Optimizer & Loss:** `Adam` optimizer ($lr = 0.001$) with Mean Squared Error (`MSE`).
- **Callbacks:** `EarlyStopping` monitoring `val_loss` with `restore_best_weights=True` to preserve the best-performing model weights.

---

## 📊 Evaluation & Metric Calibration

To compute meaningful real-world error metrics, model predictions ($\hat{y}_{scaled}$) were mapped back to the original dollar scale using `scaler_y.inverse_transform()`:

- **Mean Absolute Error (MAE):** ~$6,132.27
- **$R^2$ Score:** ~$0.3716$

---

## 🧰 Tech Stack
- **Python 3.x**
- **TensorFlow / Keras**
- **Scikit-Learn**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
