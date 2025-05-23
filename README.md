# Air Quality Forecasting Using Deep Learning

This project focuses on forecasting air quality indicators from the AirQualityUCI dataset using deep learning models implemented in PyTorch. It includes full data preprocessing, model development, training, and evaluation, with a strong emphasis on temporal sequence modeling.

## 🌫️ Dataset

- **Source:** [AirQualityUCI](https://archive.ics.uci.edu/ml/datasets/air+quality)
- **Features Used:** CO(GT), C6H6(GT), Temperature (T), Relative Humidity (RH), Absolute Humidity (AH)
- **Target:** Forecasting pollutant concentration and environmental metrics

## 🧹 Preprocessing Steps

- Dropping irrelevant or corrupt columns (`Unnamed: 15`, `Unnamed: 16`)
- Parsing and formatting timestamps
- Replacing `-200` (sensor error values) with `NaN` and filtering them out
- Standardizing numerical features with `MinMaxScaler` or `StandardScaler`

## 🧠 Deep Learning Models Implemented

- **Fully Connected Neural Networks (FCNN)**
- **Long Short-Term Memory (LSTM) networks**
- **Gated Recurrent Units (GRU)**

These models are trained using PyTorch with careful sequence preparation to capture time-dependent patterns in air quality.

## 🧪 Evaluation Metrics

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

## 📊 Visualization

- Forecast vs Actual plots
- Error distribution and residuals
- Training loss curves
- Temporal trends in predictions

## 🧰 Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch torchinfo
