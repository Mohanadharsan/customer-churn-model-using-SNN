Customer Churn Prediction using Spiking Neural Network

This project predicts customer churn using a Spiking Neural Network (SNN) built with PyTorch and Norse, and provides a Streamlit web app for predictions.

Steps to Run

1. Install dependencies

pip install pandas numpy torch norse scikit-learn matplotlib seaborn streamlit

2. Train the model

Run the training script to create the model file:

python customer_churn.py

It will save the model at:

models/snn_churn.pth

3. Run the Streamlit app

streamlit run app.py


Files in this project

Telco-Customer-Churn.csv → Dataset

customer_churn.py → Training script

app.py → Streamlit app for predictions

models/snn_churn.pth → Saved model (created after training)


Predict for single customer

Predict for multiple customers via CSV

Dashboard with accuracy and confusion matrix



