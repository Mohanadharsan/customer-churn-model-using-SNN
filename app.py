import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import norse.torch as norse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import io
class SNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def forward(self, spikes):
        lif_state = None
        outputs = []
        for step in range(spikes.size(0)):
            x = torch.relu(self.fc1(spikes[step]))
            z, lif_state = self.lif1(x, lif_state)
            out = self.fc2(z)
            outputs.append(out)
        return torch.stack(outputs).mean(dim=0)

def rate_encode(data, time_steps=100):
    data = torch.tensor(data, dtype=torch.float32)
    spike_trains = torch.rand(time_steps, data.shape[0], data.shape[1])
    return (spike_trains < data.unsqueeze(0)).float()

def load_model(input_size, hidden_size, output_size, model_path="models/snn_churn.pth"):
    model = SNNModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

st.title("Customer Churn Prediction using Spiking Neural Network")

st.sidebar.header("Options")
option = st.sidebar.radio("Choose an option:", 
                          ("Predict for Single Customer", "Batch Prediction from CSV", "Model Performance Dashboard"))
df = pd.read_csv("Telco-Customer-Churn.csv")
df.drop('customerID', axis=1, inplace=True)
categorical_cols = [col for col in df.select_dtypes(include=['object']).columns if col != 'Churn']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('Churn', axis=1).values
y = df['Churn'].values

input_size = X.shape[1]
hidden_size = 64
output_size = 2

model = load_model(input_size, hidden_size, output_size)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
if option == "Predict for Single Customer":
    st.subheader("Enter Customer Details")

    user_input = []
    for col in df.drop('Churn', axis=1).columns:
        value = st.number_input(f"{col}", value=float(df[col].mean()))
        user_input.append(value)

    if st.button("Predict Churn"):
        input_array = scaler.transform([user_input])
        spikes = rate_encode(input_array)
        with torch.no_grad():
            output = model(spikes)
            _, prediction = torch.max(output, 1)
        result = "Churn" if prediction.item() == 1 else "Not Churn"
        st.success(f"Prediction: *{result}*")
elif option == "Batch Prediction from CSV":
    st.subheader("Upload CSV file for batch churn prediction")
    file = st.file_uploader("Upload CSV", type=['csv'])

    if file:
        batch_df = pd.read_csv(file)
        for col in categorical_cols:
            if col in batch_df.columns:
                batch_df[col] = batch_df[col].map(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else 0)

        batch_df = batch_df.fillna(0)
        if 'Churn' in batch_df.columns:
            batch_df.drop('Churn', axis=1, inplace=True)

        input_data = scaler.transform(batch_df.values)
        spikes = rate_encode(input_data)

        with torch.no_grad():
            outputs = model(spikes)
            _, predictions = torch.max(outputs, 1)

        batch_df['Churn_Prediction'] = predictions.numpy()
        batch_df['Churn_Prediction'] = batch_df['Churn_Prediction'].map({0: 'Not Churn', 1: 'Churn'})

        st.write("### Predictions:")
        st.dataframe(batch_df)

       
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", data=csv, file_name="churn_predictions.csv", mime="text/csv")
elif option == "Model Performance Dashboard":
    st.subheader("Model Accuracy & Confusion Matrix")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    spikes = rate_encode(X_test)
    with torch.no_grad():
        outputs = model(spikes)
        _, predictions = torch.max(outputs, 1)

    acc = accuracy_score(y_test, predictions.numpy())
    st.metric("Test Accuracy", f"{acc*100:.2f}%")
    cm = confusion_matrix(y_test, predictions.numpy())
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)
    report = classification_report(y_test, predictions.numpy(), target_names=['Not Churn', 'Churn'])
    st.text("Classification Report:\n" + report)