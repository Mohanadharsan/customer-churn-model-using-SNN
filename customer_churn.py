import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import norse.torch as norse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
print("Loading dataset...")
df = pd.read_csv("Telco-Customer-Churn.csv")
df.drop('customerID', axis=1, inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    if col != 'Churn':
        df[col] = LabelEncoder().fit_transform(df[col])
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
df = df.fillna(0)
X = df.drop('Churn', axis=1).values
y = df['Churn'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
def rate_encode(data, time_steps=100):
    """
    Converts normalized features into spike trains using rate coding.
    data: numpy array (samples x features)
    """
    data = torch.tensor(data, dtype=torch.float32)
    spike_trains = torch.rand(time_steps, data.shape[0], data.shape[1])
    return (spike_trains < data.unsqueeze(0)).float()

time_steps = 100
print("Encoding spike trains...")
train_spikes = rate_encode(X_train, time_steps)
test_spikes = rate_encode(X_test, time_steps)
class SNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = norse.LIFCell()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, spikes):
        lif_state = None
        outputs = []
        for step in range(spikes.size(0)):
            x = torch.relu(self.fc1(spikes[step]))
            z, lif_state = self.lif1(x, lif_state)
            out = self.fc2(z)
            outputs.append(out)
        return torch.stack(outputs).mean(dim=0)  # Average over time

input_size = X.shape[1]
hidden_size = 64
output_size = 2

model = SNNModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 15
train_losses = []
print("Training model...")
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(train_spikes)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
if not os.path.exists("models"):
    os.makedirs("models")
torch.save(model.state_dict(), "models/snn_churn.pth")
print("\nModel saved as models/snn_churn.pth")
print("\nLoading model from models/snn_churn.pth...")
loaded_model = SNNModel(input_size, hidden_size, output_size)
loaded_model.load_state_dict(torch.load("models/snn_churn.pth", map_location=torch.device('cpu')))
loaded_model.eval()
print("Model loaded successfully!")
print("\nEvaluating model...")
with torch.no_grad():
    preds = loaded_model(test_spikes)
    _, predicted = torch.max(preds, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predicted))
cm = confusion_matrix(y_test, predicted)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.figure(figsize=(6, 4))
plt.bar(['Not Churn', 'Churn'], torch.bincount(predicted).numpy())
plt.title('Predicted Churn Distribution')
plt.show()
