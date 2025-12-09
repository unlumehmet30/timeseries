import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)
        return out
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
window=24
input_size=7
hidden_size=64
num_layers=2
output_size=1
# Load preprocessed data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
x_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
# Load the trained model
model=GRUNet(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('gru_model.pth'))
model.eval()
# Make predictions
with torch.no_grad():
    predictions = model(x_test).cpu().numpy()
# Load scalers
scaler_y = joblib.load('scaler_y.save')
# Inverse transform the predictions and actual values
predictions_inverse = scaler_y.inverse_transform(predictions)
y_test_inverse = scaler_y.inverse_transform(y_test.cpu().numpy())
# Calculate evaluation metrics  
rmse = np.sqrt(mean_squared_error(y_test_inverse, predictions_inverse))
mae = mean_absolute_error(y_test_inverse, predictions_inverse)
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'Mean Absolute Error: {mae:.2f}')
# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test_inverse[:200], label='Actual Traffic Volume')
plt.plot(predictions_inverse[:200], label='Predicted Traffic Volume')
plt.xlabel('Time')
plt.ylabel('Traffic Volume')
plt.title('Actual vs Predicted Traffic Volume')
plt.legend()
plt.savefig('actual_vs_predicted.png')
plt.show()

