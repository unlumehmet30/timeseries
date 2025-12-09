from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
import joblib

app = FastAPI(title="Traffic Volume Prediction API")

# Input model: seq is a list of 24 timesteps, each timestep is a list of 7 features
class TrafficData(BaseModel):
    seq: List[List[float]]
  # Annotated: List of 24 lists, each of length 7

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out

# device & model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 7
hidden_size = 64
num_layers = 2
output_size = 1

model = GRUNet(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('gru_model.pth', map_location=device))
model.eval()

scaler_y = joblib.load('scaler_y.save')

@app.post("/predict")
def predict_traffic(data: TrafficData):
    # seq should be shape (24, 7)
    input_seq = np.array(data.seq, dtype=np.float32)

    # validation
    if input_seq.ndim != 2 or input_seq.shape != (24, 7):
        return {
            "error": "Input sequence must be a 2D list with shape (24, 7). "
                     "Provide 24 time steps and 7 features per step."
        }

    # create batch dimension -> (1, 24, 7)
    input_tensor = torch.from_numpy(input_seq).float().unsqueeze(0).to(device)  # shape: (1,24,7)

    with torch.no_grad():
        prediction = model(input_tensor)                     # shape (1,1)
        prediction_np = prediction.cpu().numpy()             # (1,1)
        prediction_inverse = scaler_y.inverse_transform(prediction_np)
        return {"predicted_traffic_volume": float(prediction_inverse[0, 0])}
