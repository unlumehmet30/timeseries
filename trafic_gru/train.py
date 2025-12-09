import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#hiperparameters
batch_size = 64 
input_size = 7  # Number of features
hidden_size = 64
num_layers = 2
output_size = 1

# Load preprocessed data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
x_train=torch.tensor(X_train,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.float32)
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



class GRUNet(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out,_= self.gru(x)
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)
        return out
model=GRUNet(input_size,hidden_size,num_layers,output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
losts_list = []
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    losts_list.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
# %% 
plt.figure()
plt.plot(losts_list,marker="o")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid()
plt.tight_layout()
plt.savefig('training_loss.png')
plt.show()

# Save the trained model
torch.save(model.state_dict(), 'gru_model.pth')
 