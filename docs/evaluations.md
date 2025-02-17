# Emotions Classification on RAVDESS

### pure dataset, no preprocessing

base torch - 200 epok | Train Acc: 0.4399 | Val Acc: 0.4269 | Test Accuracy: 0.4861

```class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        
        # Spatial feature extraction using Conv1D
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # LSTM layers for temporal feature extraction
        self.lstm1 = nn.LSTM(input_size=32 * 239, hidden_size=128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128 * 2, hidden_size=64, batch_first=True)
        
        # Fully connected classification layer
        self.fc = nn.Linear(64, 8)  # 8 emotion classes

    def forward(self, x):
        # x shape: (batch_size, frames, landmarks, coordinates)
        batch_size, frames, landmarks, coordinates = x.shape
        
        # Reshape for Conv1D: (batch_size * frames, landmarks, coordinates)
        x = x.view(-1, landmarks, coordinates).permute(0, 2, 1)
        
        # Spatial feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Flatten spatial features
        x = x.view(batch_size, frames, -1)  # (batch_size, frames, features)
        
        # Temporal feature extraction
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Classification
        x = self.fc(x[:, -1, :])  # Take the last timestep's output
        return x
```

seglearn - | Train Acc: x | Val Acc: 0.64 | Test Accuracy: 0.62
todynet - | Train Acc: 57.0079 | Val Acc: 48.4919 | Test Accuracy: x