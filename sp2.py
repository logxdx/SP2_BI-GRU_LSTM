import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# Create sequences for time-series data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

# Hybrid BiGRU-LSTM Model
class HybridBiGRU_LSTM(nn.Module):
    def __init__(self, input_size, gru_hidden_size, lstm_hidden_size1, lstm_hidden_size2, dropout_rate=0.2):
        super(HybridBiGRU_LSTM, self).__init__()
        # BiGRU layer
        self.bi_gru = nn.GRU(input_size, gru_hidden_size, batch_first=True, bidirectional=True)
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(gru_hidden_size * 2, lstm_hidden_size1, batch_first=True)
        
        # Second and third LSTM layers
        self.lstm2 = nn.LSTM(lstm_hidden_size1, lstm_hidden_size2, batch_first=True)
        self.lstm3 = nn.LSTM(lstm_hidden_size2, lstm_hidden_size2, batch_first=True)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Dense (fully connected) layer
        self.fc = nn.Linear(lstm_hidden_size2, input_size)

    def forward(self, x):
        # BiGRU layer
        x, _ = self.bi_gru(x)
        
        # First LSTM layer
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        # Third LSTM layer
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        
        # Fully connected layer (using only the last time step output)
        x = self.fc(x[:, -1, :])
        return x

# Train the Model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device="cpu"):
    model.to(device)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss", marker='o')
    plt.plot(history["val_loss"], label="Validation Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    return model, history

# Plot Predictions
def plot_predictions(y_train, y_pred_train, y_val, y_pred_val, feature_names=None):
    # Determine number of features
    num_features = y_train.shape[1] if len(y_train.shape) > 1 else 1
    
    # Create subplot grid
    fig, axes = plt.subplots(num_features, 1, figsize=(16, 4*num_features), sharex=True)
    
    # If only one feature, convert axes to list for consistent indexing
    if num_features == 1:
        axes = [axes]
    
    # Ensure feature names
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(num_features)]
    
    # Plot for each feature
    for i in range(num_features):
        # Extract i-th feature
        train_true = y_train[:,i] if num_features > 1 else y_train
        train_pred = y_pred_train[:,i] if num_features > 1 else y_pred_train
        val_true = y_val[:,i] if num_features > 1 else y_val
        val_pred = y_pred_val[:,i] if num_features > 1 else y_pred_val
        
        # Plot on corresponding subplot
        axes[i].plot(range(len(train_true)), train_true, label="Train True", alpha=0.7)
        axes[i].plot(range(len(train_true)), train_pred, label="Train Pred", alpha=0.7)
        axes[i].plot(range(len(train_true), len(train_true) + len(val_true)), val_true, label="Val True", alpha=0.7)
        axes[i].plot(range(len(train_true), len(train_true) + len(val_true)), val_pred, label="Val Pred", alpha=0.7)
        
        axes[i].set_title(f"{feature_names[i]} - True vs Predicted")
        axes[i].set_ylabel("Value")
        axes[i].set_xlabel("Time")
        axes[i].legend()
        axes[i].grid()
    
    # Set common x-label
    axes[-1].set_xlabel("Time")
    
    plt.tight_layout()
    plt.show()
