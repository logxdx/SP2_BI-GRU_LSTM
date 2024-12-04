# Hybrid Bi{GRU-LSTM} Time Series Prediction Model

This project implements a **Hybrid BiGRU-LSTM Neural Network** using PyTorch for multivariate time-series prediction. The model combines Bidirectional GRU and stacked LSTM layers to effectively capture sequential and temporal dependencies.

---

## 1. **Introduction**

The hybrid **BiGRU-LSTM model** is designed to predict stock market trends, leveraging:
- **GRU**: Efficient, memory-optimized, and faster training.
- **LSTM**: Handles long-term dependencies in sequential data.

This integration ensures accurate predictions while maintaining computational efficiency.

---

## 2. **Model Architecture**

### Components:
- **Bidirectional GRU (BiGRU)**:
  - Captures sequential patterns in both forward and backward directions.
  - Outputs feature representations of size `2 × gru_hidden_size`.
- **Stacked LSTM Layers**:
  - Three stacked LSTM layers process sequential data with increasing depth.
  - Dropout regularization is applied between LSTM layers.
- **Fully Connected (Dense) Layer**:
  - Maps the final LSTM output to the prediction space using only the last time-step output.

### Flow:
1. Input sequences are processed by the BiGRU layer.
2. BiGRU outputs are passed through three LSTM layers with dropout.
3. The output from the last LSTM is fed into a fully connected layer for final prediction.

### Hyperparameters:
- `gru_hidden_size`: Number of units in the GRU layer.
- `lstm_hidden_size1` & `lstm_hidden_size2`: Number of units in LSTM layers.
- `dropout_rate`: Dropout probability to reduce overfitting.
- `input_size`: Number of features in the input sequence.

---

## 3. **Data Loading and Preprocessing**

### Steps:
1. **Load the Dataset**:
   ```python
   import pandas as pd
   data = pd.read_csv('your_data.csv')
   ```
2. **Normalize Data**:
   ```python
   from sklearn.preprocessing import MinMaxScaler
   scaler = MinMaxScaler(feature_range=(0, 1))
   scaled_data = scaler.fit_transform(data)
   ```
3. **Create Input Sequences**:
   ```python
   seq_length = 100
   X, y = create_sequences(scaled_data, seq_length)
   ```
4. **Split Data**:
   ```python
   split_idx = int(len(X) * 0.8)
   X_train, y_train = X[:split_idx], y[:split_idx]
   X_val, y_val = X[split_idx:], y[split_idx:]
   ```
5. **Create DataLoaders**:
   ```python
   from torch.utils.data import DataLoader, TensorDataset
   train_dataset = TensorDataset(X_train, y_train)
   val_dataset = TensorDataset(X_val, y_val)
   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
   ```

---

## 4. **Training the Model**

### Steps:
1. **Initialize the Model**:
   ```python
   model = HybridBiGRU_LSTM(input_size, gru_hidden_size, lstm_hidden_size1, lstm_hidden_size2, dropout_rate)
   ```
2. **Set the Loss Function and Optimizer**:
   ```python
   import torch.nn as nn
   criterion = nn.MSELoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   ```
3. **Train**:
   ```python
   num_epochs = 20
   trained_model, loss_history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
   ```

---

## 5. **Evaluate and Visualize Predictions**

1. **Generate Predictions**:
   ```python
   model.eval()
   with torch.no_grad():
       y_pred_train = model(X_train.to(device)).cpu().numpy()
       y_pred_val = model(X_val.to(device)).cpu().numpy()
   ```
2. **Plot True vs Predicted Values**:
   ```python
   plot_predictions(y_train.numpy(), y_pred_train, y_val.numpy(), y_pred_val, feature_names=['open', 'high', 'low', 'close'])
   ```

---

## 6. **Future Enhancements**

- Incorporating attention mechanisms.
- Utilizing ensemble techniques for further accuracy improvements.

---

## Reference
This model is based on the research paper: [SMP-DL: A Novel Stock Market Prediction Approach Based on Deep Learning](https://rdcu.be/d2mmW).
