import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers

# Generate synthetic data (e.g., a simple sine wave with some anomalies)
def generate_data(n_samples=1000):
    time = np.arange(0, n_samples)
    normal_data = np.sin(0.02 * time)  # Simple sine wave
    anomaly_data = normal_data.copy()
    anomaly_indices = np.random.choice(np.arange(n_samples), size=50, replace=False)
    anomaly_data[anomaly_indices] += np.random.normal(0, 0.5, size=anomaly_indices.shape)
    
    return normal_data, anomaly_data

# Prepare the dataset
def prepare_dataset(data, window_size=20):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
    return np.array(X)

# Build the autoencoder model
def build_autoencoder(input_dim):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_dim,), activity_regularizer=regularizers.l1(10e-5)),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(input_dim, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

# Train the autoencoder
def train_autoencoder(model, X_train, epochs=50, batch_size=32):
    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    return history

# Detect anomalies
def detect_anomalies(model, X, threshold=0.01):
    reconstructions = model.predict(X)
    reconstruction_error = np.mean(np.abs(reconstructions - X), axis=1)
    anomalies = reconstruction_error > threshold
    return anomalies, reconstruction_error

# Plot the results
def plot_results(time, normal_data, anomaly_data, anomalies, reconstruction_error, threshold):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time, normal_data, label="Normal Data")
    plt.plot(time, anomaly_data, label="Anomalous Data", color='r')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(time[:-window_size], reconstruction_error, label="Reconstruction Error")
    plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time[:-window_size], anomalies, label="Anomalies", color='r')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == '__main__':
    n_samples = 1000
    window_size = 20

    # Generate data
    normal_data, anomaly_data = generate_data(n_samples)
    
    # Prepare the dataset
    X_train = prepare_dataset(normal_data, window_size)
    X_test = prepare_dataset(anomaly_data, window_size)
    
    # Build and train the autoencoder
    autoencoder = build_autoencoder(X_train.shape[1])
    train_autoencoder(autoencoder, X_train, epochs=100, batch_size=32)
    
    # Detect anomalies
    anomalies, reconstruction_error = detect_anomalies(autoencoder, X_test, threshold=0.02)
    
    # Plot the results
    time = np.arange(0, n_samples)
    plot_results(time, normal_data, anomaly_data, anomalies, reconstruction_error, threshold=0.02)
