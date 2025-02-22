import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
import helper_functions as hf
import matplotlib.pyplot as plt
from real_time_predictor import RealTimePredictor
import argparse

WINDOW_SIZE = 15

# Load data
def load_and_transform(network_voltage: float, network_current: float):
    inputs = pd.read_csv('dataset/inputs.csv')  # Shape: (samples, 6)
    outputs = pd.read_csv('dataset/outputs.csv')  # Shape: (samples, 1)

    data = pd.concat([inputs, outputs], axis=1).dropna()
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values


    X[:, :3] /= network_voltage
    X[:, 3:] /= network_current

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return (X_scaled, y)

def rt_sim(X_scaled, y, file):
    # First, build the model architecture
    # change the build function according to your model
    model = hf.build_optimized_model(input_shape=(WINDOW_SIZE, 6), num_classes=8)
    # Then load the weights
    model.load_weights(f'trained_models/{file}.weights.h5')
    # Usage example:
    rt_predictor = RealTimePredictor(model, WINDOW_SIZE)

    # Set up real-time plotting
    # change start point as you want
    start = 70900
    num_of_samples = 25000 + start
    test_inputs = X_scaled[start:num_of_samples, :]
    test_outputs = y[start:num_of_samples]

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Real-Time Short Circuit Detection', fontsize=14)
    ax.set_xlabel('Sample Number', fontsize=14)
    ax.set_ylabel('Fault Class', fontsize=14)

    true_line, = ax.plot([], [], 'b-', label='True Output')
    pred_line, = ax.plot([], [], 'r--', label='Predicted Output')
    ax.legend(fontsize=12)
    x_data = []
    true_y = []
    pred_y = []

    # Set plot limits
    ax.set_xlim(0, 10000)
    ax.set_ylim(-0.5, 7.5)  # Adjust based on your class range

    # Process samples and update plot
    for i in range(len(test_inputs)):
        # Add current measurement
        rt_predictor.add_measurement(test_inputs[i,:])
        
        # Make prediction if buffer is full
        prediction = rt_predictor.predict()
        
        if prediction is not None and i < len(test_outputs) - 1:
            # Get predicted class
            predicted_class = np.argmax(prediction)
            true_class = test_outputs[i + 1]  # Offset by 1 for temporal alignment
            
            # Store results
            x_data.append(i)
            true_y.append(true_class)
            pred_y.append(predicted_class)
            
            # Update plot every 100 samples
            if i % 100 == 0:
                # Keep only last 1000 points for visibility
                keep_points = 10000
                true_line.set_xdata(x_data[-keep_points:])
                true_line.set_ydata(true_y[-keep_points:])
                pred_line.set_xdata(x_data[-keep_points:])
                pred_line.set_ydata(pred_y[-keep_points:])
                
                # Adjust view
                ax.set_xlim(max(0, i - keep_points), i + 50)
                fig.canvas.draw()
                fig.canvas.flush_events()

    plt.ioff()
    plt.show()

    # Save final results
    np.save(f'true_predictions_{file}.npy', true_y)
    np.save(f'predicted_predictions_{file}.npy', pred_y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("-f", "--file", type=str, default="model_window_size_10", help="Input file")

    args = parser.parse_args()

    print(f"File: {args.file}")

    X_scaled, y = load_and_transform(network_voltage=220, network_current=25)
    rt_sim(X_scaled, y, args.file)
