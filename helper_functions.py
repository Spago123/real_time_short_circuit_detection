import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def create_sequences(data, targets, window_size):
    X_seq, y_seq = [], []
    for i in range(len(data) - window_size):
        X_seq.append(data[i:i+window_size])
        y_seq.append(targets[i+window_size])  # Predict next state
    return np.array(X_seq), np.array(y_seq)

def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_optimized_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, return_sequences=False),  # Reduced units
        Dropout(0.2),
        Dense(16, activation='relu'),  # Smaller dense layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Use a lower learning rate for stability
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_single_layer_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        GRU(16, return_sequences=False),  # Single GRU layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_minimum_single_layer_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        GRU(8, return_sequences=False),  # Single GRU layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_ekstra_minimum_single_layer_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        GRU(2, return_sequences=False),  # Single GRU layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model