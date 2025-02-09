import numpy as np

class RealTimePredictor:
    def __init__(self, model, window_size):
        self.model = model
        self.window_size = window_size
        self.buffer = []
        
    def add_measurement(self, measurements):
        """Add new measurements (6 values)"""
        if len(self.buffer) >= self.window_size:
            self.buffer.pop(0)
        self.buffer.append(measurements)
        
    def predict(self):
        if len(self.buffer) < self.window_size:
            return 0  # Not enough data
        input_data = np.array([self.buffer])
        return self.model.predict(input_data)[0]