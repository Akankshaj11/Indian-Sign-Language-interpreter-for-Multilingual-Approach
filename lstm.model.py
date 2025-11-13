from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm(num_classes):
    model = Sequential([
        LSTM(128, input_shape=(30,63)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model
